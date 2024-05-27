import torch
from cbx.dynamics import CBO
from cbx.utils.torch_utils import norm_torch, compute_consensus_torch, normal_torch,effective_sample_size
import torch
import torch.nn as nn

class adv_loss:
    def __init__(self, model, y=1):
        self.model = model
        self.y = y
        
    def __call__(self, x):
        s = x.shape
        return ((self.model(x.view(-1, 2)) - self.y).squeeze(dim=-1).abs()**2).view(s[:-1])
    
class pclamp:
    def __init__(self, x0, x_old, tau=0.1, eps=0.1):
        self.x0 = x0
        self.x_old = x_old
        self.tau=tau
        self.eps=eps
    
    def __call__(self, dyn):
        dyn.x = self.x_old + torch.clamp(dyn.x - self.x_old, min=-self.tau, max=self.tau)
        dyn.x = self.x0 + torch.clamp(dyn.x - self.x0, min=-self.eps, max=self.eps)
    
def get_cbo(x0, x_old, E, eps=1., 
            tau=0.1, device='cpu',
            N = 50,
            **kwargs):

    x = x0 + torch.zeros(size=(1,N,2), device=device).uniform_(-tau, tau)
    dyn = CBO(lambda y: -E(y), x = x, f_dim='3D', 
              copy=torch.clone, norm=norm_torch,
              post_process=pclamp(x0, x_old, tau=tau, eps=eps),
              compute_consensus = compute_consensus_torch,
              normal=normal_torch(device), check_f_dims=False,
              N = N,
              **kwargs
             )
    return dyn

def reset_cbo(dyn, x_old):
    tau, eps = (dyn.post_process.tau, dyn.post_process.eps)
    dyn.it = 0
    dyn.post_process.x_old = x_old.clone()
    dyn.x = x_old.clone() + torch.zeros(size=(1,dyn.N,2), device=dyn.x.device).uniform_(-min(tau,eps), min(tau,eps))
    dyn.post_process(dyn)
    dyn.x_old = None

def projec_linfty(x0, x, eps=.1):
    return x0 + torch.clamp(x-x0, min=-eps, max=eps)

class optimizer:
    def optimize(self, max_iter=10):
        self.it = 0
        while (self.it < max_iter):# or (torch.max(torch.abs(self.x-self.x0)) < self.epsilon * 0.95):
            self.step()
            self.it += 1

class FGSM(optimizer):
    def __init__(self, x0, E, tau=0.05, epsilon=0.1):
        self.tau = tau
        self.epsilon = epsilon
        self.x0 = x0
        self.x = nn.Parameter(x0.clone())
        self.E = E
        self.hist = [self.x.detach().cpu()]
        
    def step(self,):
        self.x.grad = None
        loss = self.E(self.x)
        grad = torch.autograd.grad(loss, self.x)[0]
        self.x = self.project(self.x + self.tau * torch.sign(grad))
        self.hist.append(self.x.detach().cpu().clone())
        
    def project(self, x, eps=None):
        return projec_linfty(self.x0, x, eps=self.epsilon)
    
class MinMove(optimizer):
    def __init__(self, x0, E, tau=0.05, epsilon=0.1, max_inner_it=100, N=50,**kwargs):
        self.tau = tau
        self.epsilon = epsilon
        self.x0 = x0
        self.x = x0.clone()
        self.E = E
        self.hist = [self.x.detach().cpu()]
        self.max_inner_it=max_inner_it
        self.N = N
        self.dyn = get_cbo(
            self.x0[None,...], self.x0[None,...].clone(), self.E, 
            tau=self.tau, eps=self.epsilon, 
            max_it = self.max_inner_it,
            device=self.x0.device, verbosity=0, N=self.N,
            **kwargs)
    
    @torch.no_grad()
    def step(self,):
        x_old = self.x.clone()
        reset_cbo(self.dyn, x_old)
        self.dyn.optimize(sched=None)
        self.x = self.dyn.best_particle.clone()
       
        # save cuda memory
        torch.cuda.empty_cache()
        self.hist.append(self.x.detach().cpu().clone())
        