# ---------------------------------------------------------
# This script computes the difference between FGSM and the 
# minizing movement scheme for randomly sampled inital
# values

import torch
from flows import adv_loss, get_cbo, FGSM, MinMove
from plots import plot_rect_budget, plot_fun
import numpy as np
from datetime import datetime
from model import get_two_moons_model
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser(description='Process arguments to FGSM-MinMove diff computation.')
    parser.add_argument('--samples', '--s', dest='num_samples', type=int, default=1, help='Number of inital values.')
    parser.add_argument('--taus', '--t', dest='num_taus', type=int, default=1, help='Number of taus to compute.')
    
    return parser.parse_args()

def load_model(act_fun='ReLU'):
    model = get_two_moons_model(act_fun=act_fun)
    model.load_state_dict(torch.load('data-weights/two_moons_' + act_fun + '.pt'))
    model.eval()
    return model

def main():
    args = parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Operating on device: ' + str(device), flush=True)

    for act_fun in ['ReLU', 'GeLU']:
        model = load_model(act_fun=act_fun)
        model.to(device)
        
        print(30*'-')
        print('Startin for activation function: ' + act_fun)

        epsilon=.2
        x0 = torch.zeros((args.num_samples,1,2)).uniform_(-1,1).to(device)
        y = 1. * (model(x0[:,0,:]) > 0.5)

        opts = [{'opt':FGSM, 'kwargs':{}}, {'opt':MinMove, 'kwargs':{'max_inner_it':30, 'N':30}}]
        T = 1

        diffs = np.zeros(args.num_taus)
        taus = np.array([epsilon * 0.5**i for i in range(args.num_taus)])
        print('Computing for taus: ' + str(taus),flush=True)

        for i,tau in enumerate(taus):
            print('Starting for tau = ' + str(tau), flush=True)
            diff = 0
            for s in range(x0.shape[0]):
                hists = []
                for o in opts:
                    E = adv_loss(model, y=y[s])
                    opt = o['opt'](x0[s,...], E, epsilon=epsilon, tau=tau, **o['kwargs'])
                    opt.optimize(max_iter=int(T/tau))
                    hists.append(torch.stack(opt.hist).squeeze())
                diff += torch.linalg.norm(hists[0] - hists[1], ord=float('inf'))
            print('Finished tau = ' +str(tau),flush=True)
            diffs[i] = (diff/x0.shape[0]).item()
            print('Diff: ' + str(diffs[i]),flush=True)

        print('Finished all',flush=True)

        np.savetxt('results/diffs/FGSM-MinMove-' + act_fun + '-' + str(datetime.now()) + '.txt', np.stack([taus, diffs]))
    
if __name__ == "__main__":
    main()
