# Adversarial Flows

![image](https://github.com/TimRoith/AdversarialFlows/assets/44805883/bd11042d-9a2c-4c60-9fd5-9a2882f9299a)


This repository provides the code for the experiments in the paper "Adversarial Flows: A gradient flow characterization
of adversarial attacks" by Lukas Weigand, Tim Roith, and Martin Burger.

## Overview 

The following Notebooks are available:

* ``TrainClassifier.ipynb``: This notebook trains a simple fully connected neural network, defined in ``model.py``, on data points sampled from the two-moons dataset. The concrete dataset used in the paper is provided in the folder data-weights as ``two_moons.npz``. Furthermore the weights of the trained networks are also given as ``two_moons_ReLU.pt`` and ``two_moons_GeLU.pt`` 

<p float="left">
  <img src="https://github.com/TimRoith/AdversarialFlows/assets/44805883/b7643a1a-ee5c-4ddf-8afc-0fb68bb24b87" width="48%" />
  <img src="https://github.com/TimRoith/AdversarialFlows/assets/44805883/cd16232b-72f9-4e0d-a136-eeb391a2703e" width="48%" /> 
</p>

* ``Flows.ipynb``: This notebook computes and visualizes the iterates of IFGSM and the minizing movement scheme. The optimizers are defined in ``flows.py``

Furthermore, the file ``compute_diffs.py`` provides an executable script, computing the difference between IFGSM and the minimizing movement scheme for different initial vlaues.


## Citation
