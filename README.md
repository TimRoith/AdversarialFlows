# Adversarial Flows

This repository provides the code for the experiments in the paper "Adversarial Flows: A gradient flow characterization
of adversarial attacks" by Lukas Weigand, Tim Roith, and Martin Burger.

# Overview 

The following Notebooks are available:

* ``TrainClassifier.ipynb``: This notebook trains a simple fully connected neural network, on data points sampled from the two-moons dataset. The concrete dataset used in the paper is provided in the folder data-weights as ``two_moons.npz``. Furthermore the weights of the trained networks are also given as ``two_moons_ReLU.pt`` and ``two_moons_GeLU.pt`` 

<p float="left">
  <img src="https://github.com/TimRoith/AdversarialFlows/assets/44805883/fa2ceb1e-dd88-4835-a3a0-c28a3f3784c2" width="500" />
  <img src="https://github.com/TimRoith/AdversarialFlows/assets/44805883/cd16232b-72f9-4e0d-a136-eeb391a2703e" width="500" /> 
</p>

Code for the Paper "Adversarial Flows"
