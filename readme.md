# ICLR 2021 Submission No. 637
Dear reviewers,

this is the source code accompanying the ICLR 2021 Submission No. 637 "Bayesian Context Aggregation for Neural Processes".

## Installation
We kindly ask you to clone this repository and run

`conda env create -f environment.yml`

to create a new conda environment named `bayesian_aggregation` with all python packages required to run the experiments.

## Contents
We provide a script `train_evaluate_gps.py` to reproduce results from the GP-suite of experiments presented in the paper within error bounds.
This script trains the specified model for 200 epochs, computes the predictive likelihood, and plots predictions.
Weights are stored and re-used if this script is executed multiple times.

The script is called from the command line as follows:

`python ./evaluate_gps.py EXPERIMENT AGGREGATOR LIKELIHOOD_APPROXIMATION [-h]`

Description of the arguments:
1. `EXPERIMENT (str)`: the name of the experiment. Allowed values are:
`EXP_NAME = {"RBFGP" | "WeaklyPeriodicGP" | "Matern52GP"}`
2. `AGGREGATOR (str)`: the aggregator to use. Allowed values are:
`AGGREGATOR = {"BA" | "MA"}`
3. `LIKELIHOOD_APPROXIMATION (str)`: the likelihood approximation to use. Allowed values are:
`LIKELIHOOD_APPROXIMATION = {"PB" | "VI_inspired" | "MC"}`

The combination "MA+PB" corresponds to the Conditional Neural Process (Garnelo et al., "Conditional Neural Processes", ICML 2018).

The combination "MA+VI_inspired" corresponds to the Neural Process (Garnelo et al., "Neural Processes", ICML 2018 Workshop on Theoretical Foundations and Applications of Deep Generative Models) without
a deterministic path.


## Copyright
Copyright (c) 2021

Copyright holder of the paper "Bayesian Context Aggregation for Neural Processes".

Submitted to ICLR 2021 for review.

All rights reserved.
