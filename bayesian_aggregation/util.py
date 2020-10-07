# Copyright (c) 2020
# Copyright holder of the paper "Bayesian Context Aggregation for Neural Processes".
# Submitted to ICLR 2021 for review.
# All rights reserved.

from data.gp import Matern52GPBenchmark, RBFGPBenchmark, WeaklyPeriodicGPBenchmark

# dictionary containing benchmarks
benchmark_dict = {
    "RBFGP": RBFGPBenchmark,
    "Matern52GP": Matern52GPBenchmark,
    "WeaklyPeriodicGP": WeaklyPeriodicGPBenchmark,
}
