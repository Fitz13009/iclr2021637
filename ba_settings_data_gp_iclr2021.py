# Copyright (c) 2020
# Copyright holder of the paper "Bayesian Context Aggregation for Neural Processes".
# Submitted to ICLR 2021 for review.
# All rights reserved.

import os

import yaml
from torch.utils.data import DataLoader

from data.dataset import NPIterableDataset


def get_settings_and_data(benchmark_name, aggregator, likelihood_approximation):
    assert benchmark_name in ["RBFGP", "WeaklyPeriodicGP", "Matern52GP"]
    assert aggregator in ["BA", "MA"]
    assert likelihood_approximation in ["PB", "VI_inspired", "MC"]

    # load settings determined by HPO
    path = os.path.dirname(os.path.realpath(__file__))
    path = os.path.join(path, "settings_iclr2021")
    fn = (
        "settings_" + benchmark_name + "_" + aggregator + "_" + likelihood_approximation
    )
    fn += ".yaml"

    with open(os.path.join(path, fn), "r") as f:
        settings = yaml.load(f, Loader=yaml.FullLoader)

    # data
    n_task_m = 256 * 16
    n_data_per_task_m = 128
    n_task_v = 256
    n_data_per_task_v = 128
    n_task_t = 256
    n_data_per_task_t = 256 + 64
    dataloader_m = DataLoader(
        dataset=NPIterableDataset(
            benchmark_name=benchmark_name,
            n_task=n_task_m,
            n_data_per_task=n_data_per_task_m,
            seed=settings["seed"],
        ),
        batch_size=settings["training"]["batch_size"],
    )
    dataloader_v = DataLoader(
        dataset=NPIterableDataset(
            benchmark_name=benchmark_name,
            n_task=n_task_v,
            n_data_per_task=n_data_per_task_v,
            seed=2 * settings["seed"],
        ),
        batch_size=settings["training"]["batch_size"],
    )
    dataloader_t = DataLoader(
        dataset=NPIterableDataset(
            benchmark_name=benchmark_name,
            n_task=n_task_t,
            n_data_per_task=n_data_per_task_t,
            seed=3 * settings["seed"],
        ),
        batch_size=settings["training"]["batch_size"],
    )

    return settings, dataloader_m, dataloader_v, dataloader_t
