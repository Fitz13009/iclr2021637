# Copyright (c) 2020
# Copyright holder of the paper "Bayesian Context Aggregation for Neural Processes".
# Submitted to ICLR 2021 for review.
# All rights reserved.

import torch
from torch.utils.data import IterableDataset

from bayesian_aggregation.util import benchmark_dict


class NPIterableDataset(IterableDataset):
    def __init__(self, benchmark_name, n_task, n_data_per_task, seed):
        super(NPIterableDataset).__init__()
        self.benchmark = benchmark_dict[benchmark_name](
            n_data_per_task=n_data_per_task, seed=seed
        )
        self.n_task = n_task

    def __iter__(self):
        for _ in range(self.n_task):
            x, y = self.benchmark.generate_one_task()
            yield torch.Tensor(x), torch.Tensor(y)
