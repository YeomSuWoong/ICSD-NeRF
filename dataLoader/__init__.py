# Copyright (c) Meta Platforms, Inc. and affiliates.

from .dynamic_video import DavisDataset, NvidiaDataset

dataset_dict = {"nvidia": NvidiaDataset, "davis": DavisDataset}
