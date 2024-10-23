#!/usr/bin/env python
"""
Inference script.

To run with base.yaml as the config,

> python run_inference.py

To specify a different config,

> python run_inference.py --config-name symmetry

where symmetry can be the filename of any other config (without .yaml extension)
See https://hydra.cc/docs/advanced/hydra-command-line-flags/ for more options.

"""

import re
import os, time, pickle
import torch
from omegaconf import OmegaConf
import hydra
import logging
from rfdiffusion.util import writepdb_multi, writepdb
from rfdiffusion.inference import utils as iu
from hydra.core.hydra_config import HydraConfig
import numpy as np
import random
import glob
from tqdm import tqdm


def make_deterministic(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


@hydra.main(version_base=None, config_path="../config/inference", config_name="base")
def main(conf: HydraConfig) -> None:
    log = logging.getLogger(__name__)
    if conf.inference.deterministic:
        make_deterministic()

    # Check for available GPU and print result of check
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(torch.cuda.current_device())
        log.info(f"Found GPU with device_name {device_name}. Will run RFdiffusion on {device_name}")
    else:
        log.info("////////////////////////////////////////////////")
        log.info("///// NO GPU DETECTED! Falling back to CPU /////")
        log.info("////////////////////////////////////////////////")

    # Initialize sampler and target/contig.
    sampler = iu.sampler_selector(conf)

    input_path = os.path.expanduser(conf.inference.data_path)
    output_path = os.path.expanduser(conf.inference.output_path)
    os.makedirs(output_path, exist_ok=True)
    num_split = conf.num_split
    split_id = conf.split_id
    pdbs = sorted(os.listdir(input_path))
    size = len(pdbs) // num_split + 1
    pdbs = pdbs[size * split_id: size * (split_id + 1)]
    for pdb in tqdm(pdbs):
        pdb_path = os.path.join(input_path, pdb)
        pdb_name = pdb[:-4]
        pkl_path = os.path.join(output_path, pdb_name+".pt")
        if os.path.exist(pkl_path):
            continue
        states = sampler.extract_representation(pdb_path)
        output_dict = {'label': pdb_name, 'mean_representations': states}
        torch.save(output_dict, pkl_path)


if __name__ == "__main__":
    main()
