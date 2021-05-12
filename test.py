# ---------------------------------------------------------------------------
# Learning Lane Graph Representations for Motion Forecasting
#
# Copyright (c) 2020 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Written by Ming Liang, Yun Chen
# ---------------------------------------------------------------------------

import argparse
import os
os.umask(0)
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import pickle
import sys
from importlib import import_module

import torch
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from data import ArgoTestDataset
from utils import Logger, load_pretrain


root_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_path)


# define parser
parser = argparse.ArgumentParser(description="Argoverse Motion Forecasting in Pytorch")
parser.add_argument(
    "-m", "--model", default="lanegcn", type=str, metavar="MODEL", help="model name"
)
parser.add_argument("--eval", action="store_true", default=True)
parser.add_argument(
    "--split", type=str, default="test", help='data split, "val" or "test"'
)
parser.add_argument(
    "--weight", default="/home/carla_challenge/Desktop/francis/LaneGCN/36.000.ckpt", type=str, metavar="WEIGHT", help="checkpoint path"
)
parser.add_argument(
    "--map_path", default="/home/carla_challenge/Desktop/francis/Scenic/tests/formats/opendrive/maps/CARLA/Town05.xodr", type=str, help="absolute path to carla map"
)
parser.add_argument(
    "-w", "--worker_num", default=0, type=int, help="Parallel worker number"
)


def main():
    # Import all settings for experiment.
    args = parser.parse_args()
    model = import_module(args.model)
    config, _, collate_fn, net, loss, post_process, opt = model.get_model(args.worker_num)

    # load pretrain model
    ckpt_path = args.weight
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(config["save_dir"], ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    load_pretrain(net, ckpt["state_dict"])
    net.eval()

    # Data loader for evaluation
    dataset = ArgoTestDataset(args.split, config, args.map_path, train=False)
    data_loader = DataLoader(
        dataset,
        batch_size=config["val_batch_size"],
        num_workers=config["val_workers"],
        collate_fn=collate_fn,
        shuffle=True,
        pin_memory=True,
    )

    # begin inference
    preds = {}
    for ii, data in tqdm(enumerate(data_loader)):
        data = dict(data)
        with torch.no_grad():
            output = net(data)
            results = [x[0:1].detach().cpu().numpy() for x in output["reg"]]
        for idx, pred_traj in zip(data["argo_id"], results):
            preds[idx] = pred_traj.squeeze()

    # save predictions
    import csv
    for _, pred in preds.items():
        for i, mode in enumerate(pred):
            with open(f"{config['save_dir']}/predictions_{args.worker_num}_{i}.csv", 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['X', 'Y'])
                for row in mode:
                    writer.writerow(row)
            csvfile.close()

if __name__ == "__main__":
    main()
