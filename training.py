import yaml
import wandb
import torch
import argparse
import numpy as np
import typing as tp
import torch.nn as nn

from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from modelling.autoencoders import TimeDomainAE, FrequencyDomainAE
from modelling.utils import get_models, get_optmizers, TModels, TOptimizers
from data.random_data import jumping_mean_data, scaling_variance_data, changing_coefficients_data, gaussians_mixtures_data
from data.data_preparation import get_dataset, get_dataloader, TimeSeriesDataset


def init_wandb_project(config: tp.Dict[str, tp.Any]) -> None:
    wandb.init(
        project="Structural_Breaks_Identification",
        config=config,
        save_code=False,
    )


def time_domain_model_batch_step(model: TimeDomainAE, optimizer: Adam, td_input: torch.Tensor, lambda_coef: float, k: int, device: str) -> float:
    td_input = td_input.to(device)
    model.zero_grad()

    reconstruction, time_inv_feat, _ = model(td_input)
    loss: torch.Tensor = nn.functional.mse_loss(reconstruction[k :: k + 1], td_input[k :: k + 1])

    for idx in range(0, len(time_inv_feat), k + 1):
        loss += lambda_coef * nn.functional.mse_loss(time_inv_feat[idx : idx + k + 1][1:], time_inv_feat[idx : idx + k + 1][:-1])

    loss.backward()
    optimizer.step()
    return loss.item()


def frequency_domain_model_batch_step(model: FrequencyDomainAE, optimizer: Adam, fd_input: torch.Tensor, lambda_coef: float, k: int, device: str) -> float:
    fd_input = fd_input.to(device)
    model.zero_grad()

    reconstruction, time_inv_feat, _ = model(fd_input)
    loss: torch.Tensor = nn.functional.mse_loss(reconstruction[k :: k + 1], fd_input[k :: k + 1])
    for idx in range(len(time_inv_feat), k + 1):
        loss += lambda_coef * nn.functional.mse_loss(time_inv_feat[idx : idx + k + 1][1:], time_inv_feat[idx : idx + k + 1][:-1])

    loss.backward()
    optimizer.step()
    return loss.item()


def train_batch_step(models: TModels, optimizers: TOptimizers, td_data: torch.Tensor, fd_data: torch.Tensor, config: tp.Dict[str, tp.Any], epoch: int) -> None:
    td_loss: float = time_domain_model_batch_step(
        model=models["td"],
        optimizer=optimizers["td"],
        td_input=td_data,
        lambda_coef=config["training"]["lambda_td"],
        k=config["training"]["k"],
        device=config["training"]["device"],
    )
    fd_loss: float = frequency_domain_model_batch_step(
        model=models["fd"],
        optimizer=optimizers["fd"],
        fd_input=fd_data,
        lambda_coef=config["training"]["lambda_fd"],
        k=config["training"]["k"],
        device=config["training"]["device"],
    )
    wandb.log({"time_domain_loss": td_loss, "frequency_domain_loss": fd_loss, "epoch": epoch})


def train_epoch(models: TModels, optimizers: TOptimizers, dataloader: DataLoader, config: tp.Dict[str, tp.Any], epoch: int):
    models["td"].train()
    models["fd"].train()

    for td_data, fd_data in dataloader:
        train_batch_step(models, optimizers, td_data, fd_data, config, epoch)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_type", default="jm")
    return parser.parse_args()


def main():
    args = parse_arguments()
    config: tp.Dict[str, tp.Any] = yaml.safe_load(open("config.yaml"))
    init_wandb_project(config)

    models: TModels = get_models(config)
    optimizers: TOptimizers = get_optmizers(models, config)

    if args.data_type == "jm":
        ts_data, _ = jumping_mean_data()
    elif args.data_type == "sv":
        ts_data, _ = scaling_variance_data()
    elif args.data_type == "cc":
        ts_data, _ = changing_coefficients_data()
    elif args.data_type == "gm":
        ts_data, _ = gaussians_mixtures_data()
    dataloader: DataLoader = get_dataloader(ts_data, config)

    for epoch in tqdm(range(config["training"]["num_epochs"])):
        train_epoch(models, optimizers, dataloader, config, epoch)
    torch.save(models, "./checkpoints/checkpoint.pth")


if __name__ == "__main__":
    main()