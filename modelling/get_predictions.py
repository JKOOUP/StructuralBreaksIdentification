import torch
import numpy as np
import typing as tp
import torch.nn as nn


from data.data_preparation import get_test_dataloader
from data.post_processing import calculate_prominence_measure
from data.post_processing import calc_dissimilarity_measures_for_td_and_fd
from modelling.autoencoders import TimeDomainAE, FrequencyDomainAE


def predict_change_points(ts_data: np.ndarray, model_checkpoint_path: str, config: tp.Dict[str, tp.Any]) -> np.ndarray:
    models = torch.load(model_checkpoint_path)
    dataloader = get_test_dataloader(ts_data, config)
    
    td_preds = []
    fd_preds = []
    with torch.no_grad():
        for td_data, fd_data in dataloader:
            _, td_features, _ = models["td"](td_data.to(config["training"]["device"]))
            _, fd_features, _ = models["fd"](fd_data.to(config["training"]["device"]))
            td_preds.append(td_features.detach().cpu())
            fd_preds.append(fd_features.detach().cpu())
    
    td_preds = torch.cat(td_preds, dim=0).numpy()
    fd_preds = torch.cat(fd_preds, dim=0).numpy()
    dissimilarity_measures: np.ndarray = calc_dissimilarity_measures_for_td_and_fd(td_preds, fd_preds, config["modelling"]["window_size"])
    prominence: np.ndarray = calculate_prominence_measure(dissimilarity_measures, config["modelling"]["window_size"])
    return prominence
