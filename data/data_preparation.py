import torch
import numpy as np
import typing as tp
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader


def construct_time_domain_windows(ts_data: np.ndarray, window_size: int) -> np.ndarray:
    time_domain_windows: tp.List[np.ndarray] = []
    for idx in range(window_size, ts_data.shape[0]):
        time_domain_windows.append(ts_data[idx - window_size : idx])
    return np.stack(time_domain_windows)


def construct_frequency_domain_windows(ts_data: np.ndarray, window_size: int, dft_shape: int) -> np.ndarray:
    windows: np.ndarray = construct_time_domain_windows(ts_data, window_size)
    windows = windows - windows.mean()
    windows_fft: np.ndarray = np.abs(np.fft.fft(windows, dft_shape))[..., :dft_shape // 2 + 1]
    return windows_fft


def rescale_data(data: np.ndarray) -> np.ndarray:
    data = (data - data.min()) / (data.max() - data.min())
    return 2.0 * data - 1.0


class TimeSeriesDataset(Dataset):
    def __init__(self, ts_data: np.ndarray, window_size: int, dft_shape: int):
        super().__init__()

        # Is rescaling applied over all data statistics or over window statistics only?
        self.time_domain_data: np.ndarray = rescale_data(construct_time_domain_windows(ts_data, window_size))
        self.frequency_domain_data: np.ndarray = rescale_data(construct_frequency_domain_windows(ts_data, window_size, dft_shape))

    def __len__(self) -> int:
        return len(self.time_domain_data)

    def __getitem__(self, index) -> tp.Tuple[torch.Tensor, ...]:
        return (
            torch.tensor(self.time_domain_data[index], dtype=torch.float32),
            torch.tensor(self.frequency_domain_data[index], dtype=torch.float32),
        )


def get_dataset(ts_data: np.ndarray, config: tp.Dict[str, tp.Any]) -> TimeSeriesDataset:
    window_size: int = config["modelling"]["window_size"]
    dft_shape: int = config["modelling"]["dft_shape"]
    return TimeSeriesDataset(ts_data, window_size, dft_shape)


class CustomBatchSampler:
    def __init__(self, dataset: TimeSeriesDataset, batch_size: int, k: int):
        self.dataset: TimeSeriesDataset = dataset
        self.batch_size: int = batch_size
        self.k: int = k

    def __len__(self):
        return len(self.dataset - self.k) // self.batch_size

    def __iter__(self):
        indices: tp.List[int] = np.array(range(self.k, len(self.dataset)))
        np.random.shuffle(indices)

        for start_idx in range(self.batch_size, len(indices), self.batch_size):
            end_idx = start_idx + self.batch_size
            yield np.stack([indices[start_idx : end_idx] - elem for elem in range(self.k, -1, -1)]).T.reshape(-1).tolist()


def collate_function(batch: tp.List[tp.Tuple[torch.Tensor, ...]]) -> tp.Tuple[torch.Tensor, ...]:
    td_batch: torch.Tensor = torch.stack([elem[0] for elem in batch])
    fd_batch: torch.Tensor = torch.stack([elem[1] for elem in batch])
    return td_batch, fd_batch


def get_dataloader(ts_data: np.ndarray, config: tp.Dict[str, tp.Any]) -> DataLoader:
    dataset: TimeSeriesDataset = get_dataset(ts_data, config)
    dataloader: DataLoader = DataLoader(
        dataset,
        batch_sampler=CustomBatchSampler(dataset, batch_size=config["training"]["batch_size"], k=config["training"]["k"]),
        collate_fn=collate_function,
        pin_memory=True,
        pin_memory_device=config["training"]["device"],
    )
    return dataloader

def get_test_dataloader(ts_data: np.ndarray, config: tp.Dict[str, tp.Any]) -> DataLoader:
    ts_data = np.pad(ts_data, pad_width=(0, config["modelling"]["window_size"]), mode="reflect")
    dataset: TimeSeriesDataset = get_dataset(ts_data, config)
    dataloader: DataLoader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        collate_fn=collate_function,
        shuffle=False,
        pin_memory=True,
        pin_memory_device=config["training"]["device"],
    )
    return dataloader
