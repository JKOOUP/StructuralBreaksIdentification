import numpy as np
import typing as tp
from scipy.signal import argrelmax 


def get_weights(window_size: int) -> np.ndarray:
    result: np.ndarray = np.array([k / window_size ** 2 for k in range(1, window_size + 1)])
    result = np.append(result, result[::-1][1:])
    return result


def apply_weightened_moving_average(vector: np.ndarray, weights: np.ndarray, padding: bool = True) -> np.ndarray:
    window_size: int = weights.shape[0] // 2
    result: tp.List[np.ndarray] = []

    if len(vector.shape) == 1:
        vector = vector.reshape(-1, 1)

    for channel_idx in range(vector.shape[1]):
        result.append(np.convolve(np.pad(vector[:, channel_idx], (window_size, window_size), mode="edge"), weights, mode="valid"))
    return np.stack(result).T


def calc_dissimilarity_measures(vector: np.ndarray, window_size: int) -> np.ndarray:
    vector = np.pad(vector, pad_width=((0, window_size), (0, 0)), mode="reflect")
    return np.square(vector[:-window_size] - vector[window_size:]).sum(axis=-1)


def calc_weights_of_td_and_fd_time_inv_features(td_features: np.ndarray, fd_features: np.ndarray, window_size: int) -> tp.Tuple[float, float]:
    weights: np.ndarray = get_weights(window_size)
    td_features = apply_weightened_moving_average(td_features, weights)
    fd_features = apply_weightened_moving_average(fd_features, weights)
    
    td_dissim_measure: np.ndarray = apply_weightened_moving_average(calc_dissimilarity_measures(td_features, window_size), weights)
    fd_dissim_measure: np.ndarray = apply_weightened_moving_average(calc_dissimilarity_measures(fd_features, window_size), weights)

    alpha: float = np.quantile(fd_dissim_measure, 0.95)
    beta: float = np.quantile(td_dissim_measure, 0.95)
    return alpha, beta


def calc_dissimilarity_measures_for_td_and_fd(td_features: np.ndarray, fd_features: np.ndarray, window_size: int) -> np.ndarray:
    alpha, beta = calc_weights_of_td_and_fd_time_inv_features(td_features, fd_features, window_size)
    features: np.ndarray = np.concatenate((alpha * td_features, beta * fd_features), axis=1)
    weights: np.ndarray = get_weights(window_size)
    dissimilarity_measures: np.ndarray = calc_dissimilarity_measures(features, window_size)
    return apply_weightened_moving_average(dissimilarity_measures, weights).reshape(-1)


def calculate_prominence_measure(dissimilarity_measures: np.ndarray, window_size: int) -> np.ndarray:
    local_maximas: np.ndarray = argrelmax(dissimilarity_measures)[0]
    t_left: np.ndarray = -np.ones_like(dissimilarity_measures, dtype=int)
    t_right: np.ndarray = dissimilarity_measures.shape[0] * np.ones_like(dissimilarity_measures, dtype=int)
    for maxima_idx in local_maximas:
        for idx in range(0, maxima_idx):
            if dissimilarity_measures[idx] > dissimilarity_measures[maxima_idx]:
                t_left[maxima_idx] = idx
                break

        for idx in range(maxima_idx + 1, dissimilarity_measures.shape[0]):
            if dissimilarity_measures[idx] > dissimilarity_measures[maxima_idx]:
                t_right[maxima_idx] = idx
                break
    t_left = np.maximum(t_left, window_size)
    t_right = np.minimum(t_right, dissimilarity_measures.shape[0] - window_size)

    prominence: np.ndarray = np.zeros_like(dissimilarity_measures)
    for idx in local_maximas:
        left, right = t_left[idx], t_right[idx]
        if (left == -1) or (right == -1) or (idx <= window_size) or (idx > dissimilarity_measures.shape[0] - window_size):
            continue

        min_left = np.min(dissimilarity_measures[left : idx])
        min_right = np.min(dissimilarity_measures[idx: right + 1])
        prominence[idx] = dissimilarity_measures[idx] - max(min_left, min_right)
    prominence = np.append([0] * window_size, prominence[:-window_size])
    return prominence / prominence.max()
