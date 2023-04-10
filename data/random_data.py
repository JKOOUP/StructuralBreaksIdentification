import typing as tp
import numpy as np


def random_structural_breaks(num_structural_breaks: int, segment_length_var: float = 10, segment_length_mean: float = 100) -> np.ndarray:
    seg_lengths: np.ndarray = np.sqrt(segment_length_var) * np.random.randn(num_structural_breaks - 1) + segment_length_mean
    seg_lengths = np.append([0], seg_lengths)
    seg_lengths = seg_lengths.astype(int)
    structural_breaks_positions: np.ndarray = np.cumsum(seg_lengths, dtype=int)
    return structural_breaks_positions


def jumping_mean_data(a1: float = 0.6, a2: float = -0.5, sigma_t: float = 1.5, num_structural_breaks: int = 50) -> np.ndarray:
    sb_positions: np.ndarray = random_structural_breaks(num_structural_breaks)

    mus: np.ndarray = np.zeros(sb_positions[1])
    for sb_pos_prev, sb_pos in zip(sb_positions[1:-1], sb_positions[2:]):
        mus = np.append(mus, mus[-1] + (num_structural_breaks / 16) * np.ones(sb_pos - sb_pos_prev))

    result: tp.List[float] = [0, 0]
    for curr_mu in mus[2:]:
        result.append(a1 * result[-1] + a2 * result[-2] + sigma_t * np.random.randn(1)[0] + curr_mu)

    return np.array(result), sb_positions


def scaling_variance_data(a1: float = 0.6, a2: float = -0.5, mu_t: float = 0.0, num_structural_breaks: int = 50) -> np.ndarray:
    sb_positions: np.ndarray = random_structural_breaks(num_structural_breaks)

    sigmas: np.ndarray = np.ones(sb_positions[1])
    for idx, (sb_pos_prev, sb_pos) in enumerate(zip(sb_positions[1:-1], sb_positions[2:])):
        if idx % 2 == 0:
            sigmas = np.append(sigmas, np.log(np.exp(1) + num_structural_breaks / 4) * np.ones(sb_pos - sb_pos_prev))
        else:
            sigmas = np.append(sigmas, np.ones(sb_pos - sb_pos_prev))

    result: tp.List[float] = [0, 0]
    for curr_sigma in sigmas[2:]:
        result.append(a1 * result[-1] + a2 * result[-2] + curr_sigma * np.random.randn(1)[0] + mu_t)

    return np.array(result), sb_positions


def changing_coefficients_data(a2: float = 0.0, mu_t: float = 0.0, sigma_t: float = 1.5, num_structural_breaks: int = 50) -> np.ndarray:
    sb_positions: np.ndarray = random_structural_breaks(num_structural_breaks, 100, 1000)

    a1s: np.ndarray = []
    for idx, (sb_pos_prev, sb_pos) in enumerate(zip(sb_positions[:-1], sb_positions[1:])):
        if idx % 2 == 0:
            a1s = np.append(a1s, 0.5 * np.random.rand(sb_pos - sb_pos_prev))
        else:
            a1s = np.append(a1s, 0.15 * np.random.rand(sb_pos - sb_pos_prev) + 0.8)

    result: tp.List[float] = [0, 0]
    for curr_a1 in a1s[2:]:
        result.append(curr_a1 * result[-1] + a2 * result[-2] + sigma_t * np.random.randn(1)[0] + mu_t)

    return np.array(result), sb_positions

def gaussians_mixtures_data(num_structural_breaks: int = 50) -> np.ndarray:
    sb_positions: np.ndarray = random_structural_breaks(num_structural_breaks)

    result: np.ndarray = []
    for idx, (sb_pos_prev, sb_pos) in enumerate(zip(sb_positions[:-1], sb_positions[1:])):
        if idx % 2 == 0:
            curr_seg_data: np.ndarray = 0.5 * (0.5 * np.random.randn(sb_pos - sb_pos_prev) - 1) + 0.5 * (0.5 * np.random.randn(sb_pos - sb_pos_prev) + 1)
            result = np.append(result, curr_seg_data)
        else:
            curr_seg_data: np.ndarray = 0.8 * (1.0 * np.random.randn(sb_pos - sb_pos_prev) - 1) + 0.2 * (0.1 * np.random.randn(sb_pos - sb_pos_prev) + 1)
            result = np.append(result, curr_seg_data)
    return result, sb_positions
