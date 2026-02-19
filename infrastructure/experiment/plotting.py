from typing import Any

import numpy as np
import torch
from dimarray import DimArray
from matplotlib import pyplot as plt

from infrastructure.experiment import *

COLOR_LIST = np.array([
    [127, 113, 240],
    [247, 214, 124],
    [76, 186, 182],
    [245, 184, 120],
    [217, 17, 17],
    [240, 127, 189],
    [127, 242, 107],
    [237, 92, 208],
], dtype=float) / 255


def plot_experiment(
        plot_name: str,
        configurations: list[tuple[str, dict[str, list[Any] | np.ndarray[Any]]]],
        result: DimArray,
        loss_type: str = "empirical",
        xscale: str = "log",
        normalize: bool = True,
        clip: float = 1e-6,
        lstsq: bool = True,
        n_experiment_idx: int = 0
):
    plt.rcParams["figure.figsize"] = (8.0, 6.0)
    assert result.ndim == 2, f"Method {plot_experiment.__name__} can only be called if exactly 2 hyperparameters are swept but got {result.ndim}."

    xhp_name, xhp_values = (*configurations[-1][1].items(),)[0]
    xhp_values = torch.Tensor(xhp_values)

    get_result_attr(result, "learned_kfs")
    M = get_metric_namespace_from_result(result)

    if loss_type == "empirical":
        snvl_arr = M.l - M.eil if normalize else M.l
    elif loss_type == "analytical":
        snvl_arr = M.al - M.il if normalize else M.al
    else:
        snvl_arr = None
        assert loss_type in ("empirical", "analytical"), f"Loss type must be one of ('empirical', 'analytical') but got {repr(loss_type)}."
    snvl_arr = snvl_arr[..., n_experiment_idx, :, :, :]

    yhp_name, yhp_dict = configurations[0]
    yhp_values = yhp_dict.get("name", list(yhp_dict.values())[0])
    for snvl, name, color in zip(snvl_arr, yhp_values, COLOR_LIST):
        # DONE: Plotting code
        name = str(name)
        quantiles = torch.tensor([0.25, 0.75])

        # Take the median over validation traces, mean over validation systems, and median over training resampling
        snvl_median = snvl.median(-1).values.mean(-1).median(-1).values
        # Take the median over validation traces, mean over validation systems, and quartiles over training resampling
        snvl_train_quantiles = torch.quantile(snvl.median(-1).values.mean(-1), quantiles, dim=-1)
        # Take the median over training traces, mean over validation systems, and quartiles over validation resampling
        snvl_valid_quantiles = torch.quantile(snvl, quantiles, dim=-1).mean(-1).median(-1).values

        # Generate the plots
        plt.plot(xhp_values.cpu(), snvl_median.cpu(), color=color, marker=".", markersize=16, label=f"{name}_median")
        def plot_eb(quantiles: torch.Tensor, format_str: str, alpha: float) -> None:
            plt.fill_between(
                xhp_values.cpu(),
                *quantiles.clamp_min(clip).cpu(),
                color=color,
                alpha=alpha,
                label=format_str.format(name)
            )
        plot_eb(snvl_train_quantiles, "{0}_training_quartiles", 0.3)
        if loss_type == "empirical":
            plot_eb(snvl_valid_quantiles, "{0}_validation_quartiles", 0.1)

        # Compute the best fit line
        if lstsq:
            log_seq_lengths, log_snvl_median = torch.log(xhp_values), torch.log(snvl_median)
            augmented_log_seq_lengths = torch.stack([log_seq_lengths, torch.ones_like(log_seq_lengths)], dim=-1)
            line = (torch.linalg.pinv(augmented_log_seq_lengths) @ log_snvl_median.unsqueeze(-1)).squeeze(0)
            snvl_median_fit = torch.exp(augmented_log_seq_lengths @ line).squeeze(-1)
            plt.plot(
                xhp_values.cpu(),
                snvl_median_fit.cpu(),
                color="black",
                linestyle="dashed",
                label=f"$y = {line[1].exp().item()}x^\u007B{line[0].item()}\u007D$"
            )

    plt.xscale(xscale)
    plt.xlabel(xhp_name)
    plt.yscale("log")
    plt.ylabel(r"normalized_validation_loss: $\frac{1}{L}|| F_\theta(\tau) - \tau ||^2 - || KF(\tau) - \tau ||^2$")
    plt.title(plot_name)
    plt.legend(fontsize=6)
    plt.show()




