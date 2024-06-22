from typing import *

import numpy as np
import torch
from dimarray import DimArray
from matplotlib import pyplot as plt

from infrastructure.experiment import *

COLOR_LIST = np.array([
    [76, 186, 182],
    [237, 125, 102],
    [127, 113, 240],
    [247, 214, 124],
    [217, 17, 17],
    [237, 92, 208]
], dtype=float) / 255


def plot_experiment(
        plot_name: str,
        configurations: List[Tuple[str, Dict[str, List[Any] | np.ndarray[Any]]]],
        result: DimArray,
        loss_type: str = "empirical",
        clip: float = 1e-6,
        lstsq: bool = True,
        n_experiment_idx: int = 0
):
    plt.rcParams['figure.figsize'] = (8.0, 6.0)
    assert result.ndim == 2, f"Method {plot_experiment.__name__} can only be called if exactly 2 hyperparameters are swept but got {result.ndim}."

    seq_lengths = torch.Tensor(configurations[-1][1]['dataset.train.total_sequence_length'])
    get_result_attr(result, "learned_kfs")
    M = get_metric_namespace_from_result(result)

    if loss_type == "empirical":
        snvl_arr = M.l - M.eil
    elif loss_type == "analytical":
        snvl_arr = M.al - M.il
    else:
        snvl_arr = None
        assert loss_type in ("empirical", "analytical"), f"Loss type must be one of ('empirical', 'analytical') but got {repr(loss_type)}."
    snvl_arr = torch.index_select(snvl_arr, -4, torch.tensor([n_experiment_idx])).squeeze(-4)

    hp_name, hp_dict = configurations[0]
    hp_list = hp_dict.get("name", list(hp_dict.values())[0])
    for snvl, name, color in zip(snvl_arr, hp_list, COLOR_LIST):
        # DONE: Plotting code
        name = str(name)
        quantiles = torch.tensor([0.25, 0.75])

        # Take the median over validation traces, mean over validation systems, and median over training resampling
        snvl_median = snvl.median(-1).values.mean(-1).median(-1).values
        # Take the median over validation traces, mean over validation systems, and quartiles over training resampling
        snvl_train_quantiles = torch.quantile(snvl.median(-1).values.mean(-1), quantiles, dim=-1)
        # Take the median over validation traces, mean over validation systems, and quartiles over training resampling
        snvl_valid_quantiles = torch.quantile(snvl, quantiles, dim=-1).mean(-1).median(-1).values

        # Generate the plots
        plt.plot(seq_lengths.cpu(), snvl_median.cpu(), color=color, marker='.', markersize=16, label=f'{name}_median')
        def plot_eb(quantiles: torch.Tensor, format_str: str, alpha: float) -> None:
            plt.fill_between(
                seq_lengths.cpu(),
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
            log_seq_lengths, log_snvl_median = torch.log(seq_lengths), torch.log(snvl_median)
            augmented_log_seq_lengths = torch.stack([log_seq_lengths, torch.ones_like(log_seq_lengths)], dim=-1)
            line = (torch.linalg.pinv(augmented_log_seq_lengths) @ log_snvl_median.unsqueeze(-1)).squeeze(0)
            snvl_median_fit = torch.exp(augmented_log_seq_lengths @ line).squeeze(-1)
            plt.plot(
                seq_lengths.cpu(),
                snvl_median_fit.cpu(),
                color='black',
                linestyle='dashed',
                label=f'$y = {line[1].exp().item()}x^\u007B{line[0].item()}\u007D$'
            )

    plt.xscale('log')
    plt.xlabel('total_trace_length')
    plt.yscale('log')
    plt.ylabel(r'normalized_validation_loss: $\frac{1}{L}|| F_\theta(\tau) - \tau ||^2 - || KF(\tau) - \tau ||^2$')
    plt.title(plot_name)
    plt.legend(fontsize=6)
    plt.show()

# def plot_comparison(
#         args: Namespace,
#         configurations: OrderedDict,
#         systems: List[LinearSystem],
#         learned_kf_arr: np.ndarray[TensorDict],
#         base_exp_name: str,
#         output_dir: str,
#         log_xscale: bool
# ):
#     plt.rcParams['figure.figsize'] = (8.0, 6.0)
#     exp_name = f'{output_dir}/{base_exp_name}'
#
#     outer_hp_name, _ = configurations[0]
#     outer_hp_values = _.get('name', list(_.values())[0])
#
#     inner_hp_name, _ = configurations[1]
#     inner_hp_values = _.get('name', list(_.values())[0])
#
#     learned_kf_arr = result['learned_kf']
#     M = result['metric']
#     # snvl_ = (M.l - M.eil).cpu()
#     # snvl_median = snvl_.median(-1).values.median(-1).values.permute(-1, *range(snvl_.ndim - 3))[n_idx]
#
#     snvl_ = (M.al - M.il).squeeze(-1).cpu()
#     snvl_median = snvl_.median(-1).values.permute(-1, *range(snvl_.ndim - 2))[n_idx]
#
#     c = plt.cm.pink(np.linspace(0, 0.8, len(outer_hp_values)))
#     for i, outer_hp_value in enumerate(outer_hp_values):
#         plt.plot(inner_hp_values, snvl_median[i], c=c[i], marker='.', markersize=16, label=f'{outer_hp_name}{outer_hp_value}')
#         argmin = torch.argmin(snvl_median[i])
#         plt.scatter([inner_hp_values[argmin]], [snvl_median[i, argmin]], c=c[i] * 0.5, s=256, marker='*')
#     # Use snvl_median[:, 0, i] for multiple RNN initializations
#
#     plt.xlabel(inner_hp_name)
#     if log_xscale:
#         plt.xscale('log')
#     # plt.xticks(hp_values)
#     # plt.gca().xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
#     plt.ylabel(r'normalized_validation_loss: $\frac{1}{L}|| F_\theta(\tau) - \tau ||^2 - || KF(\tau) - \tau ||^2$')
#     plt.yscale('log')
#     plt.title(exp_name)
#     plt.legend(fontsize=6)
#
#     plt.show()




