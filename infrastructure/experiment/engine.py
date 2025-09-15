import torch
from tensordict import TensorDict

from infrastructure import utils
from infrastructure.utils import ModelPair
from model.base import Predictor
from model.copy_predictor import CopyPredictor
from model.zero_predictor import ZeroPredictor
from system.base import SystemGroup

from .losses import SimpleLossFn, SIMPLE_LOSS_DICT


def error(
    model_pair: str | ModelPair,
    sg: SystemGroup | TensorDict,
    loss_fn: str | SimpleLossFn = "mse",
) -> TensorDict:
    if isinstance(loss_fn, str):
        loss_fn = SIMPLE_LOSS_DICT[loss_fn]

    if model_pair == "irreducible":
        if isinstance(sg, SystemGroup):
            if hasattr(sg, "irreducible_loss"):
                return sg.irreducible_loss
            else:
                problem_shape = sg.environment.problem_shape
                return TensorDict({
                    (*k.split("."),): torch.zeros(sg.group_shape)
                    for k in utils.nested_vars(problem_shape).keys()
                }, batch_size=sg.group_shape)
        else:
            output_key = ("environment", "target_observation_estimation",)
            target_key = ("environment", "observation",)
            irreducible_observation_loss = loss_fn(sg.get(output_key, sg[target_key]), sg[target_key])
            controller_losses = {
                ac_name: torch.zeros_like(irreducible_observation_loss)
                for ac_name in sg["controller"].keys()
            }
            return TensorDict({target_key: irreducible_observation_loss, "controller": controller_losses,}, batch_size=sg.shape)

    elif isinstance(model_pair, str):
        match model_pair:
            case "zero_predictor":
                model_arr = utils.array_of(ZeroPredictor(None))
                model_pair = utils.stack_module_arr(model_arr)
            case "copy_predictor":
                model_arr = utils.array_of(CopyPredictor(None))
                model_pair = utils.stack_module_arr(model_arr)
            case _:
                raise ValueError(model_pair)

    reference_module: Predictor = model_pair[0]
    kf_td: TensorDict = model_pair[1]

    if isinstance(sg, SystemGroup):
        return reference_module.analytical_error(kf_td, sg.td())
    else:
        run = Predictor.run(model_pair, sg)
        return TensorDict({
            k: loss_fn(v, sg[k])
            for k, v in run.items()
        }, batch_size=sg.shape)
















