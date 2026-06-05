"""Recipe-resolution and stage/model compatibility checks for the stage registry.

Run: PYTHONPATH=. python tests/test_stage_recipes.py
"""
import warnings
warnings.filterwarnings("ignore")

from argparse import Namespace

# Importing training registers the universal "sgd" stage.
import infrastructure.experiment.training  # noqa: F401
from infrastructure.experiment.stages import build_stages, STAGE_REGISTRY

from model.convolutional.cnn_predictor import (
    CnnPredictor,
    CnnLeastSquaresPredictor,
    CnnLeastSquaresPretrainPredictor,
    CnnAnalyticalPredictor,
    CnnAnalyticalLeastSquaresPredictor,
    CnnLeastSquaresRandomStepPredictor,
    CnnLeastSquaresNegationPredictor,
)
from model.sequential.rnn_predictor import RnnKalmanPredictor, RnnKalmanInitializedPredictor
from model.sequential.rnn_ho_kalman import (
    RnnHoKalmanAnalyticalPredictor,
    RnnHoKalmanAnalyticalLeastSquaresPredictor,
)
from model.zero_predictor import ZeroPredictor
from model.copy_predictor import CopyPredictor


def _model_args() -> Namespace:
    return Namespace(
        problem_shape=Namespace(
            environment=Namespace(observation=2),
            controller=Namespace(),
        ),
        S_D=4,
        ir_length=4,
        ridge=0.0,
    )


EXPECTED = {
    CnnPredictor: ["sgd"],
    CnnLeastSquaresPredictor: ["online_least_squares"],
    CnnLeastSquaresPretrainPredictor: ["online_least_squares", "sgd"],
    CnnAnalyticalPredictor: ["analytical_init"],
    CnnAnalyticalLeastSquaresPredictor: ["newton_init"],
    CnnLeastSquaresRandomStepPredictor: ["online_least_squares", "sgd", "random_step"],
    CnnLeastSquaresNegationPredictor: ["online_least_squares", "sgd", "negation"],
    RnnKalmanPredictor: ["analytical_init"],
    RnnKalmanInitializedPredictor: ["analytical_init", "sgd"],
    RnnHoKalmanAnalyticalPredictor: ["fir.analytical_init", "ho_kalman"],
    RnnHoKalmanAnalyticalLeastSquaresPredictor: ["fir.newton_init", "ho_kalman"],
    ZeroPredictor: [],
    CopyPredictor: [],
}


def test_recipes() -> None:
    for model_cls, expected_names in EXPECTED.items():
        model = model_cls(_model_args())
        stages = build_stages(model.training_recipe(), model)
        names = [s.name for s in stages]
        assert names == expected_names, f"{model_cls.__name__}: expected {expected_names}, got {names}"
        print(f"OK  {model_cls.__name__:45s} -> {names}")


def test_incompatibility() -> None:
    # A plain CnnPredictor lacks analytical/least-squares/newton/ho-kalman capabilities.
    model = CnnPredictor(_model_args())
    for bad in ("analytical_init", "least_squares_init", "newton_init", "ho_kalman"):
        try:
            build_stages([bad], model)
        except TypeError as e:
            print(f"OK  CnnPredictor rejects {bad!r}: {e}")
        else:
            raise AssertionError(f"Expected TypeError for stage {bad!r} on CnnPredictor")

    # Unknown stage name raises KeyError.
    try:
        build_stages(["does_not_exist"], model)
    except KeyError:
        print("OK  unknown stage name raises KeyError")
    else:
        raise AssertionError("Expected KeyError for unknown stage name")


if __name__ == "__main__":
    print(f"Registered stages: {sorted(STAGE_REGISTRY)}")
    test_recipes()
    test_incompatibility()
    print("ALL_OK")
