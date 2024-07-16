import copy
import inspect
import functools
import json
import os
import sys
import time
from argparse import Namespace
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from typing import *

import numpy as np
import scipy as sc
import sklearn.manifold
import torch
import torch.nn as nn
import torch.nn.functional as Fn
import torch.optim as optim
import tensordict
from dimarray import DimArray
from tensordict import TensorDict

# from huggingface_hub import hf_hub_download
from transformers import GPT2Config, GPT2Model

from infrastructure import utils, loader
from infrastructure.utils import PTR
from infrastructure.experiment import *
from infrastructure.settings import DTYPE, DEVICE
from infrastructure.discrete_are import Riccati, solve_discrete_are
from model.base import Predictor
from model.sequential import *
from model.convolutional import *
from model.transformer import *
from model.zero_predictor import ZeroPredictor

from system.linear_time_invariant import LTISystem, MOPDistribution


if __name__ == '__main__':
    torch.set_printoptions(precision=12, sci_mode=False, linewidth=200)

    """ Sandbox 1 """
    # systemArgs = Namespace(
    #     S_D=2,
    #     I_D=0,
    #     O_D=1,
    #     SNR=2.,
    #     input_enabled=False
    # )
    # n_sys = 200 # 10
    # systems = [LinearSystem.sample_stable_system(systemArgs) for _ in range(n_sys)]
    # stacked_systems = TensorDict(torch.func.stack_module_state(systems)[0], batch_size=(n_sys,))[:, None]
    #
    # # modelArgs = Namespace(
    # #     S_D=1000,
    # #     I_D=3,
    # #     O_D=2,
    # #     input_enabled=False
    # # )
    # # n_kfs = 5
    #
    # # initialization = TensorDict({
    # #     'F': torch.stack([utils.sample_stable_state_matrix(modelArgs.S_D) for _ in range(n_kfs)]),
    # #     'B': torch.randn((n_kfs, modelArgs.S_D, modelArgs.I_D)) / 3.,
    # #     'H': torch.randn((n_kfs, modelArgs.O_D, modelArgs.S_D)) / 3.,
    # #     'K': torch.randn((n_kfs, modelArgs.S_D, modelArgs.O_D)) / 3.
    # # }, batch_size=(n_kfs,))
    # # kfs = [RnnKF(modelArgs, **initialization[_]) for _ in range(n_kfs)]
    # kfs = [AnalyticalKF(sys) for sys in systems]
    # stacked_kfs = TensorDict(utils.stack_modules(kfs), batch_size=(n_sys,))[None, :]
    #
    # il = torch.Tensor([torch.trace(sys.S_observation_inf) for sys in systems])
    # zl = torch.Tensor([torch.trace(sys.H @ sys.S_state_inf @ sys.H.mT + sys.S_V) for sys in systems])
    # analytical_error = SequentialKF.analytical_error(stacked_kfs, stacked_systems)
    #
    # # print(analytical_error)
    # # print(torch.diag(il))
    # print(analytical_error)
    #
    # normalized_error = analytical_error / torch.diag(analytical_error)[:, None]
    # indices = torch.arange(n_sys)[:, None].expand_as(normalized_error)
    #
    # symmetric_normalized_error = (normalized_error + normalized_error.T) / 2 - 1
    # C = torch.eye(n_sys) - (1 / n_sys)
    # gram_matrix = -0.5 * (C @ symmetric_normalized_error @ C)
    #
    # """ Singular values plot """
    # plt.plot(torch.linalg.svdvals(normalized_error)[:20].detach(), label='normalized_error')
    # plt.plot(torch.linalg.svdvals(analytical_error)[:20].detach(), label='unnormalized_error')
    # plt.title('similarity_matrix_singular_values')
    # plt.yscale('log')
    # plt.legend()
    # plt.show()
    #
    # """ Eigenvalues plot """
    # plt.plot(torch.linalg.eig(gram_matrix)[0].abs().detach())
    # plt.title('gram_matrix_eigenvalues')
    # plt.yscale('log')
    # plt.show()
    # # print(torch.linalg.eig(gram_matrix)[0].shape)
    #
    # # X_control = torch.randn(n_sys, 12)
    # # symmetric_normalized_errors = torch.Tensor(sc.spatial.distance_matrix(X_control, X_control))
    # # print(symmetric_normalized_errors)
    #
    # """ MDS Plot """
    # mds = sklearn.manifold.MDS(n_components=2)
    # pos = mds.fit_transform(symmetric_normalized_error.detach())
    #
    # similarities = (symmetric_normalized_error.max() / (symmetric_normalized_error + 1e-9) * 100).detach().numpy()
    # np.fill_diagonal(similarities, 0)
    #
    # # Plot the edges
    # start_idx, end_idx = np.where(pos)
    # segments = [[pos[i, :], pos[j, :]] for i in range(len(pos)) for j in range(len(pos))]
    # values = np.abs(similarities)
    # # lc = LineCollection(
    # #     segments, zorder=0, cmap=plt.cm.Blues, norm=plt.Normalize(0, values.max())
    # # )
    # # lc.set_array(similarities.flatten())
    # # lc.set_linewidths(np.full(len(segments), 0.5))
    # # ax.add_collection(lc)
    # #
    # # plt.scatter(pos[:, 0], pos[:, 1], color="turquoise", lw=0, label="MDS")
    # # plt.legend(scatterpoints=1, loc="best", shadow=False)
    # # plt.title('MDS')
    # # plt.show()
    #
    # """ Error comparison with zero predictor """
    # plt.scatter(*torch.stack([indices, analytical_error], dim=0).flatten(1, -1).detach())
    # plt.scatter(torch.arange(n_sys), il, color='black', s=128, marker='*', label='kalman_filter')
    # plt.scatter(torch.arange(n_sys), zl, color='purple', s=64, label='zero_predictor')
    # plt.xlabel('system_index')
    # plt.ylabel('normalized_analytical_error')
    # plt.yscale('log')
    # plt.legend()
    # plt.show()
    #
    # # print(torch.diag(analytical_error) > il)
    # # print(torch.diag(analytical_error) - il)

    """ Sandbox 2 """
    # torch.manual_seed(1212)
    # file = hf_hub_download(
    #     repo_id="hf-internal-testing/tourism-monthly-batch", filename="train-batch.pt", repo_type="dataset"
    # )
    # batch = TensorDict(torch.load(file), batch_size=(64,))
    # model_ = TimeSeriesTransformerModel.from_pretrained("huggingface/time-series-transformer-tourism-monthly")
    # config_ = model_.config
    #
    # D = 1   # Size of the output we want
    # new_config_params = utils.deepcopy(config_.__dict__)
    # new_config_params.update({
    #     'input_size': D,
    #     "num_static_categorical_features": 0,
    #     "num_static_real_features": 0,
    #     "prediction_length": 1,
    #     "num_parallel_samples": 1
    # })
    # config = TimeSeriesTransformerConfig(**{
    #     k: new_config_params[k]
    #     for k in inspect.getfullargspec(TimeSeriesTransformerConfig).args
    #     if k != 'self'
    # })
    #
    # for k, v in config.__dict__.items():
    #     print(f'{k}: {config_.__dict__[k]}, {v}')
    #
    # model = TimeSeriesTransformerModel(config)
    #
    # if D > 1:
    #     batch["past_values"] = batch["past_values"].unsqueeze(-1).expand(-1, -1, D)
    #     batch["past_observed_mask"] = batch["past_observed_mask"].unsqueeze(-1).expand(-1, -1, D)
    # for k in ("observed_mask", "time_features", "values"):
    #     past_k = "past_" + k
    #     batch[past_k] = batch[past_k][:, 0:]
    #
    #     future_k = "future_" + k
    #     batch[future_k] = batch[future_k][:, :1]
    # batch = batch[:5]
    # print(batch)
    # print(batch["future_observed_mask"][0, :])
    # outputs = model(**{
    #     k: batch[k]
    #     for k in ('past_values', 'past_time_features', 'past_observed_mask', 'future_time_features')
    # }, attention_mask=torch.zeros_like(batch["past_values"]))
    # print(outputs.sequences.squeeze())

    """ Sandbox 3 """
    # torch.set_default_dtype(torch.float64)
    #
    # S_D, I_D, O_D = 7, 6, 4
    # S_Dh = 10
    # L = 20
    #
    # F = torch.randn((S_D, S_D)) / 3
    # B = torch.randn((S_D, I_D)) / 3
    # H = torch.randn((O_D, S_D)) / 3
    #
    # Fh = torch.randn((S_Dh, S_Dh)) / 3
    # Bh = torch.randn((S_Dh, I_D)) / 3
    # Hh = torch.randn((O_D, S_Dh)) / 3
    # Kh = torch.randn((S_Dh, O_D)) / 3
    #
    # x0 = x = torch.randn((S_D, 1))
    # xh0 = xh = torch.zeros((S_Dh, 1))
    # u = torch.randn((L + 1, I_D, 1))
    # w = torch.randn((L + 1, S_D, 1))
    # v = torch.randn((L + 1, O_D, 1))
    #
    # observations, observationsh = [torch.empty((O_D, 1))], [torch.empty((O_D, 1))]
    # for i in range(1, L + 1):
    #     x = F @ x + B @ u[i] + w[i]
    #     y = H @ x + v[i]
    #
    #     xh = Fh @ xh + Bh @ u[i]
    #     yh = Hh @ xh
    #     xh = xh + Kh @ (y - yh)
    #
    #     observations.append(y)
    #     observationsh.append(yh)
    # observations = torch.stack(observations)
    # observationsh = torch.stack(observationsh)
    #
    # print(observations.shape, observationsh.shape)
    #
    # t = torch.randint(1, L + 1, ()).item()
    #
    # M, Mh = F, Fh @ (torch.eye(S_Dh) - Kh @ Hh)
    # FhKhH = Fh @ Kh @ H
    #
    # sum_ = lambda arg, default_shape: torch.tensor(sum(arg)) if len(arg) > 0 else torch.zeros(default_shape, dtype=torch.float64)
    # term1 = (H @ torch.matrix_power(M, t) - Hh @ sum_([torch.matrix_power(Mh, t - 1 - k) @ FhKhH @ torch.matrix_power(M, k) for k in range(1, t)], (S_Dh, S_D))) @ x0
    # term2 = sum_([(H @ torch.matrix_power(M, t - k) @ B - Hh @ torch.matrix_power(Mh, t - k) @ Bh) @ u[k] for k in range(1, t + 1)], (O_D, 1)) - Hh @ sum_([sum_([torch.matrix_power(Mh, l) @ FhKhH @ torch.matrix_power(M, (t - 1 - k) - l) @ B for l in range(t - k)], (S_Dh, I_D)) @ u[k] for k in range(1, t)], (S_Dh, 1))
    # term3 = H @ sum_([torch.matrix_power(M, t - k) @ w[k] for k in range(1, t + 1)], (S_D, 1)) - Hh @ sum_([sum_([torch.matrix_power(Mh, l) @ FhKhH @ torch.matrix_power(M, (t - 1 - k) - l) for l in range(t - k)], (S_Dh, S_D)) @ w[k] for k in range(1, t)], (S_Dh, 1))
    # term4 = v[t] - Hh @ sum_([torch.matrix_power(Mh, t - 1 - k) @ Fh @ Kh @ v[k] for k in range(1, t)], (S_Dh, 1))
    #
    # y_yh = term1 + term2 + term3 + term4
    #
    # # print(term1)
    # # print(term2)
    # # print(term3)
    # # print(term4)
    #
    # print(y_yh)
    # print(observations[t] - observationsh[t])

    """ Sandbox 4 """
    # I_D, O_D, input_enabled = 1, 1, False
    # systemArgs = Namespace(
    #     S_D=3,
    #     I_D=I_D,
    #     O_D=O_D,
    #     SNR=2.,
    #     input_enabled=input_enabled
    # )
    # modelArgs = Namespace(
    #     S_D=2,
    #     I_D=I_D,
    #     O_D=O_D,
    #     ir_length=32,
    #     input_enabled=input_enabled
    # )
    #
    # n = 1
    # systems = [LinearSystem.sample_stable_system(systemArgs) for _ in range(n)]
    # stacked_systems = TensorDict(torch.func.stack_module_state(systems)[0], batch_size=(n,))
    #
    # B, L = 2, 200
    #
    # test_trace = TensorDict({
    #     'state': torch.randn((B, modelArgs.S_D)),
    #     'input': torch.randn((B, L, modelArgs.I_D)),
    #     'observation': torch.randn((B, L, modelArgs.O_D))
    # }, batch_size=(B,))
    #
    # n_kfs = n
    # kfs = [CnnKFAnalytical(modelArgs).eval() for _ in range(n_kfs)]
    # # kfs = [CnnKFLeastSquares(modelArgs).eval() for _ in range(n_kfs)]
    # for kf, sys in zip(kfs, systems):
    #     initialization, error = kf._analytical_initialization(sys.state_dict())
    #     # initialization, error = kf._least_squares_initialization(dict(test_trace))
    #     kf.load_state_dict(initialization)
    #
    # stacked_kfs = TensorDict(torch.func.stack_module_state(kfs)[0], batch_size=(n_kfs,))
    #
    # il = torch.Tensor([torch.trace(sys.S_observation_inf) for sys in systems])
    # analytical_error = CnnKF.analytical_error(stacked_kfs[:, None], stacked_systems[None, :])
    #
    # print('Analytical error:', analytical_error.squeeze())
    # print('Irreducible loss:', il.squeeze())
    # print('Difference:', (analytical_error - il).squeeze())
    # print((analytical_error > torch.diag(il)).squeeze())

    # stacked_sequentialized_kfs = TensorDict(torch.func.stack_module_state(sequentialized_kfs)[0], batch_size=(n,))
    # batch_stacked_sequentialized_kfs = CnnKF.to_sequential_batch(stacked_kfs, input_enabled)
    # SequentialKF.analytical_error(batch_stacked_sequentialized_kfs, stacked_systems)

    # cnn_kf = CnnKFLeastSquares(modelArgs).eval()
    # initialization, error = cnn_kf._least_squares_initialization(dict(test_trace))
    # cnn_kf.load_state_dict(initialization)
    # sequentialized_cnn_kf = cnn_kf.to_sequential().eval()
    #
    # result = cnn_kf(test_trace)
    # sequential_result = sequentialized_cnn_kf(test_trace)
    #
    # print(Fn.mse_loss(result['observation_estimation'], sequential_result['observation_estimation']))
    # print(torch.max((result['observation_estimation'] - sequential_result['observation_estimation']) ** 2))
    #
    # print(torch.allclose(result['observation_estimation'], sequential_result['observation_estimation']))

    """ Sandbox 5 """
    # modelArgs = Namespace(
    #     S_D=2,
    #     I_D=1,
    #     O_D=1,
    #     SNR=2.,
    #     ir_length=4,
    #     input_enabled=False
    # )
    # n = 4
    # kfs = [CnnKFLeastSquares(modelArgs) for _ in range(n)]
    #
    # shape = (2, 3, 4)
    # t = torch.randn((5, 3))
    # print(t.unsqueeze(-1).shape)
    # params = torch.func.stack_module_state(kfs)[0]
    # td = TensorDict(params, batch_size=(n,))
    #
    # print(kfs[0].observation_IR.data.data_ptr())
    # print(params['observation_IR'][0].data_ptr())
    # print(td['observation_IR'][0].data_ptr())

    """ Sandbox 6 """
    # I_D, O_D, input_enabled = 1, 1, False
    # systemArgs = Namespace(
    #     S_D=3,
    #     I_D=I_D,
    #     O_D=O_D,
    #     SNR=2.,
    #     input_enabled=input_enabled
    # )
    # modelArgs = Namespace(
    #     I_D=I_D,
    #     O_D=O_D,
    #     ir_length=64,
    #     input_enabled=input_enabled
    # )
    #
    # system = torch.load("system_logs2.pt")
    # # system = LinearSystem.sample_stable_system(systemArgs)
    # # torch.save(system, "system_logs2.pt")
    # analytical_kf = AnalyticalKF(system)
    #
    # system_td = TensorDict(system.state_dict(), batch_size=())
    # il = torch.trace(system.S_observation_inf)
    #
    # analytical_kf_td = TensorDict(analytical_kf.state_dict(), batch_size=())
    # analytical_il = SequentialKF.analytical_error(analytical_kf_td, system_td)
    #
    # kf = CnnKFAnalytical(modelArgs)
    # initialization, error = kf._analytical_initialization(system.state_dict())
    # kf.load_state_dict(initialization)
    # kf_params = torch.func.stack_module_state([kf])[0]
    # kf_td = TensorDict(kf_params, batch_size=(1,)).squeeze(0)
    # al = CnnKF.analytical_error(kf_td, system_td)
    #
    #
    # print(al - analytical_il)
    #
    # sequential_logs = torch.load('sequential_logs.pt')
    # cnn_logs = torch.load('cnn_logs.pt')
    #
    # print(cnn_logs['ws_recent_err'] - sequential_logs['ws_geometric_err'])
    # print(cnn_logs['ws_geometric_err'])
    # print(cnn_logs['v_recent_err'] - sequential_logs['v_geometric_err'])
    #
    # print(cnn_logs['ws_recent_err'] - sequential_logs['ws_geometric_err']
    #       + cnn_logs['ws_geometric_err']
    #       + cnn_logs['v_recent_err'] - sequential_logs['v_geometric_err'])

    # try:
    #     os.remove("sandbox_logs2.txt")
    #     # os.remove("system_logs2.pt")
    # except FileNotFoundError:
    #     pass
    # torch.save(system, "system_logs2.pt")

    # r_max = 32
    # for r in range(1, r_max + 1):
    #     modelArgs = utils.deepcopy_namespace(_modelArgs)
    #     modelArgs.ir_length = r
    #
    #     kf = CnnKFAnalytical(modelArgs)
    #     # initialization, error = kf._analytical_initialization(system.state_dict())
    #     # kf.load_state_dict(initialization)
    #     kf_params = torch.func.stack_module_state([kf])[0]
    #     kf_td = TensorDict(kf_params, batch_size=(1,)).squeeze(0)
    #
    #     # optimizer = optim.LBFGS(kf_params.values(), lr=1)
    #     optimizer = optim.SGD(kf_params.values(), lr=0.1, momentum=0.0, weight_decay=0.0)
    #
    #     gn = 1
    #     t = 0
    #     while gn > 1e-27:
    #     # for _ in range(12):
    #         # print(f'Iteration {t}')
    #         analytical_error = CnnKF.analytical_error(kf_td, system_td)
    #         normalized_error = analytical_error - il
    #         loss = normalized_error.sum()
    #         # print('\tNormalized error:', normalized_error)
    #
    #         optimizer.zero_grad()
    #         loss.backward()
    #
    #         # print({k: v.grad for k, v in kf_params.items()})
    #         gn = torch.sum(torch.stack([
    #             v.grad.norm() ** 2
    #             for v in kf_params.values() if v.grad is not None]), dim=0)
    #         # print('\tGradient norm:', gn)
    #
    #         optimizer.step()
    #         # optimizer.step(lambda: CnnKF.analytical_error(kf_td, system_td).sum().item())
    #         t += 1
    #
    #     normalized_error = CnnKF.analytical_error(kf_td, system_td) - il
    #     print(f"{r}: {normalized_error}")

    # print('Analytical error:', analytical_error.squeeze())
    # print('Irreducible loss:', il.squeeze())
    # print('Difference:', (analytical_error - il).squeeze())
    # print((analytical_error > torch.diag(il)).squeeze())

    """ Sandbox 7 """
    # I_D, O_D, input_enabled = 1, 1, False
    # systemArgs = Namespace(
    #     S_D=3,
    #     I_D=I_D,
    #     O_D=O_D,
    #     SNR=2.,
    #     input_enabled=input_enabled
    # )
    # modelArgs = Namespace(
    #     I_D=I_D,
    #     O_D=O_D,
    #     ir_length=4,
    #     # ridge=0.01,
    #     tikhonov=torch.randn((4, 4)),
    #     input_enabled=input_enabled
    # )
    #
    # M = nn.Parameter(torch.randn(5, 5))
    # M2 = nn.Parameter(M)
    #
    # cnn_kf = CnnKFLeastSquares(modelArgs)
    # print(cnn_kf.state_dict())

    """ Sandbox 8 """
    # systemArgs = Namespace(
    #     S_D=3,
    #     I_D=3,
    #     O_D=2,
    #     SNR=2.,
    #     input_enabled=False
    # )
    # modelArgs = Namespace(
    #     S_D=1000,
    #     I_D=3,
    #     O_D=2,
    #     input_enabled=False
    # )
    #
    # n_sys = 5
    # systems = [LinearSystem.sample_stable_system(systemArgs) for _ in range(n_sys)]
    # td = TensorDict(torch.func.stack_module_state(systems)[0], batch_size=(n_sys,))[:, None]
    #
    # print(td)
    # print(TensorDict({
    #     k: v for k, v in td.items()
    #     if v.requires_grad
    # }, batch_size=td.batch_size))
    #
    # kf_items = list(td.items(include_nested=False))
    # print({k: v.shape for k, v in kf_items})
    # cum_parameter_lengths = [0] + list(np.cumsum([np.prod(v.shape[2:]) for k, v in kf_items]))
    #
    # print(cum_parameter_lengths)

    """ Sandbox 9 """
    # base_exp_name = 'AdversarialSystemsBasicDebug3'
    # output_dir = 'system2_CNN'
    # output_fname = 'result'
    #
    # system2, args = loader.load_system_and_args('data/2dim_scalar_system_matrices')
    # systems = [system2]
    #
    # args.model.model = CnnKFAnalyticalLeastSquares
    # args.model.ir_length = 8
    # args.experiment.ensemble_size = 1
    # args.experiment.metrics = {'analytical_validation'}
    # args.experiment.exp_name = base_exp_name
    #
    # configurations = []
    #
    # # os.removedirs(f"output/{output_dir}/{base_exp_name}")
    # result = run_experiments(
    #     args, configurations, {
    #         'dir': output_dir,
    #         'fname': output_fname
    #     }, systems
    # )
    #
    # for k, v in result['system_ptr'].items():
    #     v.grad = None
    # observation_IR = result['learned_kf'][()][1][0, 0]['observation_IR']
    # observation_IR.sum().backward()
    # for k, v in result['system_ptr'].items():
    #     print(f"{k}: {v.grad}")

    """ Sandbox 10 """
    # # systemArgs = Namespace(
    # #     S_D=3,
    # #     I_D=2,
    # #     O_D=2,
    # #     SNR=2.,
    # #     input_enabled=False
    # # )
    # # sys = LinearSystem.sample_stable_system(systemArgs)
    # sys, args = loader.load_system_and_args('data/2dim_scalar_system_matrices')
    #
    # # dataset = LinearSystem.generate_dataset([sys], 1, 20)[0]
    # # torch.save(dataset, 'sandbox_dataset.pt')
    # dataset = torch.load('sandbox_dataset.pt')
    #
    #
    # kf = AnalyticalKF(sys).eval()
    # kf_est = kf(dataset, steady_state=True)
    # print(dataset[0]['observation'].shape)
    #
    # def cdn(t: torch.Tensor) -> np.ndarray:
    #     return t.clone().detach().numpy()
    #
    # filterpy_kf = KalmanFilter(sys.S_D, sys.O_D)
    # filterpy_kf.x = filterpy_kf.x_prior = filterpy_kf.x_post = filterpy_kf.x.flatten()
    # filterpy_kf.F = cdn(sys.F)
    # filterpy_kf.H = cdn(sys.H)
    # filterpy_kf.Q = cdn(sys.S_W)
    # filterpy_kf.R = cdn(sys.S_V)
    #
    # filterpy_kf.P = cdn(sys.S_state_inf_intermediate)
    # filterpy_kf.K = cdn(sys.K)
    #
    # print(f"filterpy_kf y0: {filterpy_kf.H @ filterpy_kf.x}")
    # filterpy_kf.update_steadystate(cdn(dataset[0]['observation'][0]))
    # filterpy_kf.predict_steadystate()
    #
    # print(f"filterpy_kf y1: {filterpy_kf.H @ filterpy_kf.x}")
    # filterpy_kf.update_steadystate(cdn(dataset[0]['observation'][1]))
    # print(f"filterpy_kf x1: {filterpy_kf.x}")
    # filterpy_kf.predict_steadystate()
    #
    # print(f"filterpy_kf x2: {filterpy_kf.x}")
    # print(f"filterpy_kf y2: {filterpy_kf.H @ filterpy_kf.x}")
    #
    # # filterpy_kf.predict_steadystate()
    # # filterpy_kf.update_steadystate(cdn(dataset[0]['observation'][1]))
    # #
    # # print(f"filterpy_kf x2: {filterpy_kf.x}")
    # #
    # # # print(cdn(sys.H) @ filterpy_kf.x)
    # print(f"kf x0: {kf_est['state_estimation'][0, 0]}")
    # print(f"kf y0: {kf_est['observation_estimation'][0, 0]}")
    # print(f"kf x1: {kf_est['state_estimation'][0, 1]}")
    # print(f"kf y1: {kf_est['observation_estimation'][0, 1]}")

    """ Sandbox 11 """
    # I_D, O_D, input_enabled = 1, 1, False
    # systemArgs = Namespace(
    #     S_D=3,
    #     I_D=I_D,
    #     O_D=O_D,
    #     SNR=2.,
    #     input_enabled=input_enabled
    # )
    # modelArgs = Namespace(
    #     I_D=I_D,
    #     O_D=O_D,
    #     ir_length=32,
    #     input_enabled=input_enabled
    # )
    #
    # n = 200
    # systems = [LinearSystem.sample_stable_system(systemArgs) for _ in range(n)]
    # stacked_systems = TensorDict(utils.stack_modules(systems), batch_size=(n,))
    #
    # kfs = [CnnKFAnalytical(modelArgs).eval() for _ in range(n)]
    # for kf, sys_td in zip(kfs, stacked_systems):
    #     initialization, error = kf._analytical_initialization(dict(sys_td))
    #     kf.load_state_dict(initialization)
    #
    # stacked_kfs = TensorDict(torch.func.stack_module_state(kfs)[0], batch_size=(n,))
    #
    # irs = stacked_kfs['observation_IR'].flatten(1, -1)
    # ir_norms = torch.linalg.norm(irs, dim=-1)
    #
    # cosine_similarity = (irs @ irs.mT) / (ir_norms[:, None] * ir_norms[None, :])
    #
    # fig, axs = plt.subplots(nrows=1, ncols=2)
    #
    # im = axs[0].imshow(cosine_similarity.detach())
    # axs[0].set_title('cosine_similarity_matrix')
    # fig.colorbar(im, ax=axs[0])
    #
    # axs[1].plot(torch.linalg.svdvals(cosine_similarity).detach())
    # axs[1].set_yscale('log')
    # axs[1].set_title('cosine_similarity_svdvals')
    # plt.show()
    #
    # # il = torch.Tensor([torch.trace(sys.S_observation_inf) for sys in systems])
    # # analytical_error = CnnKF.analytical_error(stacked_kfs[:, None], stacked_systems[None, :])
    # #
    # # print('Analytical error:', analytical_error.squeeze())
    # # print('Irreducible loss:', il.squeeze())
    # # print('Difference:', (analytical_error - il).squeeze())
    # # print((analytical_error > torch.diag(il)).squeeze())

    """ Sandbox 12 """
    # I_D, O_D, input_enabled = 1, 5, False
    # model_shape = ()
    #
    # n_embd = 8 # 256
    # n_positions = 16
    # configuration = GPT2Config(
    #     n_positions=n_positions,  # set to sthg large advised
    #     n_embd=n_embd,
    #     n_layer=2,  # 12
    #     n_head=4,   # 8
    #     resid_pdrop=0.0,
    #     embd_pdrop=0.0,
    #     attn_pdrop=0.0,
    #     use_cache=False,
    # )
    # n_sys, B, L = 3, 5, 12
    #
    # MHP = Namespace(gpt2=configuration, I_D=I_D, O_D=O_D, input_enabled=input_enabled)
    # # model = GPT2InContextKF(MHP)
    # # dataset = TensorDict({
    # #     "input": torch.randn((B, L, I_D)),
    # #     "observation": nn.Parameter(torch.randn((B, L, O_D)))
    # # }, batch_size=(B, L))
    # # model(dataset)
    #
    # models = utils.multi_map(
    #     lambda _: GPT2InContextKF(MHP),
    #     np.empty(model_shape), dtype=object
    # )
    #
    # dataset = TensorDict({
    #     "input": torch.randn((*model_shape, n_sys, B, L, I_D)),
    #     "observation": nn.Parameter(torch.randn((*model_shape, n_sys, B, L, O_D)))
    # }, batch_size=(*model_shape, n_sys, B, L))
    #
    #
    # out = KF.run(*utils.stack_module_arr(models), dataset)
    # print(out.shape)
    # print(torch.autograd.grad(
    #     out[:, :, 0].norm(),
    #     dataset["observation"]
    # )[0][0, 0])

    """ Sandbox 13 """
    # base_exp_name = 'Basic'
    # output_dir = 'system2'
    # output_fname = 'result'
    #
    # system2, args = loader.load_system_and_args('data/2dim_scalar_system_matrices')
    # systems = [system2]
    # args.model.model = RnnKF
    # args.model.S_D = 2
    # args.train.scheduler = 'exponential'
    # args.train.epochs = 2500
    # args.experiment.ensemble_size = 1
    # args.experiment.exp_name = base_exp_name
    # args.experiment.metrics = {'analytical_validation'}
    #
    # configurations = []
    #
    # result = run_experiments(
    #     args, configurations, {
    #         'dir': output_dir,
    #         'fname': output_fname
    #     }, systems
    # )

    """ Sandbox 14 """
    # base_exp_name = "SingleTrace2"
    # output_dir = "system2_CNN"s
    # output_fname = "result"
    #
    # system2, args = loader.load_system_and_args('data/2dim_scalar_system_matrices')
    # args.model.ir_length = 16
    # args.train.subsequence_length = 32
    # args.train.warmup_duration = 12
    # args.train.epochs = 20
    # args.experiment.exp_name = base_exp_name
    # args.experiment.metrics = {"validation_analytical"}
    #
    # configurations = [
    #     ('model', {
    #         'model.model': [CnnKF, CnnKFLeastSquares]
    #     }),
    #     ('total_trace_length', {
    #         'dataset.train.total_sequence_length': [100, 200, 500, 1000, 2000, 5000, 10000]
    #     })
    # ]
    #
    # result = run_experiments(
    #     args, configurations, {
    #         'dir': output_dir,
    #         'fname': output_fname
    #     }, system2, save_experiment=False
    # )
    # M = get_metric_namespace_from_result(result)
    # print(utils.stack_tensor_arr(utils.multi_map(
    #     lambda metrics: metrics.obj,
    #     get_result_attr(result, "metrics"), dtype=TensorDict
    # )))
    # print(Namespace(**{k: v.shape for k, v in vars(M).items()}))

    """ Sandbox 15 """
    # base_exp_name = "SingleTrace"
    # output_dir = "system2_CNN"
    # output_fname = "result"
    #
    # system2, args = loader.load_system_and_args("data/2dim_scalar_system_matrices")
    # args.model.ir_length = 16
    # args.train.epochs = 500
    # args.train.subsequence_length = 32
    # args.experiment.exp_name = base_exp_name
    # args.experiment.metrics = {"validation_analytical"}
    #
    # configurations = [
    #     ("model", {
    #         "model.model": [CnnKF, CnnKFLeastSquares]
    #     }),
    #     ("total_trace_length", {
    #         "dataset.train.total_sequence_length": [100, 200, 500, 1000, 2000, 5000, 10000]
    #     })
    # ]
    #
    # result = run_experiments(
    #     args, configurations, {
    #         "dir": output_dir,
    #         "fname": output_fname
    #     }, system2, save_experiment=True
    # )
    #
    # plot_experiment(f"{output_dir}/{base_exp_name}", configurations, result, loss_type="analytical")

    """ Sandbox 16 """
    # from system.linear_quadratic_gaussian import LQGDistribution
    # from model.sequential.rnn_controller import RnnControllerAnalytical
    #
    # SHP = Namespace(S_D=2, I_D=1, O_D=1, input_enabled=True)
    # args = loader.generate_args(SHP)
    #
    # args.model.model = RnnControllerAnalytical
    # args.model.S_D = SHP.S_D
    # args.dataset.train.system.distribution = LQGDistribution("gaussian", "gaussian", 0.1, 0.1, 1.0, 1.0)
    # args.dataset.train.system.n_systems = 1
    #
    # args.experiment.n_experiments = 6
    # args.experiment.ensemble_size = 1
    # args.experiment.exp_name = "test"
    # # args.experiment.metrics = {"validation_analytical"}
    # args.experiment.ignore_metrics = {"impulse_target"}
    #
    # result, dataset = run_experiments(args, [], {}, save_experiment=False)
    #
    # M = get_metric_namespace_from_result(result)
    # dataset = dataset.values[()].obj
    # print(dataset["controller", "input"].shape)
    # print(M.output.input_estimation.shape)
    #
    # print(Fn.mse_loss(M.output.input_estimation, dataset["controller", "input"]))
    # print(Fn.mse_loss(M.output.observation_estimation, dataset["target"]))
    #
    # raise Exception()

    """ Sandbox 17 """
    # from transformers import GPT2Config, GPT2Model
    # from transformers import TransfoXLConfig, TransfoXLModel
    #
    # context_length = 250
    # d_embed = 256
    # n_layer = 12
    # n_head = 8
    # d_inner = 4 * d_embed
    #
    # gpt2 = GPT2Model(GPT2Config(
    #     n_positions=context_length,
    #     n_embd=d_embed,
    #     n_layer=n_layer,
    #     n_head=n_head,
    #     n_inner=d_inner,
    #     resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0, use_cache=False,
    # ))
    # transfoxl = TransfoXLModel(TransfoXLConfig(
    #     d_model=d_embed,
    #     d_embed=d_embed,
    #     n_layer=n_layer,
    #     n_head=n_head,
    #     d_head=d_embed // n_head,
    #     d_inner=d_inner,
    #     dropout=0.0,
    # ))
    #
    #
    # from system.linear_quadratic_gaussian import LQGDistribution
    # from model.transformer.transformerxl_iccontroller import TransformerXLInContextController
    #
    # exp_name = "TransformerXL"
    #
    # SHP = Namespace(S_D=2, I_D=1, O_D=1, input_enabled=True)
    # args = loader.generate_args(SHP)
    # args.model.model = TransformerXLInContextController
    # args.model.transformerxl = TransfoXLConfig(
    #     d_model=16,
    #     d_embed=8,
    #     n_head=4,
    #     d_head=4,
    #     d_inner=32,
    #     n_layer=2
    # )
    # args.dataset.train = Namespace(
    #     dataset_size=1,
    #     total_sequence_length=200,
    #     system=Namespace(
    #         n_systems=1,
    #         distribution=LQGDistribution("gaussian", "gaussian", 0.1, 0.1, 1.0, 1.0)
    #     )
    # )
    # args.dataset.valid = args.dataset.test = Namespace(
    #     dataset_size=5,
    #     total_sequence_length=10000,
    # )
    #
    # del args.train.warmup_duration
    # args.train.epochs = 100
    # args.train.subsequence_length = 32
    # args.train.batch_size = 128
    # args.train.iterations_per_epoch = 1
    #
    # args.train.optim_type = "GD"
    # args.train.max_lr = 1e-4
    # args.train.lr_decay = 1.0
    # args.train.weight_decay = 1e-2
    #
    # args.experiment.n_experiments = 1
    # args.experiment.ensemble_size = 1
    # args.experiment.exp_name = exp_name
    # args.experiment.metrics = {"validation"}
    #
    # result, dataset = run_experiments(args, [], {}, save_experiment=False)

    """ Sandbox 18 """
    from model.sequential.rnn_controller import RnnController
    torch.set_printoptions(precision=12, sci_mode=False)
    # SHP = Namespace(S_D=10, problem_shape=Namespace(
    #     environment=Namespace(observation=5),
    #     controller=Namespace(),
    #     # controller=Namespace(input=2),
    # ))
    # # systems = torch.load("output/imitation_learning/ControlNoiseComparison/training/systems.pt", map_location=DEVICE)["train"].values[()][0]
    # # systems = LTISystem(SHP.problem_shape, systems.td().squeeze(1).squeeze(0))
    #
    # dist = MOPDistribution("gaussian", "gaussian", 0.1, 0.1)
    # sys = dist.sample(SHP, ())
    #
    # # ds = systems.generate_dataset(1, 12)["environment"].squeeze(0)
    # # print(ds)
    # #
    # # x, xh, y, w, v = map(ds.__getitem__, ("state", "target_state_estimation", "observation", "w", "v"))
    # #
    # # augmented_x0 = torch.cat([x[0], xh[0]], dim=0)
    # #
    # # augmented_x1 = systems.effective.F @ augmented_x0 + torch.cat([
    # #     w[1], systems.environment.K @ (systems.environment.H @ w[1] + v[1])
    # # ], dim=0)
    # # print("a0", augmented_x0)
    # # print("b0", torch.cat([x[0], xh[0]], dim=0))
    # #
    # # print("a1", augmented_x1)
    # # print("b1", torch.cat([x[1], xh[1]], dim=0))
    #
    #
    # sys_td = sys.td()
    # reference_module = RnnController(SHP).eval()
    # kf_td = TensorDict.from_dict({
    #     k: v for k, v in {
    #         **sys_td.get("environment", {}),
    #         **sys_td.get("controller", {})
    #     }.items()
    #     if hasattr(reference_module, k)
    # }, batch_size=sys_td.shape) # .apply(torch.zeros_like)
    #
    # print(sys_td["environment", "irreducible_loss"])
    # print(sys_td["irreducible_loss"].to_dict())
    # print(sys_td["zero_predictor_loss"].to_dict())
    # raise Exception()
    # ds = sys.generate_dataset(1, 12)
    # out = Predictor.run(reference_module, kf_td, ds)
    # print(ds["controller", "input"][0, :4])
    # print(out["controller", "input"][0, :4])
    #
    # raise Exception()
    #
    # print(torch.autograd.grad(
    #     # sys_td["effective", "L", "input"].norm(),
    #     SequentialPredictor.analytical_error(kf_td, sys_td),
    #     sys_td["environment", "B", "input"]
    # ))

    """ Sandbox 19 """
    def analytical_error(kfs: TensorDict[str, torch.Tensor], systems: TensorDict[str, torch.Tensor]) -> torch.Tensor:
        # Variable definition
        Q = utils.complex(kfs["observation_IR"])                                                # [B... x O_D x R x O_D]
        Q = Q.permute(*range(Q.ndim - 3), -2, -1, -3)                                           # [B... x R x O_D x O_D]

        F = utils.complex(systems["environment", "F"])                                          # [B... x S_D x S_D]
        H = utils.complex(systems["environment", "H"])                                          # [B... x O_D x S_D]
        sqrt_S_W = utils.complex(systems["environment", "sqrt_S_W"])                            # [B... x S_D x S_D]
        sqrt_S_V = utils.complex(systems["environment", "sqrt_S_V"])                            # [B... x O_D x O_D]

        R = Q.shape[-3]

        L, V = torch.linalg.eig(F)                                                              # [B... x S_D], [B... x S_D x S_D]
        Vinv = torch.linalg.inv(V)                                                              # [B... x S_D x S_D]

        Hs = H @ V                                                                              # [B... x O_D x S_D]
        sqrt_S_Ws = Vinv @ sqrt_S_W                                                             # [B... x S_D x S_D]

        # State evolution noise error
        # Highlight
        ws_current_err = (Hs @ sqrt_S_Ws).norm(dim=(-2, -1)) ** 2                               # [B...]

        L_pow_series = L.unsqueeze(-2) ** torch.arange(1, R + 1)[:, None]                       # [B... x R x S_D]
        L_pow_series_inv = 1. / L_pow_series                                                    # [B... x R x S_D]

        QlHsLl = (Q @ Hs.unsqueeze(-3)) * L_pow_series_inv.unsqueeze(-2)                        # [B... x R x O_D x S_D]
        Hs_cumQlHsLl = Hs.unsqueeze(-3) - torch.cumsum(QlHsLl, dim=-3)                          # [B... x R x O_D x S_D]
        Hs_cumQlHsLl_Lk = Hs_cumQlHsLl * L_pow_series.unsqueeze(-2)                             # [B... x R x O_D x S_D]

        # Highlight
        ws_recent_err = (Hs_cumQlHsLl_Lk @ sqrt_S_Ws.unsqueeze(-3)).flatten(-3, -1).norm(dim=-1) ** 2   # [B...]

        Hs_cumQlHsLl_R = Hs_cumQlHsLl.index_select(-3, torch.tensor([R - 1])).squeeze(-3)       # [B... x O_D x S_D]
        cll = L.unsqueeze(-1) * L.unsqueeze(-2)                                                 # [B... x S_D x S_D]

        # Highlight
        _ws_geometric = (Hs_cumQlHsLl_R.mT @ Hs_cumQlHsLl_R) * ((cll ** (R + 1)) / (1 - cll))   # [B... x S_D x S_D]
        ws_geometric_err = utils.batch_trace(sqrt_S_Ws.mT @ _ws_geometric @ sqrt_S_Ws)          # [B...]

        # Observation noise error
        # Highlight
        v_current_err = sqrt_S_V.norm(dim=(-2, -1)) ** 2                                        # [B...]

        # Highlight
        # TODO: Backward pass on the above one breaks when Q = 0 for some reason
        # v_recent_err = (Q @ sqrt_S_V.unsqueeze(-3)).flatten(-3, -1).norm(dim=-1) ** 2           # [B...]
        v_recent_err = utils.batch_trace(sqrt_S_V.mT @ (Q.mT @ Q).sum(dim=-3) @ sqrt_S_V)       # [B...]


        # print("ws_current_err:", ws_current_err)
        # print("ws_recent_err:", ws_recent_err)
        # print("ws_geometric_err:", ws_geometric_err)
        # print("v_current_err:", v_current_err)
        # print("v_recent_err:", v_recent_err)

        err = ws_current_err + ws_recent_err + ws_geometric_err + v_current_err + v_recent_err  # [B...]
        return err.real

    import pickle
    with open("data/val_upperTriA_gauss_C_tensor_dicts.pkl", "rb") as fp:
        d = pickle.load(fp)[2]

    td = TensorDict.from_dict({"environment": {
         "F": d["A"], "H": d["C"], "sqrt_S_W": 0.1 * torch.eye(10), "sqrt_S_V": 0.1 * torch.eye(5), "B": TensorDict({}, batch_size=())
    }}, batch_size=()).apply(lambda t: t.to(DTYPE))


    SHP = Namespace(S_D=10, problem_shape=Namespace(
        environment=Namespace(observation=5),
        controller=Namespace(),
    ))
    sys = LTISystem(SHP.problem_shape, td)

    sys_td = sys.td()
    print(sys_td["environment", "irreducible_loss"])
    print(sys_td["irreducible_loss"].to_dict())
    print(sys_td["zero_predictor_loss"].to_dict())

    # t = torch.tensor([[[-0.213108941913, -0.076482892036, -0.002886268310, -0.399394601583,
    #       -0.531960129738],
    #      [-0.295079201460, -0.365193754435,  0.276217609644,  0.258012145758,
    #       -0.325427919626]],
    #
    #     [[-0.353170305490, -0.244755879045,  0.287547379732,  0.545907199383,
    #       -0.130725130439],
    #      [ 0.155552610755,  0.225297629833, -0.110139131546, -0.045099522918,
    #        0.083198539913]],
    #
    #     [[-0.736998200417,  0.084076076746, -0.059113919735,  0.179010629654,
    #       -0.370384752750],
    #      [ 0.130812391639, -0.330177664757,  0.242046400905,  0.091028898954,
    #       -0.264540642500]],
    #
    #     [[-0.905120491982, -0.545689702034,  0.347147256136,  0.538557529449,
    #       -0.267436683178],
    #      [-0.933974087238, -1.178467512131,  0.606216073036,  0.847504854202,
    #       -0.748944520950]],
    #
    #     [[-0.607982277870, -0.355590015650,  0.052718941122,  0.430850952864,
    #        0.086465097964],
    #      [ 0.468368589878, -0.142236992717, -0.128549367189,  0.158418118954,
    #        0.409350693226]]])

    t = torch.load("data/observation_IR_2.pt", map_location=DEVICE).to(DTYPE)
    kf_td = TensorDict.from_dict({"observation_IR": t}, batch_size=())

    print(analytical_error(kf_td, sys_td))











