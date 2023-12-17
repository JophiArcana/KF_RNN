from infrastructure.train import *


def validate(
        flattened_dataset: TensorDict[str, torch.Tensor],
        base_kf: KF,
        flattened_ensembled_learned_kfs: TensorDict[str, torch.Tensor],
        dev_type: str
) -> torch.Tensor:
    flattened_ensembled_learned_kfs = flattened_ensembled_learned_kfs.to(dev_type)
    base_kf.eval()
    with torch.set_grad_enabled(False):
        return evaluate_run(
            run(flattened_dataset, base_kf, dict(flattened_ensembled_learned_kfs)),
            torch.Tensor(flattened_dataset['observation']),
            batch_mean=False
        )


def subtraction_normalized_validation_loss(
        systems: List[LinearSystem],
        base_kf: KF,
        learned_kf_arr: np.ndarray[TensorDict],
        dev_type: str
) -> torch.Tensor:
    systems = [sys.to(dev_type) for sys in systems]
    analytical_kfs = list(map(AnalyticalKF, systems))

    n_systems, ensemble_size = learned_kf_arr.flatten()[0].shape
    ValidArgs = Namespace(
        valid_dataset_size=500,
        valid_sequence_length=800,
        sequence_buffer=50
    )
    flattened_valid_dataset = add_targets(analytical_kfs, generate_dataset(
        systems=systems,
        batch_size=ValidArgs.valid_dataset_size,
        seq_length=ValidArgs.valid_sequence_length,
        dev_type=dev_type
    ))[:, None].expand(
        n_systems,
        ensemble_size,
        ValidArgs.valid_dataset_size,
        ValidArgs.valid_sequence_length
    ).flatten(0, 1)

    # Compute the empirical irreducible loss: [n_systems x 1 x valid_dataset_size]
    flattened_eil = evaluate_run(
        torch.Tensor(flattened_valid_dataset['target']),
        torch.Tensor(flattened_valid_dataset['observation']),
        batch_mean=False
    )

    flattened_vl = torch.stack([validate(
        flattened_valid_dataset,
        base_kf,
        learned_kf.flatten(),
        dev_type
    ) for learned_kf in learned_kf_arr.flatten()]).unflatten(0, learned_kf_arr.shape)

    return (flattened_vl - flattened_eil).unflatten(-2, (n_systems, ensemble_size)).cpu()




