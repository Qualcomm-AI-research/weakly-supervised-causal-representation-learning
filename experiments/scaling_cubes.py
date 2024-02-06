#!/usr/bin/env python3
# Copyright (c) 2022 Qualcomm Technologies, Inc.
# All rights reserved.

import hydra
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import trange
from pathlib import Path
import mlflow
import pandas as pd
from collections import defaultdict

from ws_crl.encoder import SONEncoder, GaussianEncoder
from ws_crl.lcm import FlowLCM
from ws_crl.training import VAEMetrics
from ws_crl.metrics import compute_dci
from ws_crl.causal.graph import FixedGraph, create_graph
from ws_crl.causal.scm import (
    FixedOrderSCM,
    MLPFixedOrderSCM,
    MLPVariableOrderCausalModel,
    UnstructuredPrior,
    FixedGraphLinearANM,
)
from ws_crl.causal.implicit_scm import DEFAULT_BASE_DENSITY, MLPImplicitSCM
from ws_crl.lcm import ELCM, ILCM
from ws_crl.posthoc_graph_learning import (
    compute_implicit_causal_effects,
    find_topological_order,
    run_enco,
)

from experiments.experiment_utils import (
    initialize_experiment,
    save_config,
    save_model,
    logger,
    create_optimizer_and_scheduler,
    set_manifold_thickness,
    compute_metrics_on_dataset,
    reset_optimizer_state,
    create_intervention_encoder,
    update_dict,
    log_training_step,
    optimizer_step,
    step_schedules,
    determine_graph_learning_settings,
    frequency_check,
)
import nflows
from nflows import transforms


@hydra.main(config_path="../config", config_name="scaling_cubes_ilcm")
def main(cfg):
    """Hydra wrapper around experiment to be called from commandline"""
    experiment(cfg)


def experiment(cfg):
    """High-level experiment function"""
    # Initialization
    experiment_id = initialize_experiment(cfg)

    with mlflow.start_run(experiment_id=experiment_id, run_name=cfg.general.run_name):
        save_config(cfg)

        # Train
        model = create_model(cfg)
        train(cfg, model)
        save_model(cfg, model)

        # Test
        metrics = evaluate(cfg, model)

    logger.info("Anders nog iets?")
    return metrics


def create_model(cfg):
    """Instantiates a (learnable) VAE model"""

    # Create model
    logger.info(f"Creating {cfg.model.type} model")
    scm = create_scm(cfg)
    encoder, decoder = create_encoder_decoder(cfg)

    if cfg.model.type == "causal_vae":
        model = ELCM(
            scm, encoder=encoder, decoder=decoder, intervention_prior=None, dim_z=cfg.model.dim_z
        )
    elif cfg.model.type == "intervention_noise_vae":
        intervention_encoder = create_intervention_encoder(cfg)
        model = ILCM(
            scm,
            encoder=encoder,
            decoder=decoder,
            intervention_encoder=intervention_encoder,
            intervention_prior=None,
            averaging_strategy=cfg.model.averaging_strategy,
            dim_z=cfg.model.dim_z,
        )
    else:
        raise ValueError(f"Unknown value for cfg.model.type: {cfg.model.type}")

    # Load model checkpoint, if specified
    if "load" in cfg.model and cfg.model.load is not None:
        filename = cfg.model.load
        logger.info(f"Loading model checkpoint from {filename}")
        state_dict = torch.load(filename, map_location="cpu")
        model.load_state_dict(state_dict)

    return model


def create_scm(cfg):
    """Create SCM or implicit causal structure"""

    logger.info(f"Creating {cfg.model.scm.type} SCM")
    noise_centric = cfg.model.type in {
        "noise_vae",
        "intervention_noise_vae",
        "alt_intervention_noise_vae",
    }

    if cfg.model.scm.type == "ground_truth":
        raise NotImplementedError
    elif cfg.model.scm.type == "unstructured":  # Baseline VAE
        scm = UnstructuredPrior(dim_z=cfg.model.dim_z)
    elif noise_centric and cfg.model.scm.type == "mlp":
        logger.info(
            f"Graph parameterization for noise-centric learning: {cfg.model.scm.adjacency_matrix}"
        )
        scm = MLPImplicitSCM(
            graph_parameterization=cfg.model.scm.adjacency_matrix,
            manifold_thickness=cfg.model.scm.manifold_thickness,
            hidden_units=cfg.model.scm.hidden_units,
            hidden_layers=cfg.model.scm.hidden_layers,
            homoskedastic=cfg.model.scm.homoskedastic,
            dim_z=cfg.model.dim_z,
            min_std=cfg.model.scm.min_std,
        )
    elif (
        not noise_centric
        and cfg.model.scm.type == "mlp"
        and cfg.model.scm.adjacency_matrix in {"enco", "dds"}
    ):
        logger.info(
            f"Adjacency matrix: learnable, {cfg.model.scm.adjacency_matrix} parameterization"
        )
        scm = MLPVariableOrderCausalModel(
            graph_parameterization=cfg.model.scm.adjacency_matrix,
            manifold_thickness=cfg.model.scm.manifold_thickness,
            hidden_units=cfg.model.scm.hidden_units,
            hidden_layers=cfg.model.scm.hidden_layers,
            homoskedastic=cfg.model.scm.homoskedastic,
            dim_z=cfg.model.dim_z,
            enhance_causal_effects_at_init=False,
            min_std=cfg.model.scm.min_std,
        )
    elif (
        not noise_centric
        and cfg.model.scm.type == "mlp"
        and cfg.model.scm.adjacency_matrix == "fixed_order"
    ):
        logger.info(f"Adjacency matrix: learnable, fixed topological order")
        scm = MLPFixedOrderSCM(
            manifold_thickness=cfg.model.scm.manifold_thickness,
            hidden_units=cfg.model.scm.hidden_units,
            hidden_layers=cfg.model.scm.hidden_layers,
            homoskedastic=cfg.model.scm.homoskedastic,
            dim_z=cfg.model.dim_z,
            enhance_causal_effects_at_init=False,
            min_std=cfg.model.scm.min_std,
        )
    else:
        raise ValueError(f"Unknown value for cfg.model.scm.type: {cfg.model.scm.type}")

    return scm


def create_encoder_decoder(cfg):
    """Create encoder and decoder"""

    logger.info(f"Creating {cfg.model.encoder.type} encoder / decoder")

    if cfg.model.encoder.type == "mlp":
        encoder_hidden_layers = cfg.model.encoder.hidden_layers
        encoder_hidden = [cfg.model.encoder.hidden_units for _ in range(encoder_hidden_layers)]
        decoder_hidden_layers = cfg.model.decoder.hidden_layers
        decoder_hidden = [cfg.model.decoder.hidden_units for _ in range(decoder_hidden_layers)]

        encoder = GaussianEncoder(
            hidden=encoder_hidden,
            input_features=cfg.model.dim_x,
            output_features=cfg.model.dim_z,
            fix_std=cfg.model.encoder.fix_std,
            init_std=cfg.model.encoder.std,
            min_std=cfg.model.encoder.min_std,
        )
        decoder = GaussianEncoder(
            hidden=decoder_hidden,
            input_features=cfg.model.dim_z,
            output_features=cfg.model.dim_x,
            fix_std=cfg.model.decoder.fix_std,
            init_std=cfg.model.decoder.std,
            min_std=cfg.model.decoder.min_std,
        )
    else:
        raise ValueError(f"Unknown value for encoder_cfg.type: {cfg.model.encoder.type}")

    return encoder, decoder


def train(cfg, model):
    """High-level training function"""

    if "skip" in cfg.training and cfg.training.skip and cfg.training.skip != "None":
        return {}, {}

    logger.info("Starting training")
    logger.info(f"Training on {cfg.training.device}")
    device = torch.device(cfg.training.device)

    # Training
    criteria = VAEMetrics(dim_z=cfg.data.dim_z)
    optim, scheduler = create_optimizer_and_scheduler(cfg, model, separate_param_groups=True)

    train_metrics = defaultdict(list)
    val_metrics = defaultdict(list)
    best_state = {"state_dict": None, "loss": None, "step": None}

    train_data = load_dataset(cfg, "train", normalize=cfg.data.normalize)
    val_data = load_dataset(cfg, "val", normalize=cfg.data.normalize)
    train_loader = DataLoader(train_data, batch_size=cfg.training.batchsize, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=cfg.training.batchsize, shuffle=False)
    steps_per_epoch = len(train_loader)

    # GPU
    model = model.to(device)

    step = 0
    nan_counter = 0
    epoch_generator = trange(cfg.training.epochs, disable=not cfg.general.verbose)
    for epoch in epoch_generator:
        mlflow.log_metric("train.epoch", epoch, step=step)

        # Graph sampling settings
        graph_kwargs = determine_graph_learning_settings(cfg, epoch, model)

        # Epoch-based schedules
        model_interventions, pretrain, deterministic_intervention_encoder = epoch_schedules(
            cfg, model, epoch, optim
        )

        for x1, x2, z1, z2, intervention_labels, true_interventions in train_loader:
            fractional_epoch = step / steps_per_epoch

            model.train()

            # Step-based schedules
            (
                beta,
                beta_intervention,
                consistency_regularization_amount,
                cyclicity_regularization_amount,
                edge_regularization_amount,
                inverse_consistency_regularization_amount,
                z_regularization_amount,
                intervention_entropy_regularization_amount,
                intervention_encoder_offset,
            ) = step_schedules(cfg, model, fractional_epoch)

            # GPU
            x1, x2, z1, z2, intervention_labels, true_interventions = (
                x1.to(device),
                x2.to(device),
                z1.to(device),
                z2.to(device),
                intervention_labels.to(device),
                true_interventions.to(device),
            )

            # Model forward pass
            log_prob, model_outputs = model(
                x1,
                x2,
                beta=beta,
                beta_intervention_target=beta_intervention,
                pretrain_beta=cfg.training.pretrain_beta,
                full_likelihood=cfg.training.full_likelihood,
                likelihood_reduction=cfg.training.likelihood_reduction,
                pretrain=pretrain,
                model_interventions=model_interventions,
                deterministic_intervention_encoder=deterministic_intervention_encoder,
                intervention_encoder_offset=intervention_encoder_offset,
                **graph_kwargs,
            )

            # Loss and metrics
            loss, metrics = criteria(
                log_prob,
                true_intervention_labels=intervention_labels,
                z_regularization_amount=z_regularization_amount,
                edge_regularization_amount=edge_regularization_amount,
                cyclicity_regularization_amount=cyclicity_regularization_amount,
                consistency_regularization_amount=consistency_regularization_amount,
                inverse_consistency_regularization_amount=inverse_consistency_regularization_amount,
                intervention_entropy_regularization_amount=intervention_entropy_regularization_amount,
                **model_outputs,
            )

            # Optimizer step
            finite, grad_norm = optimizer_step(cfg, loss, model, model_outputs, optim, x1, x2)
            if not finite:
                nan_counter += 1

            # Log loss and metrics
            step += 1
            log_training_step(
                cfg,
                beta,
                epoch_generator,
                finite,
                grad_norm,
                metrics,
                model,
                step,
                train_metrics,
                nan_counter,
            )

            # Validation loop
            if frequency_check(step, cfg.training.validate_every_n_steps):
                validation_loop(
                    cfg, model, criteria, val_loader, best_state, val_metrics, step, device
                )

            # Save model checkpoint
            if frequency_check(step, cfg.training.save_model_every_n_steps):
                save_model(cfg, model, f"model_step_{step}.pt")

        # LR scheduler
        if scheduler is not None and epoch < cfg.training.epochs - 1:
            scheduler.step()
            mlflow.log_metric(f"train.lr", scheduler.get_last_lr()[0], step=step)

            # Optionally reset Adam stats
            if (
                cfg.training.lr_schedule.type == "cosine_restarts_reset"
                and (epoch + 1) % cfg.training.lr_schedule.restart_every_epochs == 0
                and epoch + 1 < cfg.training.epochs
            ):
                logger.info(f"Resetting optimizer at epoch {epoch + 1}")
                reset_optimizer_state(optim)

    # Final validation loop, wrapping up early stopping
    if cfg.training.validate_every_n_steps is not None and cfg.training.validate_every_n_steps > 0:
        validation_loop(cfg, model, criteria, val_loader, best_state, val_metrics, step, device)

        if cfg.training.early_stopping and best_state["step"] < step:
            logger.info(
                f'Early stopping after step {best_state["step"]} with validation loss '
                f'{best_state["loss"]}'
            )
            model.load_state_dict(best_state["state_dict"])

    # Reset model: back to CPU, reset manifold thickness
    set_manifold_thickness(cfg, model, None)

    return train_metrics, val_metrics


def get_train_mean_std(cfg):
    """Calculate mean and std of x1 and x2"""
    if (
        cfg.data.x1_mean is not None
        and cfg.data.x1_std is not None
        and cfg.data.x2_mean is not None
        and cfg.data.x2_std is not None
    ):
        # Return float tensors
        return (
            torch.FloatTensor(cfg.data.x1_mean),
            torch.FloatTensor(cfg.data.x1_std),
            torch.FloatTensor(cfg.data.x2_mean),
            torch.FloatTensor(cfg.data.x2_std),
        )

    train_data = load_dataset(cfg, "train", normalize=False)
    train_loader = DataLoader(train_data, batch_size=len(train_data), shuffle=False)
    x1, x2, *_ = next(iter(train_loader))

    # x1_mean per variable
    if cfg.data.x1_mean is None:
        cfg.data.x1_mean = x1.mean(dim=0).tolist()
    if cfg.data.x1_std is None:
        cfg.data.x1_std = x1.std(dim=0).tolist()
    if cfg.data.x2_mean is None:
        cfg.data.x2_mean = x2.mean(dim=0).tolist()
    if cfg.data.x2_std is None:
        cfg.data.x2_std = x2.std(dim=0).tolist()

    logger.info(f"x1_mean: {cfg.data.x1_mean}")
    logger.info(f"x1_std: {cfg.data.x1_std}")
    logger.info(f"x2_mean: {cfg.data.x2_mean}")
    logger.info(f"x2_std: {cfg.data.x2_std}")

    return (
        torch.FloatTensor(cfg.data.x1_mean),
        torch.FloatTensor(cfg.data.x1_std),
        torch.FloatTensor(cfg.data.x2_mean),
        torch.FloatTensor(cfg.data.x2_std),
    )


def load_dataset(cfg, tag, normalize=True):
    """Load train or test dataset and regernate if necessary"""

    assert tag in {"train", "dci_train", "test", "val"}

    # Regenerate data if necessary
    if cfg.data.always_generate_new_data or not Path(cfg.data.data_dir).exists():
        generate_datasets(cfg)
        cfg.data.always_generate_new_data = (
            False  # We only want to regenerate data once per experiment
        )

    # Load data
    filename = Path(cfg.data.data_dir) / f"{tag}.pt"
    logger.debug(f"Loading data from {filename}")
    data = torch.load(filename)
    x1, x2, *other = data

    # normalize x1 and x2
    if normalize:
        x1_mean, x1_std, x2_mean, x2_std = get_train_mean_std(cfg)
        x_mean = (x1_mean + x2_mean) / 2
        x_std = (x1_std + x2_std) / 2

        x1 = (x1 - x_mean) / x_std
        x2 = (x2 - x_mean) / x_std
    dataset = TensorDataset(x1, x2, *other)

    return dataset


class Uniform(torch.distributions.Uniform):
    def log_prob(self, value):
        return super().log_prob(value)

    def sample(self, sample_shape=torch.Size()):
        if isinstance(sample_shape, int):
            sample_shape = (sample_shape,)
        return super().sample(sample_shape)


class ConditionalAveragingTransform(nflows.transforms.Transform):
    def __init__(self, noise_scale=1):
        super().__init__()

        self.noise_scale = noise_scale

    def forward(self, inputs, context=None):
        batch_size = inputs.size(0)
        scale = torch.tensor([1 / self.noise_scale])

        num_dims = context.shape[1] // 2
        z_parents, parent_mask = context[:, :num_dims], context[:, num_dims:]

        # print(f"{z_parents=}")
        # print(f"{parent_mask=}")
        # Mean of non-masked variables
        shift = -torch.sum(z_parents, dim=1) / (
            torch.count_nonzero(parent_mask, dim=1) * self.noise_scale
        )
        shift = shift.unsqueeze(1)
        # print(f"{shift=}")
        outputs = (inputs - shift) / scale
        # raise NotImplementedError

        # Wrong, but not used for generating data
        logabsdet = 1
        return outputs, logabsdet

    def inverse(self, inputs, context=None):
        raise NotImplementedError


class ScaleTransform(nflows.transforms.Transform):
    def __init__(self, scale=1):
        super().__init__()

        self.scale = scale

    def forward(self, inputs, context):
        batch_size = inputs.size(0)

        # Wrong, but not used for generating data
        logabsdet = 1
        outputs = inputs * self.scale
        return outputs, logabsdet

    def inverse(self, inputs, context):
        batch_size = inputs.size(0)

        # Wrong, but not used for generating data
        logabsdet = 1
        outputs = inputs / self.scale
        return outputs, logabsdet


def create_true_model(cfg, uniform_noise=False):
    """Creates an instance of the ground-truth model, though with arbitrary network weights"""

    assert cfg.data.dim_z == 3
    # Create graph
    adjacency_matrix = torch.tensor([[0, 0, 1], [0, 0, 1], [0, 0, 0]])
    graph = FixedGraph(adjacency_matrix, dim_z=cfg.data.dim_z)

    if cfg.data.nature.noise == "uniform":
        base_density = Uniform(-1, 1)
    elif cfg.data.nature.noise == "gaussian":
        base_density = nflows.distributions.StandardNormal((1,))
    else:
        base_density = DEFAULT_BASE_DENSITY
    # Create SCM
    scm = FixedOrderSCM(
        graph=graph,
        dim_z=cfg.data.dim_z,
        manifold_thickness=cfg.data.nature.manifold_thickness,
        base_density=base_density,
        structure_transforms=[
            ScaleTransform(scale=cfg.data.nature.noise_scale),
            ScaleTransform(scale=cfg.data.nature.noise_scale),
            ConditionalAveragingTransform(noise_scale=cfg.data.nature.child_noise_scale),
        ],
        intervention_transforms=[
            ScaleTransform(scale=cfg.data.nature.noise_scale),
            ScaleTransform(scale=cfg.data.nature.noise_scale),
            ScaleTransform(scale=cfg.data.nature.noise_scale),
        ],
    )

    # Create SO(n) decoder
    encoder = SONEncoder(
        coeff_std=1.0, input_features=cfg.data.dim_z, output_features=cfg.data.dim_z
    )

    # Put together
    nature = FlowLCM(scm, encoder, dim_z=cfg.data.dim_z)

    # Disable learning NAture's parameters
    for param in nature.parameters():
        param.requires_grad = False

    return nature


def create_true_model2d(cfg, uniform_noise=False):
    """Creates an instance of the ground-truth model, though with arbitrary network weights"""

    assert cfg.data.dim_z == 6
    # Create graph
    adjacency_matrix = torch.tensor(
        [
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ]
    )
    graph = FixedGraph(adjacency_matrix, dim_z=cfg.data.dim_z)

    if cfg.data.nature.noise == "uniform":
        base_density = Uniform(-1, 1)
    elif cfg.data.nature.noise == "gaussian":
        base_density = nflows.distributions.StandardNormal((1,))
    else:
        base_density = DEFAULT_BASE_DENSITY
    # Create SCM
    scm = FixedOrderSCM(
        graph=graph,
        dim_z=cfg.data.dim_z,
        manifold_thickness=cfg.data.nature.manifold_thickness,
        base_density=base_density,
        structure_transforms=[
            ScaleTransform(scale=cfg.data.nature.noise_scale),
            ScaleTransform(scale=cfg.data.nature.noise_scale),
            ScaleTransform(scale=cfg.data.nature.noise_scale),
            ScaleTransform(scale=cfg.data.nature.noise_scale),
            ConditionalAveragingTransform(noise_scale=cfg.data.nature.child_noise_scale),
            ConditionalAveragingTransform(noise_scale=cfg.data.nature.child_noise_scale),
        ],
        intervention_transforms=[
            ScaleTransform(scale=cfg.data.nature.noise_scale),
            ScaleTransform(scale=cfg.data.nature.noise_scale),
            ScaleTransform(scale=cfg.data.nature.noise_scale),
            ScaleTransform(scale=cfg.data.nature.noise_scale),
            ScaleTransform(scale=cfg.data.nature.noise_scale),
            ScaleTransform(scale=cfg.data.nature.noise_scale),
        ],
    )

    # Create SO(n) decoder
    encoder = SONEncoder(
        coeff_std=1.0, input_features=cfg.data.dim_z, output_features=cfg.data.dim_z
    )

    # Put together
    nature = FlowLCM(scm, encoder, dim_z=cfg.data.dim_z)

    # Disable learning NAture's parameters
    for param in nature.parameters():
        param.requires_grad = False

    return nature


def load_true_model(cfg):
    """Load the data-generating model (ground truth) and regenerate if necessary"""

    # Regenerate true model if necessary
    if cfg.data.always_generate_new_data or not Path(cfg.data.data_dir).exists():
        generate_datasets(cfg)
        cfg.data.always_generate_new_data = (
            False  # We only want to regenerate data once per experiment
        )

    if cfg.data.nature.cube_movement == "1d":
        nature = create_true_model(cfg)
    if cfg.data.nature.cube_movement == "2d":
        nature = create_true_model2d(cfg)
    nature.load_state_dict(torch.load(Path(cfg.data.data_dir) / "nature.pt"))

    return nature


def generate_datasets(cfg):
    """(Re-)generate GT model weights, train and test data"""

    logger.info(f"Generating dataset at {cfg.data.data_dir}")
    data_dir = Path(cfg.data.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Generate nature object
    old_seed = torch.random.seed()
    torch.random.manual_seed(cfg.data.nature.seed)
    if cfg.data.nature.cube_movement == "1d":
        nature = create_true_model(cfg)
    if cfg.data.nature.cube_movement == "2d":
        nature = create_true_model2d(cfg)
    torch.save(nature.state_dict(), Path(cfg.data.data_dir) / "nature.pt")

    # Restore old random state
    torch.random.manual_seed(old_seed)

    # Generate datasets
    tags_samples = {
        "train": cfg.data.samples.train,
        "dci_train": cfg.data.samples.train,
        "val": cfg.data.samples.val,
        "test": cfg.data.samples.test,
    }
    for tag, n_samples in tags_samples.items():
        filename = data_dir / f"{tag}.pt"
        data = nature.sample(n_samples, additional_noise=cfg.data.nature.observation_noise)
        torch.save(data, filename)


@torch.no_grad()
def validation_loop(cfg, model, criteria, val_loader, best_state, val_metrics, step, device):
    """Validation loop, computing a number of metrics and checkpointing the best model"""

    loss, nll, metrics = compute_metrics_on_dataset(cfg, model, criteria, val_loader, device)
    metrics.update(eval_dci_scores(cfg, model, partition="val"))
    metrics.update(eval_implicit_graph(cfg, model, partition="val"))

    # Store validation metrics (and track with MLflow)
    update_dict(val_metrics, metrics)
    for key, value in metrics.items():
        mlflow.log_metric(f"val.{key}", value, step=step)

    # Print DCI disentanglement score
    logger.info(f"Step {step}: causal disentanglement = {metrics['causal_disentanglement']:.2f}")

    # Early stopping: compare val loss to last val loss
    new_val_loss = metrics["nll"] if cfg.training.early_stopping_var == "nll" else loss.item()
    if best_state["loss"] is None or new_val_loss < best_state["loss"]:
        best_state["loss"] = new_val_loss
        best_state["state_dict"] = model.state_dict().copy()
        best_state["step"] = step


def evaluate(cfg, model):
    """High-level test function"""

    logger.info("Starting evaluation")

    # Compute metrics
    test_metrics = eval_dci_scores(cfg, model, partition=cfg.eval.eval_partition)
    test_metrics.update(eval_enco_graph(cfg, model, partition=cfg.eval.eval_partition))
    test_metrics.update(eval_implicit_graph(cfg, model, partition=cfg.eval.eval_partition))
    test_metrics.update(eval_test_metrics(cfg, model))

    # Log results
    for key, val in test_metrics.items():
        mlflow.log_metric(f"eval.{key}", val)

    # Print DCI disentanglement score
    logger.info(
        f"Final evaluation: causal disentanglement = {test_metrics['causal_disentanglement']:.2f}"
    )

    # Store results in csv file
    # Pandas does not like scalar values, have to be iterables
    test_metrics_ = {key: [val] for key, val in test_metrics.items()}
    df = pd.DataFrame.from_dict(test_metrics_)
    df.to_csv(Path(cfg.general.exp_dir) / "metrics" / "test_metrics.csv")

    return test_metrics


@torch.no_grad()
def eval_test_metrics(cfg, model):
    """Evaluates loss terms on test data"""

    device = torch.device(cfg.training.device)
    model = model.to(device)

    criteria = VAEMetrics(dim_z=cfg.data.dim_z)
    test_data = load_dataset(cfg, cfg.eval.eval_partition, normalize=cfg.data.normalize)
    test_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=False)
    _, _, metrics = compute_metrics_on_dataset(
        cfg, model, criteria, test_loader, device=torch.device(cfg.training.device)
    )
    return metrics


@torch.no_grad()
def eval_dci_scores(cfg, model, partition="test", full_importance_matrix=True):
    """Evaluates DCI scores"""

    model.eval()
    device = torch.device(cfg.training.device)
    model = model.to(device)

    x_train, _, train_true_z, *_ = load_dataset(
        cfg, "dci_train", normalize=cfg.data.normalize
    ).tensors
    x_test, _, test_true_z, *_ = load_dataset(cfg, partition, normalize=cfg.data.normalize).tensors
    train_model_z = model.encode_to_causal(x_train.to(device), deterministic=True)
    test_model_z = model.encode_to_causal(x_test.to(device), deterministic=True)

    causal_dci_metrics = compute_dci(
        train_true_z,
        train_model_z,
        test_true_z,
        test_model_z,
        return_full_importance_matrix=full_importance_matrix,
    )

    renamed_metrics = {}
    for key, val in causal_dci_metrics.items():
        renamed_metrics[f"causal_{key}"] = val

    return renamed_metrics


@torch.no_grad()
def eval_implicit_graph(cfg, model, partition="val"):
    """Evaluates implicit graph"""

    # This is only defined for noise-centric models (ILCMs)
    if cfg.model.type not in ["intervention_noise_vae", "alt_intervention_noise_vae"]:
        return {}

    # Let's skip this for large latent spaces
    if cfg.model.dim_z > 5:
        return {}

    model.eval()
    device = torch.device(cfg.training.device)

    # Load data and compute noise encodings
    x, *_ = load_dataset(cfg, partition, normalize=cfg.data.normalize).tensors
    noise = model.encode_to_noise(x.to(device), deterministic=True).detach()

    # Evaluate causal strength
    causal_effects, topological_order = compute_implicit_causal_effects(model, noise)

    # Package as dict
    results = {
        f"implicit_graph_{i}_{j}": causal_effects[i, j].item()
        for i in range(model.dim_z)
        for j in range(model.dim_z)
    }

    return results


def eval_enco_graph(cfg, model, partition="train"):
    """Post-hoc graph evaluation with ENCO"""

    # Only want to do this for ILCMs
    if cfg.model.type not in ["intervention_noise_vae", "alt_intervention_noise_vae"]:
        return {}

    logger.info("Evaluating learned graph with ENCO")
    model.eval()
    device = torch.device(cfg.training.device)

    # Load data and compute noise encodings
    with torch.no_grad():
        x0, x1, *_ = load_dataset(cfg, partition, normalize=cfg.data.normalize).tensors
        _, _, _, _, e0, e1, _, _, intervention = model.encode_decode_pair(
            x0.to(device), x1.to(device)
        )
        z0 = model.scm.noise_to_causal(e0)
        z1 = model.scm.noise_to_causal(e1)

    # Run ENCO
    adjacency_matrix = (
        run_enco(z0, z1, intervention, lambda_sparse=cfg.eval.enco_lambda, device=device)
        .cpu()
        .detach()
    )
    logger.info(f"ENCO adjacency matrix: {adjacency_matrix}")

    # Package as dict
    results = {
        f"enco_graph_{i}_{j}": adjacency_matrix[i, j].item()
        for i in range(model.dim_z)
        for j in range(model.dim_z)
    }

    return results


def epoch_schedules(cfg, model, epoch, optim):
    """Epoch-based schedulers"""

    # Pretraining?
    pretrain = cfg.training.pretrain_epochs is not None and epoch < cfg.training.pretrain_epochs
    if epoch == cfg.training.pretrain_epochs:
        logger.info(f"Stopping pretraining at epoch {epoch}")

    # Model interventions in SCM / noise model?
    model_interventions = (
        cfg.training.model_interventions_after_epoch is None
        or epoch >= cfg.training.model_interventions_after_epoch
    )
    if epoch == cfg.training.model_interventions_after_epoch:
        logger.info(f"Beginning to model intervention distributions at epoch {epoch}")

    # Freeze encoder?
    if cfg.training.freeze_encoder_epoch is not None and epoch == cfg.training.freeze_encoder_epoch:
        logger.info(f"Freezing encoder and decoder at epoch {epoch}")
        optim.param_groups[0]["lr"] = 0.0

    # Fix noise-centric model to a topological order encoder?
    if (
        "fix_topological_order_epoch" in cfg.training
        and cfg.training.fix_topological_order_epoch is not None
        and epoch == cfg.training.fix_topological_order_epoch
    ):
        logger.info(f"Determining topological order at epoch {epoch}")
        fix_topological_order(cfg, model, partition="val")

    # Deterministic intervention encoders?
    if cfg.training.deterministic_intervention_encoder_after_epoch is None:
        deterministic_intervention_encoder = False
    else:
        deterministic_intervention_encoder = (
            epoch >= cfg.training.deterministic_intervention_encoder_after_epoch
        )
    if epoch == cfg.training.deterministic_intervention_encoder_after_epoch:
        logger.info(f"Switching to deterministic intervention encoder at epoch {epoch}")

    return model_interventions, pretrain, deterministic_intervention_encoder


@torch.no_grad()
def fix_topological_order(cfg, model, partition="val", dataloader=None):
    """Fixes topological order in ILCM"""

    # This is only defined for noise-centric models (ILCMs)
    assert cfg.model.type == "intervention_noise_vae"

    model.eval()
    device = torch.device(cfg.training.device)
    cpu = torch.device("cpu")
    model.to(device)

    # Dataloader
    if dataloader is None:
        dataset = load_dataset(cfg, partition, normalize=cfg.data.normalize)
        dataloader = DataLoader(dataset, batch_size=cfg.eval.batchsize, shuffle=False)

    # Load data and compute noise encodings
    noise = []
    for x_batch, *_ in dataloader:
        x_batch = x_batch.to(device)
        noise.append(model.encode_to_noise(x_batch, deterministic=True).to(cpu))

    noise = torch.cat(noise, dim=0).detach()

    # Median values of each noise component (to be used as dummy values when masking)
    dummy_values = torch.median(noise, dim=0).values
    logger.info(f"Dummy noise encodings: {dummy_values}")

    # Find topological order
    model = model.to(cpu)
    topological_order = find_topological_order(model, noise)
    logger.info(f"Topological order: {topological_order}")

    # Fix topological order
    model.scm.set_causal_structure(
        None, "fixed_order", topological_order=topological_order, mask_values=dummy_values
    )
    model.to(device)


if __name__ == "__main__":
    main()
