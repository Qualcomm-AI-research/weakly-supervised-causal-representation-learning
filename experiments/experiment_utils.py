# Copyright (c) 2022 Qualcomm Technologies, Inc.
# All rights reserved.

import datetime
import logging
from pathlib import Path
import os
import mlflow
import numpy as np
import torch
from itertools import chain
from omegaconf import OmegaConf
from collections import defaultdict

from ws_crl.causal.interventions import HeuristicInterventionEncoder
from ws_crl.nets import make_mlp
from ws_crl.plotting import init_plt
from ws_crl.utils import flatten_dict, update_dict

logger = logging.getLogger(__name__)
logging_initialized = False


# noinspection PyUnresolvedReferences
def initialize_experiment(cfg):
    """Initialize experiment folder (and plenty of other initialization thingies)"""

    # Initialize exp folder and logger
    initialize_experiment_folder(cfg)
    initialize_logger(cfg)

    # Set up MLflow tracking location
    Path(cfg.general.mlflow.db).parent.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{Path(cfg.general.mlflow.db).resolve()}")

    # Create MLflow experiment if it doesn't exist already
    Path(cfg.general.mlflow.artifacts).mkdir(exist_ok=True)
    try:
        experiment_id = mlflow.create_experiment(
            cfg.general.exp_name,
            artifact_location=f"file:{Path(cfg.general.mlflow.artifacts).resolve()}",
        )
        logger.info(f"Created experiment {cfg.general.exp_name} with ID {experiment_id}")
    except mlflow.exceptions.MlflowException:
        pass  # Experiment exists already

    # Set MLflow experiment details
    experiment = mlflow.set_experiment(cfg.general.exp_name)
    experiment_id = experiment.experiment_id
    artifact_loc = experiment.artifact_location
    logger.info(
        f"Set experiment {cfg.general.exp_name} with ID {experiment_id}, artifact location {artifact_loc}"
    )

    # Silence other loggers (thanks MLflow)
    silence_the_lambs()

    # Re-initialize logger - this is annoying, something in the way that MLflow interacts with logging means that
    # we otherwise get duplicate logging to stdout
    initialize_logger(cfg)

    # Print config to log
    logger.info(f"Running experiment at {cfg.general.exp_dir}")
    logger.info(f"Config: \n{OmegaConf.to_yaml(cfg)}")

    # Set random seed
    torch.random.manual_seed(cfg.general.seed)
    np.random.seed(cfg.general.seed)

    # Initialize plotting
    init_plt()

    return experiment_id


def initialize_experiment_folder(cfg):
    """Creates experiment folder"""

    # Check experiment folder
    exp_dir = Path(cfg.general.exp_dir).resolve()
    if exp_dir.exists():
        if cfg.general.overwrite_existing:
            backup_dir = (
                exp_dir.parent
                / f'{exp_dir.stem}_backup_{datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}'
            )
            exp_dir.rename(backup_dir)
        else:
            raise RuntimeError(f"Experiment directory {exp_dir} already exists")

    # Create experiment folder and all necessary subfolders
    directories = [
        exp_dir,
        exp_dir / "models",
        exp_dir / "figures",
        exp_dir / "metrics",
        exp_dir / "data",
    ]
    for path in directories:
        path.mkdir(parents=True)


def initialize_logger(cfg):
    """Initializes logging"""

    global logging_initialized

    # In sweeps (multiple experiments in one job) we don't want to set up the handlers again
    if logging_initialized:
        logger.info("Logger already initialized - hi again!")
        return

    logger.setLevel(logging.DEBUG if cfg.general.debug else logging.INFO)
    formatter = logging.Formatter(
        "[%(asctime)-19.19s %(levelname)-1.1s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler = logging.FileHandler(Path(cfg.general.exp_dir) / "output.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # This is important to avoid duplicate log outputs
    # See https://stackoverflow.com/questions/7173033/duplicate-log-output-when-using-python-logging-module
    logger.propagate = False

    logging_initialized = True
    logger.info("Hoi.")


def save_config(cfg):
    """Stores the config in the experiment folder and tracks it with mlflow"""

    # Save config
    config_filename = Path(cfg.general.exp_dir) / "config.yml"
    logger.info(f"Saving config at {config_filename}")
    with open(config_filename, "w") as file:
        file.write(OmegaConf.to_yaml(cfg))

    # Store config as MLflow params
    for key, value in flatten_dict(cfg).items():
        mlflow.log_param(key, value)

    # Store cluster job ID
    if job_id := os.environ.get("LSB_JOBID"):
        mlflow.log_param("job_id", job_id)


def save_model(cfg, model, filename="model.pt"):
    """Saves the model's state dict"""

    # Store model in experiment folder
    model_path = Path(cfg.general.exp_dir) / "models" / filename
    logger.debug(f"Saving model at {model_path}")
    torch.save(model.state_dict(), model_path)


def compute_metrics_on_dataset(cfg, model, criteria, data_loader, device):
    """Computes metrics on a full dataset"""
    # At test time, always use the canonical manifold thickness
    model.eval()
    set_manifold_thickness(cfg, model, None)

    nll, samples = 0.0, 0
    loss = 0.0
    metrics = None
    batches = 0

    # Loop over batches
    for x1, x2, z1, z2, intervention_labels, true_interventions, *_ in data_loader:
        batches += 1

        x1, x2, z1, z2, intervention_labels, true_interventions = (
            x1.to(device),
            x2.to(device),
            z1.to(device),
            z2.to(device),
            intervention_labels.to(device),
            true_interventions.to(device),
        )
        log_prob, model_outputs = model(
            x1,
            x2,
            beta=cfg.eval.beta,
            full_likelihood=cfg.eval.full_likelihood,
            likelihood_reduction=cfg.eval.likelihood_reduction,
            graph_mode=cfg.eval.graph_sampling.mode,
            graph_temperature=cfg.eval.graph_sampling.temperature,
            graph_samples=cfg.eval.graph_sampling.samples,
        )

        batch_loss, batch_metrics = criteria(
            log_prob,
            true_interventions=true_interventions,
            true_intervention_labels=intervention_labels,
            **model_outputs,
        )
        batch_log_likelihood = torch.mean(
            model.log_likelihood(
                x1,
                x2,
                n_latent_samples=cfg.training.iwae_samples,
                beta=cfg.eval.beta,
                full_likelihood=cfg.eval.full_likelihood,
                likelihood_reduction=cfg.eval.likelihood_reduction,
                graph_mode=cfg.eval.graph_sampling.mode,
                graph_temperature=cfg.eval.graph_sampling.temperature,
                graph_samples=cfg.eval.graph_sampling.samples,
            )
        ).item()

        # Tally up metrics
        loss += batch_loss
        if metrics is None:
            metrics = batch_metrics
        else:
            for key, val in metrics.items():
                metrics[key] += batch_metrics[key]
        nll -= batch_log_likelihood

    # Average over batches
    loss /= batches
    for key, val in metrics.items():
        metrics[key] = val / batches
    metrics["nll"] = nll / batches

    return loss, nll, metrics


def create_optimizer_and_scheduler(cfg, model, separate_param_groups=False):
    """Initializes optimizer and scheduler"""

    if separate_param_groups:
        encoder_params = chain(
            model.encoder.freezable_parameters(), model.decoder.freezable_parameters()
        )
        try:
            other_params = chain(
                model.encoder.unfreezable_parameters(),
                model.decoder.unfreezable_parameters(),
                model.intervention_encoder.parameters(),
                model.scm.parameters(),
                model.intervention_prior.parameters(),
            )
        except AttributeError:  # Using a model without intervention encoder
            other_params = chain(
                model.encoder.unfreezable_parameters(),
                model.decoder.unfreezable_parameters(),
                model.scm.parameters(),
                model.intervention_prior.parameters(),
            )

        optim = torch.optim.Adam(
            [{"params": encoder_params}, {"params": other_params}],
            lr=cfg.training.lr_schedule.initial,
        )
    else:
        optim = torch.optim.Adam(model.parameters(), lr=cfg.training.lr_schedule.initial)

    if cfg.training.lr_schedule.type == "constant":
        scheduler = None
    elif cfg.training.lr_schedule.type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, cfg.training.epochs, eta_min=cfg.training.lr_schedule.minimal
        )
    elif cfg.training.lr_schedule.type in ["cosine_restarts", "cosine_restarts_reset"]:
        try:
            t_mult = cfg.training.lr_schedule.increase_period_by_factor
        except:
            t_mult = 1.0
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optim,
            cfg.training.lr_schedule.restart_every_epochs,
            eta_min=cfg.training.lr_schedule.minimal,
            T_mult=t_mult,
        )
    elif cfg.training.lr_schedule.type == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optim,
            step_size=cfg.training.lr_schedule.step_every_epochs,
            gamma=cfg.training.lr_schedule.step_gamma,
        )
    else:
        raise ValueError(
            f"Unknown value in cfg.training.lr_schedule: {cfg.training.lr_schedule.type}"
        )

    return optim, scheduler


def reset_optimizer_state(optimizer):
    """Resets optimizer state"""
    optimizer.__setstate__({"state": defaultdict(dict)})


def set_manifold_thickness(cfg, model, epoch):
    """For models with non-zero manifold thickness, set that according to a scheduler"""
    manifold_thickness = generic_scheduler(cfg, cfg.training.manifold_thickness_schedule, epoch)
    if manifold_thickness is None:  # Reset manifold thickness
        manifold_thickness = cfg.model.scm.manifold_thickness
    model.scm.manifold_thickness = manifold_thickness


def exponential_scheduler(step, total_steps, initial, final):
    """Exponential scheduler"""

    if step >= total_steps:
        return final
    if step <= 0:
        return initial
    if total_steps <= 1:
        return final

    t = step / (total_steps - 1)
    log_value = (1.0 - t) * np.log(initial) + t * np.log(final)
    return np.exp(log_value)


def linear_scheduler(step, total_steps, initial, final):
    """Linear scheduler"""

    if step >= total_steps:
        return final
    if step <= 0:
        return initial
    if total_steps <= 1:
        return final

    t = step / (total_steps - 1)
    return (1.0 - t) * initial + t * final


def generic_scheduler(cfg, schedule_cfg, epoch, default_value=None):
    """Generic scheduler (wraps around constant / exponential / ... schedulers)"""
    if epoch is None:
        return default_value
    elif schedule_cfg.type == "constant":
        return schedule_cfg.final
    elif schedule_cfg.type == "constant_constant":
        if epoch < schedule_cfg.initial_constant_epochs:
            return schedule_cfg.initial
        else:
            return schedule_cfg.final
    elif schedule_cfg.type == "exponential":
        return exponential_scheduler(
            epoch, cfg.training.epochs, schedule_cfg.initial, schedule_cfg.final
        )
    elif schedule_cfg.type == "exponential_constant":
        return exponential_scheduler(
            epoch,
            schedule_cfg.decay_epochs,
            schedule_cfg.initial,
            schedule_cfg.final,
        )
    elif schedule_cfg.type == "constant_exponential_constant":
        return exponential_scheduler(
            epoch - schedule_cfg.initial_constant_epochs,
            schedule_cfg.decay_epochs,
            schedule_cfg.initial,
            schedule_cfg.final,
        )
    elif schedule_cfg.type == "constant_linear_constant":
        return linear_scheduler(
            epoch - schedule_cfg.initial_constant_epochs,
            schedule_cfg.decay_epochs,
            schedule_cfg.initial,
            schedule_cfg.final,
        )
    else:
        raise ValueError(f"Unknown scheduler type: {schedule_cfg.type}")


def silence_the_lambs():
    """Silences other loggers"""
    # noinspection PyUnresolvedReferences
    for name, other_logger in logging.root.manager.loggerDict.items():
        if not "experiment_utils" in name:
            other_logger.level = logging.WARNING


def create_intervention_encoder(cfg):
    """Creates an intervention encoder"""
    logger.info(f"Creating {cfg.model.intervention_encoder.type} intervention encoder")

    if cfg.model.intervention_encoder.type == "learnable_heuristic":
        intervention_encoder = HeuristicInterventionEncoder()

    elif cfg.model.intervention_encoder.type == "mlp":
        n_interventions = cfg.model.dim_z + 1  # atomic or empty interventions
        features = (
            [2 * cfg.model.dim_z]
            + [cfg.model.intervention_encoder.hidden_units]
            * cfg.model.intervention_encoder.hidden_layers
            + [n_interventions]
        )
        intervention_encoder = make_mlp(features, final_activation="softmax")
    else:
        raise ValueError(
            f"Unknown value for cfg.model.intervention_encoder.type: "
            f"{cfg.model.intervention_encoder.type}"
        )

    return intervention_encoder


def log_training_step(
    cfg,
    beta,
    epoch_generator,
    finite,
    grad_norm,
    metrics,
    model,
    step,
    train_metrics,
    nan_counter=0,
):
    """Logs metrics from a training step"""
    metrics["grad_norm"] = grad_norm
    metrics["beta"] = beta

    for key, value in model.scm.get_scm_parameters().items():
        try:
            metrics[f"scm_{key}"] = value.detach().cpu().item()
        except AttributeError:  # Some numbers may already be ints or floats
            metrics[f"scm_{key}"] = float(value)
    try:
        for key, value in model.intervention_encoder.get_parameters().items():
            metrics[f"intervention_encoder_{key}"] = value.detach().cpu().item()
    except:
        pass

    update_dict(train_metrics, metrics)

    if cfg.training.log_every_n_steps is not None:
        if not finite:  # When facing NaNs, log every step without averaging
            for key, values in train_metrics.items():
                mlflow.log_metric(f"train.{key}", values[-1], step=step)

        elif step == 1 or step % cfg.training.log_every_n_steps == 0:
            for key, values in train_metrics.items():
                rolling_average = np.mean(values[-cfg.training.log_every_n_steps :])
                mlflow.log_metric(f"train.{key}", rolling_average, step=step)

    # Update progress bar with rolling average of loss
    if cfg.general.verbose:
        rolling_average = np.mean(train_metrics["loss"][-cfg.training.log_every_n_steps :])
        epoch_generator.set_description(f"Loss = {rolling_average:.2f}")

    elif cfg.training.print_every_n_steps is not None and (
        step % cfg.training.print_every_n_steps == 0 or step == 1
    ):
        rolling_average = np.mean(train_metrics["loss"][-cfg.training.print_every_n_steps :])
        logger.info(f"Step {step}: Loss = {rolling_average:.2f}")

    # Abort training if too many NaNs
    if nan_counter >= 10:
        logger.error(f"Hit {nan_counter} NaNs, aborting training")
        raise RuntimeError("NaNs everywhere")
    elif not finite:
        logger.warning("Continuing training anyway, hoping for better times...")


def optimizer_step(cfg, loss, model, model_outputs, optim, x1, x2):
    """Optimizer step, plus some logging and NaN handling"""
    finite = torch.isfinite(loss)

    if finite:
        optim.zero_grad()
        loss.backward()
        try:
            grad_norm = (
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.training.clip_grad_norm, error_if_nonfinite=True
                )
                .cpu()
                .item()
            )
        except RuntimeError:  # NaN in gradients
            finite = False
            grad_norm = float("inf")
            logger.warning("NaN in gradients!")

    else:
        grad_norm = float("inf")
        logger.warning("NaN in loss!")

    if finite:
        optim.step()
    else:
        logger.warning("NaN detected!")
    return finite, grad_norm


def step_schedules(cfg, model, fractional_epoch):
    """Step-based schedulers"""

    set_manifold_thickness(cfg, model, fractional_epoch)
    beta = generic_scheduler(cfg, cfg.training.beta_schedule, fractional_epoch, default_value=1.0)
    beta_intervention = cfg.training.increase_intervention_beta * beta
    consistency_regularization_amount = generic_scheduler(
        cfg, cfg.training.consistency_regularization_schedule, fractional_epoch, default_value=0.0
    )
    inverse_consistency_regularization_amount = generic_scheduler(
        cfg,
        cfg.training.inverse_consistency_regularization_schedule,
        fractional_epoch,
        default_value=0.0,
    )
    z_regularization_amount = generic_scheduler(
        cfg, cfg.training.z_regularization_schedule, fractional_epoch, default_value=0.0
    )
    edge_regularization_amount = generic_scheduler(
        cfg, cfg.training.edge_regularization_schedule, fractional_epoch, default_value=0.0
    )
    cyclicity_regularization_amount = generic_scheduler(
        cfg, cfg.training.cyclicity_regularization_schedule, fractional_epoch, default_value=0.0
    )
    intervention_entropy_regularization_amount = generic_scheduler(
        cfg,
        cfg.training.intervention_entropy_regularization_schedule,
        fractional_epoch,
        default_value=0.0,
    )
    intervention_encoder_offset = generic_scheduler(
        cfg, cfg.training.intervention_encoder_offset_schedule, fractional_epoch, default_value=0.0
    )
    return (
        beta,
        beta_intervention,
        consistency_regularization_amount,
        cyclicity_regularization_amount,
        edge_regularization_amount,
        inverse_consistency_regularization_amount,
        z_regularization_amount,
        intervention_entropy_regularization_amount,
        intervention_encoder_offset,
    )


def determine_graph_learning_settings(cfg, epoch, model):
    """Put together kwargs for graph parameterization"""

    if model.scm.graph is None:
        return {}

    if epoch < cfg.training.graph_sampling.initial.unfreeze_epoch:
        if epoch == 0:
            logger.info(
                f"Freezing adjacency matrix initially. Value:\n{model.scm.graph.adjacency_matrix}"
            )
            model.scm.graph.freeze()
        graph_kwargs = dict(
            graph_mode=cfg.training.graph_sampling.mode,
            graph_temperature=cfg.training.graph_sampling.temperature,
            graph_samples=cfg.training.graph_sampling.samples,
        )
    elif epoch >= cfg.training.graph_sampling.final.freeze_epoch:
        if epoch == cfg.training.graph_sampling.final.freeze_epoch:
            logger.info(
                f"Freezing adjacency matrix after epoch {epoch}. Value:\n"
                f"{model.scm.graph.adjacency_matrix}"
            )
            model.scm.graph.freeze()
        graph_kwargs = dict(
            graph_mode=cfg.training.graph_sampling.final.mode,
            graph_temperature=cfg.training.graph_sampling.final.temperature,
            graph_samples=cfg.training.graph_sampling.final.samples,
        )
    else:
        if epoch == cfg.training.graph_sampling.initial.unfreeze_epoch:
            logger.info(
                f"Unfreezing adjacency matrix after epoch {epoch}. Value:\n"
                f"{model.scm.graph.adjacency_matrix}"
            )
            model.scm.graph.unfreeze()
        graph_kwargs = dict(
            graph_mode=cfg.training.graph_sampling.mode,
            graph_temperature=cfg.training.graph_sampling.temperature,
            graph_samples=cfg.training.graph_sampling.samples,
        )

    return graph_kwargs


def frequency_check(step, every_n_steps):
    """Checks whether certain frequency-based schedules should trigger"""

    if every_n_steps is None or every_n_steps == 0:
        return False

    return step % every_n_steps == 0 or step == 1
