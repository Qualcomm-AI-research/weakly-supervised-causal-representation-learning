#!/usr/bin/env python3
# Copyright (c) 2022 Qualcomm Technologies, Inc.
# All rights reserved.

import hydra
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from pathlib import Path
import mlflow
from collections import defaultdict
from PIL import Image
from io import BytesIO

from ws_crl.encoder import ImageResNetEncoder, ImageResNetDecoder, CoordConv2d

from ws_crl.training import VAEMetrics
from ws_crl.causal.scm import (
    MLPFixedOrderSCM,
    MLPVariableOrderCausalModel,
    UnstructuredPrior,
)
from ws_crl.causal.implicit_scm import MLPImplicitSCM
from ws_crl.lcm import ELCM, ILCM
from ws_crl.utils import get_first_batch
from experiments.experiment_utils import (
    initialize_experiment,
    save_config,
    save_model,
    logger,
    create_optimizer_and_scheduler,
    set_manifold_thickness,
    reset_optimizer_state,
    create_intervention_encoder,
    log_training_step,
    optimizer_step,
    step_schedules,
    determine_graph_learning_settings,
    frequency_check,
)


@hydra.main(config_path="../config", config_name="causalcircuit_reduce_dim")
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
        load_checkpoint(cfg, model)
        train(cfg, model)
        save_model(cfg, model)
        save_representations(cfg, model)

        # Test
        plot_results(cfg, model)

    logger.info("Anders nog iets?")


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

    return model


def load_checkpoint(cfg, model):
    """Loads a model checkpoint"""
    if "load" not in cfg.model or cfg.model.load is None or cfg.model.load == "None":
        return

    filename = cfg.model.load
    logger.info(f"Loading model checkpoint from {filename}")

    state_dict = torch.load(filename, map_location="cpu")
    try:
        model.load_state_dict(state_dict)

    # Loading pretrained ILCM for beta-VAE model
    except RuntimeError:
        truncate_state_dict_for_baseline(cfg, state_dict)
        model.load_state_dict(state_dict)


def truncate_state_dict_for_baseline(cfg, state_dict):
    """Fix to allow loading a pretrained ILCM and continue training it as beta-VAE model"""

    if cfg.model.scm.type != "unstructured":
        return

    logger.warning("Removing keys from state dict to run beta-VAE")
    deleted_keys = []
    for key in list(state_dict.keys()):
        if (
            key.startswith("intervention_encoder.") or key.startswith("scm.")
        ) and key != "scm._manifold_thickness":
            del state_dict[key]
            deleted_keys.append(key)

    logger.warning(f'  Deleted keys: {", ".join(deleted_keys)}')


def create_scm(cfg):
    """Create SCM / implicit causal model"""
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

    if cfg.model.encoder.type == "conv":
        encoder = ImageResNetEncoder(
            in_resolution=cfg.model.dim_x[0],
            in_features=cfg.model.dim_x[2],
            out_features=cfg.model.dim_z,
            hidden_features=cfg.model.encoder.hidden_channels,
            batchnorm=cfg.model.encoder.batchnorm,
            conv_class=CoordConv2d if cfg.model.encoder.coordinate_embeddings else torch.nn.Conv2d,
            mlp_layers=cfg.model.encoder.extra_mlp_layers,
            mlp_hidden=cfg.model.encoder.extra_mlp_hidden_units,
            elementwise_hidden=cfg.model.encoder.elementwise_hidden_units,
            elementwise_layers=cfg.model.encoder.elementwise_layers,
            min_std=cfg.model.encoder.min_std,
            permutation=cfg.model.encoder.permutation,
        )
        decoder = ImageResNetDecoder(
            in_features=cfg.model.dim_z,
            out_resolution=cfg.model.dim_x[0],
            out_features=cfg.model.dim_x[2],
            hidden_features=cfg.model.decoder.hidden_channels,
            batchnorm=cfg.model.decoder.batchnorm,
            min_std=cfg.model.decoder.min_std,
            fix_std=cfg.model.decoder.fix_std,
            conv_class=CoordConv2d if cfg.model.decoder.coordinate_embeddings else torch.nn.Conv2d,
            mlp_layers=cfg.model.decoder.extra_mlp_layers,
            mlp_hidden=cfg.model.decoder.extra_mlp_hidden_units,
            elementwise_hidden=cfg.model.decoder.elementwise_hidden_units,
            elementwise_layers=cfg.model.decoder.elementwise_layers,
            permutation=cfg.model.encoder.permutation,
        )
    else:
        raise ValueError(f"Unknown value for encoder_cfg.type: {cfg.model.encoder.type}")

    if encoder.permutation is not None:
        logger.info(f"Encoder permutation: {encoder.permutation.detach().numpy()}")
    if decoder.permutation is not None:
        logger.info(f"Decoder permutation: {decoder.permutation.detach().numpy()}")

    return encoder, decoder


# noinspection PyTypeChecker
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

    train_loader = get_dataloader(cfg, "train", batchsize=cfg.training.batchsize, shuffle=True)
    val_loader = get_dataloader(
        cfg, "val", batchsize=cfg.eval.batchsize, shuffle=False, include_noise_encodings=True
    )
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

            # Plots
            if frequency_check(step, cfg.training.plot_every_n_steps):
                filename = (
                    Path(cfg.general.exp_dir) / "figures" / f"reconstructions_step_{step}.pdf"
                )
                plot_reconstructions(
                    cfg,
                    model,
                    val_loader,
                    device=device,
                    filename=filename,
                    artifact_folder="reconstructions",
                )
                filename = (
                    Path(cfg.general.exp_dir)
                    / "figures"
                    / f"unstructured_reconstructions_step_{step}.pdf"
                )
                plot_reconstructions(
                    cfg,
                    model,
                    val_loader,
                    device=device,
                    filename=filename,
                    artifact_folder="unstructured_reconstructions",
                    project=False,
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

    # Reset model: back to CPU, reset manifold thickness
    set_manifold_thickness(cfg, model, None)

    return train_metrics, val_metrics


def get_dataloader(cfg, tag, batchsize=None, shuffle=False, include_noise_encodings=False):
    """Load data from disk and return DataLoader instance"""
    logger.debug(f"Loading data from {cfg.data.data_dir}")
    assert tag in {"train", "test", "val"}
    if tag == "train":
        filenames = [Path(cfg.data.data_dir) / f"{tag}-{i}.npz" for i in range(10)]
    else:
        filenames = [Path(cfg.data.data_dir) / f"{tag}.npz"]

    data_parts = []
    for filename in filenames:
        assert filename.exists(), f"Dataset not found at {filename}. Consult README.md."
        data_parts.append(dict(np.load(filename)))

    data = {k: np.concatenate([data[k] for data in data_parts]) for k in data_parts[0]}

    logger.debug(f"Finished loading data from {cfg.data.data_dir}")

    dataset = CausalImageDataset(
        data,
        noise=cfg.data.noise,
        include_noise_encodings=include_noise_encodings,
    )

    batchsize = len(dataset) if batchsize is None else batchsize
    dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=shuffle)

    return dataloader


class CausalImageDataset(Dataset):
    """CausalCircuit dataset"""

    def __init__(self, npz_data, noise=None, include_noise_encodings=False):
        # Only store the data we are interested in, keeping images in uint8 for now
        self._compressed_x = npz_data["imgs"]
        self._z0 = npz_data["original_latents"][:, 0, :]
        self._z1 = npz_data["original_latents"][:, 1, :]
        self._intervention_labels = npz_data["intervention_labels"][:, np.newaxis]
        self._intervention_masks = npz_data["intervention_masks"]

        if include_noise_encodings:
            self._e0 = npz_data["epsilon"][:, 0, :]
            self._e1 = npz_data["epsilon"][:, 1, :]
        else:
            self._e0, self._e1 = None, None

        self.noise = noise

    def __getitem__(self, index):
        x0 = self._uncompress_x(index, False)
        x1 = self._uncompress_x(index, True)
        z0 = torch.FloatTensor(self._z0[index])
        z1 = torch.FloatTensor(self._z1[index])
        intervention_label = torch.LongTensor(self._intervention_labels[index])  # (1,)
        intervention_mask = torch.BoolTensor(self._intervention_masks[index])

        if self.noise is not None and self.noise > 0.0:
            # noinspection PyTypeChecker
            x0 += self.noise * torch.randn_like(x0)
            x1 += self.noise * torch.randn_like(x1)

        if self._e0 is not None:
            e0 = torch.FloatTensor(self._e0[index])
            e1 = torch.FloatTensor(self._e1[index])
            return x0, x1, z0, z1, intervention_label, intervention_mask, e0, e1

        return x0, x1, z0, z1, intervention_label, intervention_mask

    def __len__(self):
        return self._z0.shape[0]

    def _uncompress_x(self, index, after=False):
        jpeg = self._compressed_x[index, 1 if after else 0]
        buffer = BytesIO()
        buffer.write(jpeg)
        array = np.array(Image.open(buffer))
        tensor = torch.FloatTensor(np.transpose(array, (2, 0, 1))) / 255.0

        return tensor


@torch.no_grad()
def save_representations(cfg, model):
    """Reduces dimensionality for full dataset by pushing images through encoder"""
    logger.info("Encoding full datasets and storing representations")

    device = torch.device(cfg.training.device)
    model.to(device)

    for partition in ["train", "test", "val"]:
        dataloader = get_dataloader(
            cfg, partition, cfg.eval.batchsize, shuffle=False, include_noise_encodings=True
        )

        z0s, z1s, true_z0s, true_z1s = [], [], [], []
        intervention_labels, interventions, true_e0s, true_e1s = [], [], [], []

        for (
            x0,
            x1,
            true_z0,
            true_z1,
            true_intervention_label,
            true_intervention,
            true_e0,
            true_e1,
        ) in dataloader:
            x0, x1 = x0.to(device), x1.to(device)
            true_z0, true_z1 = true_z0.to(device), true_z1.to(device)

            _, _, z0, z1, *_ = model.encode_decode_pair(x0, x1)

            z0s.append(z0)
            z1s.append(z1)
            true_z0s.append(true_z0)
            true_z1s.append(true_z1)
            intervention_labels.append(true_intervention_label)
            interventions.append(true_intervention)
            true_e0s.append(true_e0)
            true_e1s.append(true_e1)

        z0s = torch.cat(z0s, dim=0)
        z1s = torch.cat(z1s, dim=0)
        true_z0s = torch.cat(true_z0s, dim=0)
        true_z1s = torch.cat(true_z1s, dim=0)
        intervention_labels = torch.cat(intervention_labels, dim=0)
        interventions = torch.cat(interventions, dim=0)
        true_e0s = torch.cat(true_e0s, dim=0)
        true_e1s = torch.cat(true_e1s, dim=0)

        data = (
            z0s,
            z1s,
            true_z0s,
            true_z1s,
            intervention_labels,
            interventions,
            true_e0s,
            true_e1s,
        )

        filename = Path(cfg.general.exp_dir).resolve() / f"data/{partition}_encoded.pt"
        logger.info(f"Storing encoded {partition} data at {filename}")
        torch.save(data, filename)


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
        # model.encoder.freeze()
        # model.decoder.freeze()

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
def plot_results(cfg, model):
    """High-level plotting function"""

    logger.info("Making plots")

    test_loader = get_dataloader(cfg, "test", batchsize=cfg.eval.batchsize)
    filename = Path(cfg.general.exp_dir) / "figures" / "reconstructions_final.pdf"
    plot_reconstructions(
        cfg,
        model,
        test_loader,
        device=torch.device(cfg.training.device),
        filename=filename,
        artifact_folder="reconstructions",
    )
    filename = Path(cfg.general.exp_dir) / "figures" / "unstructured_reconstructions_final.pdf"
    plot_reconstructions(
        cfg,
        model,
        test_loader,
        device=torch.device(cfg.training.device),
        filename=filename,
        artifact_folder="unstructured_reconstructions",
        project=False,
    )


@torch.no_grad()
def plot_reconstructions(
    cfg,
    model,
    dataloader,
    device,
    filename=None,
    artifact_folder=None,
    n_samples=12,
    noise_extent=3.0,
    project=True,
):
    """Plots a few data samples and the VAE reconstructions. Also latent representation."""

    model.eval()
    model = model.to(device)

    # What to plot?
    if cfg.model.dim_z > 4 or not project:
        latent_viz_columns = 0
    else:
        latent_viz_columns = cfg.model.dim_z * (cfg.model.dim_z - 1) // 2

    intervention_columns = 1 if project else 0
    n_cols = 4 + intervention_columns + latent_viz_columns
    n_rows = n_samples

    # Get data
    x1s, x2s, _, _, true_interventions, *_ = get_first_batch(dataloader)
    x1s, x2s, true_interventions = x1s.to(device), x2s.to(device), true_interventions.to(device)
    x1s, x2s, true_interventions = x1s[:n_samples], x2s[:n_samples], true_interventions[:n_samples]

    # Compute reconstruction
    if project:
        try:
            x1s_reco, x2s_reco, e1s, e2s, e1s_proj, e2s_proj, *_ = model.encode_decode_pair(
                x1s, x2s
            )
            e1s, e2s, e1s_proj, e2s_proj = (
                e1s.cpu().detach().numpy(),
                e2s.cpu().detach().numpy(),
                e1s_proj.cpu().detach().numpy(),
                e2s_proj.cpu().detach().numpy(),
            )
        except Exception as e:
            logger.warning(f"Error when plotting pair reco: {e}")
            return
    else:
        x1s_reco = model.encode_decode(x1s)
        x2s_reco = model.encode_decode(x2s)
        e1s, e2s, e1s_proj, e2s_proj = None, None, None, None

    # Compute intervention posterior
    _, outputs = model(x1s, x2s)
    try:
        intervention_probs = outputs[
            "intervention_posterior"
        ].squeeze()  # (batchsize, intervention)
    except KeyError:
        intervention_probs = torch.zeros((n_samples, cfg.data.dim_z + 1))

    # Initialize plot
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))

    # Plot images before and after, and also reconstructions
    for i, (x1, x2, x1r, x2r) in enumerate(zip(x1s, x2s, x1s_reco, x2s_reco)):
        axes[i, 0].imshow(x1.cpu().permute([1, 2, 0]).clamp(0, 1).mul(255).to(torch.uint8))
        axes[i, 0].set_xlabel("Before intervention, true")
        axes[i, 2].imshow(x2.cpu().permute([1, 2, 0]).clamp(0, 1).mul(255).to(torch.uint8))
        axes[i, 2].set_xlabel("After intervention, true")
        axes[i, 3].imshow(x2r.cpu().permute([1, 2, 0]).clamp(0, 1).mul(255).to(torch.uint8))
        axes[i, 3].set_xlabel("After intervention, reconstructed")
        axes[i, 1].imshow(x1r.cpu().permute([1, 2, 0]).clamp(0, 1).mul(255).to(torch.uint8))
        axes[i, 1].set_xlabel("Before intervention, reconstructed")

    for i in range(n_rows):
        for j in range(4):
            axes[i, j].axis("off")

    # Plot latents
    if e1s is not None and latent_viz_columns > 0:
        m = 0
        for j in range(cfg.model.dim_z):
            for i in range(j):
                for k, (e1, e2, e1p, e2p) in enumerate(zip(e1s, e2s, e1s_proj, e2s_proj)):
                    axes[k, 4 + m].arrow(e1[i], e1[j], e1p[i] - e1[i], e1p[j] - e1[j], color="C3")
                    axes[k, 4 + m].arrow(
                        e1p[i], e1p[j], e2p[i] - e1p[i], e2p[j] - e1p[j], color="C0"
                    )
                    axes[k, 4 + m].arrow(e2p[i], e2p[j], e2[i] - e2p[i], e2[j] - e2p[j], color="C3")
                    axes[k, 4 + m].grid(True)
                    axes[k, 4 + m].set_xlabel(f"$e_{i}$")
                    axes[k, 4 + m].set_ylabel(f"$e_{j}$")
                    axes[k, 4 + m].set_ylim(-noise_extent, noise_extent)
                    axes[k, 4 + m].set_xlim(-noise_extent, noise_extent)

                m += 1

    # Plot posterior distribution
    if project:
        for i in range(n_rows):
            range_ = np.arange(cfg.model.dim_z + 1)
            true_ys = np.zeros(cfg.model.dim_z + 1)
            true_ys[true_interventions[i].cpu().item()] = 1.0
            tick_labels = [r"$\emptyset$"] + [f"$z_{i}$" for i in range(cfg.model.dim_z)]

            axes[i, -1].bar(range_, true_ys, color="0.65", width=0.5)
            axes[i, -1].bar(range_, intervention_probs[i].detach().cpu().numpy(), color="C0")
            axes[i, -1].set_xlabel("Intervention target")
            axes[i, -1].set_ylabel("Posterior probability")
            axes[i, -1].set_xticks(range_, labels=tick_labels)

    # Finalize and save / show plot
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename)
        mlflow.log_artifact(
            filename, artifact_folder if artifact_folder is not None else Path(filename).stem
        )
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    main()
