#!/usr/bin/env python3
# Copyright (c) 2022 Qualcomm Technologies, Inc.
# All rights reserved.

import json
import os
import hydra
import pandas as pd
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

from ws_crl.encoder import ImageResNetEncoder, ImageSBDecoder, CoordConv2d
from ws_crl.encoder.image_vae import ImageConvEncoder, ImageResNetDecoder
from ws_crl.metrics import compute_dci
from ws_crl.posthoc_graph_learning.enco import run_enco
from ws_crl.posthoc_graph_learning.implicit import (
    compute_implicit_causal_effects,
    find_topological_order,
)

from ws_crl.training import VAEMetrics
from ws_crl.causal.scm import (
    MLPFixedOrderSCM,
    MLPVariableOrderCausalModel,
    UnstructuredPrior,
)
from ws_crl.causal.implicit_scm import MLPImplicitSCM
from ws_crl.lcm import ELCM, ILCM
from ws_crl.utils import get_first_batch, update_dict
from experiments.experiment_utils import (
    compute_metrics_on_dataset,
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


@hydra.main(config_path="../config", config_name="2dcubes_complete")
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

        # Evaluate
        metrics = evaluate(cfg, model)

    logger.info("Noch etwas?")

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

    if cfg.model.encoder.type == "resnet":
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
    elif cfg.model.encoder.type == "conv":
        encoder = ImageConvEncoder(
            in_resolution=cfg.model.dim_x[0],
            in_features=cfg.model.dim_x[2],
            out_features=cfg.model.dim_z,
            hidden_features=cfg.model.encoder.hidden_channels,
            conv_class=CoordConv2d if cfg.model.encoder.coordinate_embeddings else torch.nn.Conv2d,
            mlp_layers=cfg.model.encoder.extra_mlp_layers,
            mlp_hidden=cfg.model.encoder.extra_mlp_hidden_units,
            elementwise_hidden=cfg.model.encoder.elementwise_hidden_units,
            elementwise_layers=cfg.model.encoder.elementwise_layers,
            min_std=cfg.model.encoder.min_std,
            permutation=cfg.model.encoder.permutation,
        )
    else:
        raise ValueError(f"Unknown value for encoder_cfg.type: {cfg.model.encoder.type}")

    if cfg.model.decoder.type == "sbd":
        decoder = ImageSBDecoder(
            in_features=cfg.model.dim_z,
            out_resolution=cfg.model.dim_x[0],
            out_features=cfg.model.dim_x[2],
            hidden_features=cfg.model.decoder.hidden_channels,
            min_std=cfg.model.decoder.min_std,
            fix_std=cfg.model.decoder.fix_std,
            conv_class=CoordConv2d if cfg.model.decoder.coordinate_embeddings else torch.nn.Conv2d,
            mlp_layers=cfg.model.decoder.extra_mlp_layers,
            mlp_hidden=cfg.model.decoder.extra_mlp_hidden_units,
            elementwise_hidden=cfg.model.decoder.elementwise_hidden_units,
            elementwise_layers=cfg.model.decoder.elementwise_layers,
            permutation=cfg.model.encoder.permutation,
        )
    elif cfg.model.decoder.type == "resnet":
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
        raise ValueError(f"Unknown value for decoder_cfg.type: {cfg.model.decoder.type}")

    if encoder.permutation is not None:
        logger.info(f"Encoder permutation: {encoder.permutation.detach().numpy()}")
    if decoder.permutation is not None:
        logger.info(f"Decoder permutation: {decoder.permutation.detach().numpy()}")

    print(f"Using encoder: {encoder} and decoder: {decoder}")

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
    recon_loader = get_dataloader(
        cfg, "val", batchsize=cfg.eval.batchsize, shuffle=True, include_noise_encodings=True
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

        for x1, x2, z1, z2, intervention_labels, *_ in train_loader:
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
            x1, x2, z1, z2, intervention_labels = (
                x1.to(device),
                x2.to(device),
                z1.to(device),
                z2.to(device),
                intervention_labels.to(device),
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

            # Plots
            if frequency_check(step, cfg.training.plot_every_n_steps):
                filename = (
                    Path(cfg.general.exp_dir) / "figures" / f"reconstructions_step_{step}.pdf"
                )
                plot_reconstructions(
                    cfg,
                    model,
                    recon_loader,
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
                    recon_loader,
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
    assert tag in {"train", "test", "val", "dci_train"}
    # if tag == "train":
    #     filenames = [Path(cfg.data.data_dir) / f"{tag}-{i}.npz" for i in range(10)]
    # else:
    #     filenames = [Path(cfg.data.data_dir) / f"{tag}.npz"]

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
        mean=cfg.data.x_mean,
        std=cfg.data.x_std,
        min=cfg.data.x_min,
        max=cfg.data.x_max,
        flatten=cfg.preprocessing.flatten,
        normalization=cfg.preprocessing.normalization.type,
        size=cfg.preprocessing.resize,
    )

    batchsize = len(dataset) if batchsize is None else batchsize
    dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=shuffle)

    return dataloader


class CausalImageDataset(Dataset):
    """CausalCircuit dataset"""

    def __init__(
        self,
        npz_data,
        noise=None,
        include_noise_encodings=False,
        # TODO: refactor this
        # size=(64, 64),
        size=None,
        mean=None,
        std=None,
        min=None,
        max=None,
        flatten=False,
        normalization="none",
    ):
        # Only store the data we are interested in, keeping images in uint8 for now
        self._x = npz_data["imgs"]
        self._z0 = npz_data["original_latents"][:, 0, :]
        self._z1 = npz_data["original_latents"][:, 1, :]
        self._intervention_labels = npz_data["intervention_labels"][:, np.newaxis]
        # self._intervention_labels = np.zeros((self._z0.shape[0], 1))
        self._intervention_masks = npz_data["intervention_masks"]
        # self._intervention_masks = np.zeros_like(npz_data["intervention_masks"])

        if size is not None:
            self._x = np.array(
                [
                    [np.array(Image.fromarray(x).resize(size, Image.BICUBIC)) for x in self._x[i]]
                    for i in range(self._x.shape[0])
                ]
            )

        if include_noise_encodings:
            self._e0 = npz_data["epsilon"][:, 0, :]
            self._e1 = npz_data["epsilon"][:, 1, :]
        else:
            self._e0, self._e1 = None, None

        self.noise = noise

        self.size = size

        self.mean = mean
        self.std = std

        self.min = min
        self.max = max

        self.flatten = flatten
        self.normalization = normalization

    def __getitem__(self, index):
        x0 = self._get_x(index, False)
        x1 = self._get_x(index, True)
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

    def _get_x(self, index, after=False):
        array = self._x[index, 1 if after else 0]
        if self.size is not None:
            array = np.array(Image.fromarray(array).resize(self.size))

        tensor = torch.FloatTensor(np.transpose(array, (2, 0, 1)))
        if self.normalization == "none":
            tensor = tensor / 255.0

        if self.normalization == "standard":
            if self.mean is None or self.std is None:
                raise ValueError("Mean and std must be provided for standard normalization")
            tensor = (tensor - torch.FloatTensor(self.mean).view(3, 1, 1)) / torch.FloatTensor(
                self.std
            ).view(3, 1, 1)

        if self.normalization == "minmax":
            if self.min is None or self.max is None:
                raise ValueError("Min and max must be provided for minmax normalization")
            tensor = (tensor - torch.FloatTensor(self.min).view(3, 1, 1)) / (
                torch.FloatTensor(self.max).view(3, 1, 1)
                - torch.FloatTensor(self.min).view(3, 1, 1)
            )

        if self.flatten:
            tensor = tensor.reshape(-1)

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
            # interventions.append(true_intervention)
            true_e0s.append(true_e0)
            true_e1s.append(true_e1)

        z0s = torch.cat(z0s, dim=0)
        z1s = torch.cat(z1s, dim=0)
        true_z0s = torch.cat(true_z0s, dim=0)
        true_z1s = torch.cat(true_z1s, dim=0)
        intervention_labels = torch.cat(intervention_labels, dim=0)
        # interventions = torch.cat(interventions, dim=0)
        true_e0s = torch.cat(true_e0s, dim=0)
        true_e1s = torch.cat(true_e1s, dim=0)

        data = (
            z0s,
            z1s,
            true_z0s,
            true_z1s,
            intervention_labels,
            # interventions,
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
    """Fixes the topological order in an ILCM"""

    # This is only defined for noise-centric models (ILCMs)
    assert cfg.model.type == "intervention_noise_vae"

    model.eval()
    device = torch.device(cfg.training.device)
    cpu = torch.device("cpu")
    model.to(device)

    # Dataloader
    if dataloader is None:
        dataloader = get_dataloader(cfg, partition, cfg.eval.batchsize)

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
        if cfg.preprocessing.normalization.type == "none":
            x1 = x1.clamp(0, 1).mul(255)
            x2 = x2.clamp(0, 1).mul(255)
            x1r = x1r.clamp(0, 1).mul(255)
            x2r = x2r.clamp(0, 1).mul(255)

        if cfg.preprocessing.normalization.type == "standard":
            x1 = x1 * torch.FloatTensor(cfg.data.x_std).view(3, 1, 1).to(
                device
            ) + torch.FloatTensor(cfg.data.x_mean).view(3, 1, 1).to(device)
            x2 = x2 * torch.FloatTensor(cfg.data.x_std).view(3, 1, 1).to(
                device
            ) + torch.FloatTensor(cfg.data.x_mean).view(3, 1, 1).to(device)
            x1r = x1r * torch.FloatTensor(cfg.data.x_std).view(3, 1, 1).to(
                device
            ) + torch.FloatTensor(cfg.data.x_mean).view(3, 1, 1).to(device)
            x2r = x2r * torch.FloatTensor(cfg.data.x_std).view(3, 1, 1).to(
                device
            ) + torch.FloatTensor(cfg.data.x_mean).view(3, 1, 1).to(device)

        if cfg.preprocessing.normalization.type == "minmax":
            x1 = x1 * (
                torch.FloatTensor(cfg.data.x_max).view(3, 1, 1)
                - torch.FloatTensor(cfg.data.x_min).view(3, 1, 1)
            ).to(device) + torch.FloatTensor(cfg.data.x_min).view(3, 1, 1).to(device)
            x2 = x2 * (
                torch.FloatTensor(cfg.data.x_max).view(3, 1, 1)
                - torch.FloatTensor(cfg.data.x_min).view(3, 1, 1)
            ).to(device) + torch.FloatTensor(cfg.data.x_min).view(3, 1, 1).to(device)
            x1r = x1r * (
                torch.FloatTensor(cfg.data.x_max).view(3, 1, 1)
                - torch.FloatTensor(cfg.data.x_min).view(3, 1, 1)
            ).to(device) + torch.FloatTensor(cfg.data.x_min).view(3, 1, 1).to(device)
            x2r = x2r * (
                torch.FloatTensor(cfg.data.x_max).view(3, 1, 1)
                - torch.FloatTensor(cfg.data.x_min).view(3, 1, 1)
            ).to(device) + torch.FloatTensor(cfg.data.x_min).view(3, 1, 1).to(device)

        if cfg.preprocessing.flatten:
            x1 = x1.reshape(cfg.model.dim_x[2], cfg.model.dim_x[0], cfg.model.dim_x[1])
            x2 = x2.reshape(cfg.model.dim_x[2], cfg.model.dim_x[0], cfg.model.dim_x[1])
            x1r = x1r.reshape(cfg.model.dim_x[2], cfg.model.dim_x[0], cfg.model.dim_x[1])
            x2r = x2r.reshape(cfg.model.dim_x[2], cfg.model.dim_x[0], cfg.model.dim_x[1])

        axes[i, 0].imshow(x1.cpu().permute([1, 2, 0]).clamp(0, 255).to(torch.uint8))
        axes[i, 0].set_xlabel("Before intervention, true")
        axes[i, 2].imshow(x2.cpu().permute([1, 2, 0]).clamp(0, 255).to(torch.uint8))
        axes[i, 2].set_xlabel("After intervention, true")
        axes[i, 3].imshow(x2r.cpu().permute([1, 2, 0]).clamp(0, 255).to(torch.uint8))
        axes[i, 3].set_xlabel("After intervention, reconstructed")
        axes[i, 1].imshow(x1r.cpu().permute([1, 2, 0]).clamp(0, 255).to(torch.uint8))
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


@torch.no_grad()
def validation_loop(cfg, model, criteria, val_loader, best_state, val_metrics, step, device):
    """Validation loop, computing a number of metrics and checkpointing the best model"""

    loss, nll, metrics = compute_metrics_on_dataset(cfg, model, criteria, val_loader, device)
    metrics.update(eval_dci_scores(cfg, model, test_loader=val_loader))
    metrics.update(eval_implicit_graph(cfg, model, dataloader=val_loader))

    # Store validation metrics (and track with MLflow)
    update_dict(val_metrics, metrics)
    for key, value in metrics.items():
        mlflow.log_metric(f"val.{key}", value, step=step)

    # Print DCI disentanglement score
    logger.info(
        f"Step {step}: causal disentanglement = {metrics['causal_disentanglement']:.2f}, "
        f"noise disentanglement = {metrics['noise_disentanglement']:.2f}"
    )

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
        f"Final evaluation: causal disentanglement = {test_metrics['causal_disentanglement']:.2f}, "
        f"noise disentanglement = {test_metrics['noise_disentanglement']:.2f}"
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
    test_loader = get_dataloader(
        cfg, cfg.eval.eval_partition, batchsize=cfg.eval.batchsize, shuffle=False
    )
    _, _, metrics = compute_metrics_on_dataset(
        cfg, model, criteria, test_loader, device=torch.device(cfg.training.device)
    )
    return metrics


@torch.no_grad()
def eval_dci_scores(
    cfg,
    model,
    partition="val",
    test_loader=None,
    dci_train_loader=None,
    full_importance_matrix=True,
):
    """Evaluates DCI scores"""

    model.eval()
    device = torch.device(cfg.training.device)
    cpu = torch.device("cpu")
    model = model.to(device)

    def _load(partition, device, out_device, dataloader=None):
        if dataloader is None:
            dataloader = get_dataloader(
                cfg, partition, cfg.eval.batchsize, include_noise_encodings=True
            )

        model_z, true_z = [], []
        model_e, true_e = [], []

        for x_batch, _, true_z_batch, *_, true_e_batch, _ in dataloader:
            x_batch = x_batch.to(device)

            z_batch = model.encode_to_causal(x_batch, deterministic=True)
            e_batch = model.encode_to_noise(x_batch, deterministic=True)

            model_z.append(z_batch.to(out_device))
            model_e.append(e_batch.to(out_device))
            true_z.append(true_z_batch.to(out_device))
            true_e.append(true_e_batch.to(out_device))

        model_z = torch.cat(model_z, dim=0).detach()
        true_z = torch.cat(true_z, dim=0).detach()
        model_e = torch.cat(model_e, dim=0).detach()
        true_e = torch.cat(true_e, dim=0).detach()
        # return true_z, model_z, true_e, model_e
        return true_z.squeeze(), model_z.squeeze(), true_e.squeeze(), model_e.squeeze()

    # train_true_z, train_model_z, train_true_e, train_model_e = _load("dci_train", device, cpu)
    train_true_z, train_model_z, train_true_e, train_model_e = (
        _load("dci_train", device, cpu)
        if dci_train_loader is None
        else _load("dci_train", device, cpu, dataloader=dci_train_loader)
    )
    test_true_z, test_model_z, test_true_e, test_model_e = _load(
        partition, device, cpu, dataloader=test_loader
    )

    causal_dci_metrics = compute_dci(
        train_true_z,
        train_model_z,
        test_true_z,
        test_model_z,
        return_full_importance_matrix=full_importance_matrix,
    )
    noise_dci_metrics = compute_dci(
        train_true_e,
        train_model_e,
        test_true_e,
        test_model_e,
        return_full_importance_matrix=full_importance_matrix,
    )

    combined_metrics = {}
    for key, val in noise_dci_metrics.items():
        combined_metrics[f"noise_{key}"] = val
    for key, val in causal_dci_metrics.items():
        combined_metrics[f"causal_{key}"] = val

    return combined_metrics


@torch.no_grad()
def eval_implicit_graph(cfg, model, partition="val", dataloader=None):
    """Evaluates implicit graph"""

    # This is only defined for noise-centric models (ILCMs)
    if cfg.model.type not in ["intervention_noise_vae", "alt_intervention_noise_vae"]:
        return {}

    # Let's skip this for large latent spaces
    if cfg.model.dim_z > 5:
        return {}

    model.eval()
    device = torch.device(cfg.training.device)
    cpu = torch.device("cpu")

    # Dataloader
    if dataloader is None:
        dataloader = get_dataloader(cfg, partition, cfg.eval.batchsize)

    # Load data and compute noise encodings
    noise = []
    for x_batch, *_ in dataloader:
        x_batch = x_batch.to(device)
        noise.append(model.encode_to_noise(x_batch, deterministic=True).to(cpu))

    noise = torch.cat(noise, dim=0).detach()

    # Evaluate causal strength
    model = model.to(cpu)
    causal_effects, topological_order = compute_implicit_causal_effects(model, noise)

    # Package as dict
    results = {
        f"implicit_graph_{i}_{j}": causal_effects[i, j].item()
        for i in range(model.dim_z)
        for j in range(model.dim_z)
    }

    model.to(device)

    return results


def eval_enco_graph(cfg, model, partition="train"):
    """Post-hoc graph evaluation with ENCO"""

    # Only want to do this for ILCMs
    if cfg.model.type not in ["intervention_noise_vae", "alt_intervention_noise_vae"]:
        return {}

    # Let's skip this for large latent spaces
    if cfg.model.dim_z > 5:
        return {}

    logger.info("Evaluating learned graph with ENCO")

    model.eval()
    device = torch.device(cfg.training.device)
    cpu = torch.device("cpu")
    model.to(device)

    # Load data and compute causal variables
    dataloader = get_dataloader(cfg, partition, cfg.eval.batchsize)
    z0s, z1s, interventions = [], [], []

    with torch.no_grad():
        for x0, x1, *_ in dataloader:
            x0, x1 = x0.to(device), x1.to(device)
            _, _, _, _, e0, e1, _, _, intervention = model.encode_decode_pair(
                x0.to(device), x1.to(device)
            )
            z0 = model.scm.noise_to_causal(e0)
            z1 = model.scm.noise_to_causal(e1)

            z0s.append(z0.to(cpu))
            z1s.append(z1.to(cpu))
            interventions.append(intervention.to(cpu))

        z0s = torch.cat(z0s, dim=0).detach()
        z1s = torch.cat(z1s, dim=0).detach()
        interventions = torch.cat(interventions, dim=0).detach()

    # Run ENCO
    adjacency_matrix = (
        run_enco(z0s, z1s, interventions, lambda_sparse=cfg.eval.enco_lambda, device=device)
        .cpu()
        .detach()
    )

    # Package as dict
    results = {
        f"enco_graph_{i}_{j}": adjacency_matrix[i, j].item()
        for i in range(model.dim_z)
        for j in range(model.dim_z)
    }

    return results


if __name__ == "__main__":
    main()
