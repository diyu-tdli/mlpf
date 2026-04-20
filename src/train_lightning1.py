#!/usr/bin/env python

import os
import sys
import glob
import torch
import lightning as L
from lightning.pytorch.loggers import WandbLogger

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from src.utils.parser_args import parser
from src.utils.train_utils import (
    train_load,
    test_load,
    get_samples_steps_per_epoch,
    model_setup,
    set_gpus,
)
from src.utils.load_pretrained_models import (
    load_train_model,
    load_test_model,
)
from src.utils.callbacks import (
    get_callbacks,
    get_callbacks_eval,
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def setup_wandb(args):
    return WandbLogger(
        project=args.wandb_projectname,
        entity=args.wandb_entity,
        name=args.wandb_displayname,
        log_model="all",
    )


def build_trainer(args, gpus, logger, training=True):
    callbacks = get_callbacks(args) if training else get_callbacks_eval(args)

    strategy = "ddp_find_unused_parameters_true" if (training and args.correction) else "ddp" if training else None

    trainer_kwargs = dict(
        callbacks=callbacks,
        accelerator="gpu",
        devices=gpus,
        default_root_dir=args.model_prefix,
        logger=logger,
        max_epochs=args.num_epochs if training else None,
        limit_train_batches=args.train_batches if training else None,
        limit_val_batches=5 if training else None,
    )
    if strategy is not None:
        trainer_kwargs["strategy"] = strategy

    return L.Trainer(**trainer_kwargs)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    args = parser.parse_args()
    torch.autograd.set_detect_anomaly(False)

    training_mode = not args.predict
    args.local_rank = 0
    args.is_muons = True

    # --------------------------------------------------
    # Data
    # --------------------------------------------------
    args = get_samples_steps_per_epoch(args)

    if training_mode:
        args.data_train = glob.glob(args.data_train[0] + "*.parquet")
        train_loader, val_loader, data_config, train_input_names = train_load(args)
    else:
        test_loaders, data_config = test_load(args)

    # --------------------------------------------------
    # Model & devices
    # --------------------------------------------------
    model = model_setup(args, data_config)
    gpus, dev = set_gpus(args)

    if training_mode and args.load_model_weights:
        model = load_train_model(args, dev)

    # --------------------------------------------------
    # Logger
    # --------------------------------------------------
    wandb_logger = setup_wandb(args)

    # --------------------------------------------------
    # Training
    # --------------------------------------------------
    if training_mode:
        trainer = build_trainer(args, gpus, wandb_logger, training=True)
        args.local_rank = trainer.global_rank

        trainer.fit(
            model=model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )

    # --------------------------------------------------
    # Evaluation
    # --------------------------------------------------
    if args.data_test:
        if args.load_model_weights:
            model = load_test_model(args, dev)

        trainer = build_trainer(args, gpus, wandb_logger, training=False)

        for name, get_test_loader in test_loaders.items():
            test_loader = get_test_loader()
            trainer.validate(
                model=model,
                dataloaders=test_loader,
            )


if __name__ == "__main__":
    main()
