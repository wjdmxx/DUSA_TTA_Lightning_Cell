"""Main script for running DUSA TTA."""
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import torch
from torch.utils.data import DataLoader

from src.models import create_combined_model
from src.data import create_imagenet_c_tasks, create_tta_transforms, custom_collate_fn
from src.tta import DUSATTAModule


def setup_logging(cfg: DictConfig):
    """Setup loggers (W&B, TensorBoard)."""
    loggers = []
    
    # W&B logger
    if cfg.logging.use_wandb:
        wandb_logger = WandbLogger(
            project=cfg.logging.wandb_project,
            entity=cfg.logging.wandb_entity,
            name=cfg.experiment_name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        loggers.append(wandb_logger)
    
    # TensorBoard logger
    tb_logger = TensorBoardLogger(
        save_dir=cfg.work_dir,
        name="tensorboard_logs",
    )
    loggers.append(tb_logger)
    
    return loggers


def create_model(cfg: DictConfig):
    """Create combined model from config."""
    print("=" * 80)
    print("Creating models...")
    
    # Prepare configs
    disc_config = OmegaConf.to_container(cfg.model.discriminative, resolve=True)
    gen_config = OmegaConf.to_container(cfg.model.generative, resolve=True) if cfg.model.generative else None
    
    # Create model
    model = create_combined_model(
        discriminative_config=disc_config,
        generative_config=gen_config,
    )
    
    print(f"Discriminative model: {disc_config['model_name']}")
    if gen_config:
        print(f"Generative model: {gen_config['sit_model_name']}")
    print("=" * 80)
    
    return model


def create_dataloaders(cfg: DictConfig):
    """Create dataloaders for all ImageNet-C tasks."""
    print("=" * 80)
    print("Creating dataloaders...")
    
    # Create transforms
    task_transform, raw_transform = create_tta_transforms(
        task_input_size=cfg.data.task_input_size,
        task_mean=cfg.data.task_mean,
        task_std=cfg.data.task_std,
    )
    
    # Create task datasets
    tasks = create_imagenet_c_tasks(
        root=cfg.data.root,
        corruptions=cfg.data.corruptions,
        severities=cfg.data.severities,
        task_transform=task_transform,
        raw_transform=raw_transform,
    )
    
    print(f"Created {len(tasks)} tasks")
    
    # Create dataloaders
    dataloaders = []
    for task_name, dataset in tasks:
        dataloader = DataLoader(
            dataset,
            batch_size=cfg.data.batch_size,
            shuffle=False,  # No shuffle for TTA
            num_workers=cfg.data.num_workers,
            pin_memory=cfg.data.pin_memory,
            persistent_workers=cfg.data.persistent_workers,
            collate_fn=custom_collate_fn,
        )
        dataloaders.append((task_name, dataloader))
    
    print("=" * 80)
    return dataloaders


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """Main TTA execution."""
    print("\n" + "=" * 80)
    print("DUSA Test-Time Adaptation")
    print("=" * 80)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)
    
    # Set seed
    pl.seed_everything(cfg.seed, workers=True)
    
    # Create model
    model = create_model(cfg)
    
    # Create Lightning module
    tta_module = DUSATTAModule(
        model=model,
        learning_rate=cfg.optimizer.learning_rate,
        weight_decay=cfg.optimizer.weight_decay,
        continual=cfg.tta.continual,
        update_auxiliary=cfg.tta.update_auxiliary,
        update_task_norm_only=cfg.tta.update_task_norm_only,
        tta_step=cfg.tta.tta_step,
        forward_mode=cfg.tta.forward_mode,
        log_aux_metrics=cfg.logging.log_aux_metrics,
    )
    
    # Manual optimization for better control
    tta_module.automatic_optimization = True
    
    # Setup logging
    loggers = setup_logging(cfg)
    
    # Create dataloaders
    task_dataloaders = create_dataloaders(cfg)
    
    # Run TTA on each task
    print("\n" + "=" * 80)
    print("Starting TTA on all tasks...")
    print("=" * 80)
    
    all_task_results = {}
    
    for task_idx, (task_name, task_dataloader) in enumerate(task_dataloaders):
        print(f"\n{'='*80}")
        print(f"Task {task_idx + 1}/{len(task_dataloaders)}: {task_name}")
        print(f"{'='*80}")

        # Trainer
        trainer = pl.Trainer(
            accelerator=cfg.trainer.accelerator,
            devices=cfg.trainer.devices,
            strategy=cfg.trainer.strategy,
            precision=cfg.trainer.precision,
            max_epochs=cfg.trainer.max_epochs,
            gradient_clip_val=cfg.trainer.gradient_clip_val,
            accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
            log_every_n_steps=cfg.trainer.log_every_n_steps,
            enable_progress_bar=cfg.trainer.enable_progress_bar,
            enable_model_summary=cfg.trainer.enable_model_summary,
            enable_checkpointing=cfg.trainer.enable_checkpointing,
            logger=loggers,
            deterministic=cfg.trainer.deterministic,
            benchmark=cfg.trainer.benchmark,
            fast_dev_run=cfg.trainer.fast_dev_run,
        )
        
        # Set current task info
        trainer.current_task_info = {
            "task_name": task_name,
            "task_idx": task_idx,
            "all_steps": len(task_dataloader),
        }
        
        # Reset model if not continual (except for first task)
        if not cfg.tta.continual and task_idx > 0:
            tta_module.reset_model()
        
        # Reset epoch count to allow training on new task
        trainer.fit_loop.epoch_progress.current.completed = 0

        # Run TTA on this task
        trainer.fit(tta_module, train_dataloaders=task_dataloader)
        
        # Log task completion
        if loggers:
            for logger in loggers:
                if isinstance(logger, WandbLogger):
                    logger.experiment.log({"task_completed": task_name, "task_idx": task_idx})
        
        print(f"Completed task: {task_name}")
    
    print("\n" + "=" * 80)
    print("All tasks completed!")
    print("=" * 80)
    
    # Finish logging
    if loggers:
        for logger in loggers:
            if hasattr(logger, "finalize"):
                logger.finalize("success")


if __name__ == "__main__":
    main()
