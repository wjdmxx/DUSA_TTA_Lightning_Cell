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
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, Callback
import torch
from torch.utils.data import DataLoader

from src.models import create_combined_model
from src.data import create_imagenet_c_tasks, create_tta_transforms, custom_collate_fn
from src.tta import DUSATTAModule


class TaskSwitchCallback(Callback):
    """Callback to handle task switching in TTA."""

    def __init__(
        self, task_name: str, task_idx: int, total_tasks: int, continual: bool = False
    ):
        super().__init__()
        self.task_name = task_name
        self.task_idx = task_idx
        self.total_tasks = total_tasks
        self.continual = continual

    def on_train_start(self, trainer, pl_module):
        """Reset model at the start of each task if not continual."""
        # Set task info on module
        pl_module.set_task_info(self.task_name, self.task_idx)

        # Reset model if not continual (except for first task)
        if not self.continual and self.task_idx > 0:
            pl_module.reset_model()
            # Need to reconfigure optimizer after model reset
            trainer.strategy.setup_optimizers(trainer)


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
    gen_config = (
        OmegaConf.to_container(cfg.model.generative, resolve=True)
        if cfg.model.generative
        else None
    )

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
            shuffle=cfg.data.get("shuffle", False),
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
        forward_mode=cfg.tta.forward_mode,
        log_aux_metrics=cfg.logging.log_aux_metrics,
        num_classes=cfg.model.discriminative.num_classes,
    )

    # Save initial model state before any TTA
    tta_module.save_initial_state()

    # Setup logging (create loggers once)
    loggers = setup_logging(cfg)

    # Create dataloaders
    task_dataloaders = create_dataloaders(cfg)
    total_tasks = len(task_dataloaders)

    # Run TTA on each task
    print("\n" + "=" * 80)
    print("Starting TTA on all tasks...")
    print("=" * 80)

    all_task_results = {}

    for task_idx, (task_name, task_dataloader) in enumerate(task_dataloaders):
        print(f"\n{'='*80}")
        print(f"Task {task_idx + 1}/{total_tasks}: {task_name}")
        print(f"{'='*80}")

        # Create task-specific callback for model reset
        task_callback = TaskSwitchCallback(
            task_name=task_name,
            task_idx=task_idx,
            total_tasks=total_tasks,
            continual=cfg.tta.continual,
        )

        # Create trainer for this task
        # Note: We create a new trainer per task to properly reset training state,
        # but we reuse the same loggers to maintain consistent logging
        trainer = pl.Trainer(
            accelerator=cfg.trainer.accelerator,
            devices=cfg.trainer.devices,
            strategy=cfg.trainer.strategy,
            precision=cfg.trainer.precision,
            max_epochs=1,  # Always 1 epoch per task
            gradient_clip_val=cfg.trainer.gradient_clip_val,
            accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
            log_every_n_steps=cfg.trainer.log_every_n_steps,
            enable_progress_bar=cfg.trainer.enable_progress_bar,
            enable_model_summary=(task_idx == 0),  # Only show summary for first task
            enable_checkpointing=cfg.trainer.enable_checkpointing,
            logger=loggers,
            deterministic=cfg.trainer.deterministic,
            benchmark=cfg.trainer.benchmark,
            fast_dev_run=cfg.trainer.fast_dev_run,
            callbacks=[task_callback],
        )

        # Run TTA on this task
        trainer.fit(tta_module, train_dataloaders=task_dataloader)

        # Store task results using module's method
        final_acc = tta_module.get_final_accuracy()
        all_task_results[task_name] = {
            "top1": final_acc["top1"],
            "top5": final_acc["top5"],
        }

        # Log task completion to W&B with unique task identifier
        if loggers:
            for logger in loggers:
                if isinstance(logger, WandbLogger):
                    logger.experiment.log(
                        {
                            f"task_results/{task_name}/top1": final_acc["top1"],
                            f"task_results/{task_name}/top5": final_acc["top5"],
                        }
                    )

        print(f"Completed task: {task_name}")

    # Print and log summary of all tasks
    print("\n" + "=" * 80)
    print("All tasks completed!")
    print("=" * 80)

    if all_task_results:
        # Calculate statistics
        top1_scores = [v["top1"] for v in all_task_results.values()]
        top5_scores = [v["top5"] for v in all_task_results.values()]
        mean_top1 = sum(top1_scores) / len(top1_scores)
        mean_top5 = sum(top5_scores) / len(top5_scores)

        # Print detailed results
        print("\n" + "=" * 80)
        print("FINAL RESULTS SUMMARY")
        print("=" * 80)
        print(f"{'Task':<40} {'Top-1 (%)':<12} {'Top-5 (%)':<12}")
        print("-" * 64)
        for task_name, acc in all_task_results.items():
            print(f"{task_name:<40} {acc['top1']:<12.2f} {acc['top5']:<12.2f}")
        print("-" * 64)
        print(f"{'MEAN':<40} {mean_top1:<12.2f} {mean_top5:<12.2f}")
        print("=" * 80)

        # Log summary to W&B
        if loggers:
            for logger in loggers:
                if isinstance(logger, WandbLogger):
                    # Log summary statistics
                    logger.experiment.log(
                        {
                            "summary/mean_top1": mean_top1,
                            "summary/mean_top5": mean_top5,
                            "summary/num_tasks": len(all_task_results),
                        }
                    )

                    # Create a summary table
                    import wandb

                    table = wandb.Table(columns=["Task", "Top-1 (%)", "Top-5 (%)"])
                    for task_name, acc in all_task_results.items():
                        table.add_data(task_name, acc["top1"], acc["top5"])
                    table.add_data("MEAN", mean_top1, mean_top5)
                    logger.experiment.log({"summary/results_table": table})

    # Finish logging
    if loggers:
        for logger in loggers:
            if hasattr(logger, "finalize"):
                logger.finalize("success")


if __name__ == "__main__":
    main()
