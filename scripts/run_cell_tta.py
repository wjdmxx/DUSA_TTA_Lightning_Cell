"""Main script for running Cell TTA."""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import Callback
import torch
from torch.utils.data import DataLoader

from src.models import create_combined_model
from src.data import create_cell_tasks, create_tta_transform, cell_collate_fn
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
        pl_module.set_task_info(self.task_name, self.task_idx)

        if not self.continual and self.task_idx > 0:
            pl_module.reset_model()
            trainer.strategy.setup_optimizers(trainer)


def setup_logging(cfg: DictConfig):
    """Setup loggers (W&B, TensorBoard)."""
    loggers = []

    if cfg.logging.use_wandb:
        wandb_logger = WandbLogger(
            project=cfg.logging.wandb_project,
            entity=cfg.logging.wandb_entity,
            name=cfg.experiment_name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        loggers.append(wandb_logger)

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

    disc_config = OmegaConf.to_container(cfg.model.discriminative, resolve=True)
    gen_config = (
        OmegaConf.to_container(cfg.model.generative, resolve=True)
        if cfg.model.generative
        else None
    )
    pixel_adapter_config = (
        OmegaConf.to_container(cfg.model.pixel_adapter, resolve=True)
        if cfg.model.get("pixel_adapter")
        else None
    )

    model = create_combined_model(
        discriminative_config=disc_config,
        generative_config=gen_config,
        pixel_adapter_config=pixel_adapter_config,
    )

    print(f"Discriminative model: {disc_config['model_name']}")
    if gen_config:
        print(f"Generative model: {gen_config['sit_model_name']}")
        print(f"Generative image size: {gen_config.get('image_size', 256)}")
    if pixel_adapter_config:
        print(f"Pixel Adapter: {pixel_adapter_config.get('type', 'standard')}")
    print("=" * 80)

    return model


def create_dataloaders(cfg: DictConfig):
    """Create dataloaders for cell dataset."""
    print("=" * 80)
    print("Creating dataloaders...")

    # Create transform (outputs [0, 1] tensor)
    transform = create_tta_transform(
        input_size=cfg.data.get("input_size", 224),
        use_center_crop=cfg.data.get("use_center_crop", False),
    )

    # Get class_to_idx from config if provided
    class_to_idx = cfg.data.get("class_to_idx", None)
    if class_to_idx is not None:
        class_to_idx = OmegaConf.to_container(class_to_idx, resolve=True)

    # Create task datasets
    tasks = create_cell_tasks(
        root=cfg.data.root,
        transform=transform,
        class_to_idx=class_to_idx,
    )

    print(f"Created {len(tasks)} tasks")

    # Create dataloaders
    dataloaders = []
    for task_name, dataset in tasks:
        dataloader = DataLoader(
            dataset,
            batch_size=cfg.data.batch_size,
            shuffle=cfg.data.get("shuffle", True),
            num_workers=cfg.data.num_workers,
            pin_memory=cfg.data.pin_memory,
            persistent_workers=cfg.data.get("persistent_workers", True),
            collate_fn=cell_collate_fn,
        )
        dataloaders.append((task_name, dataloader))

    print("=" * 80)
    return dataloaders


@hydra.main(version_base=None, config_path="../configs", config_name="config_cell")
def main(cfg: DictConfig):
    """Main TTA execution for Cell dataset."""
    print("\n" + "=" * 80)
    print("Cell Test-Time Adaptation")
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
        scheduler_config=OmegaConf.to_container(
            cfg.optimizer.get("scheduler"), resolve=True
        ),
        continual=cfg.tta.continual,
        update_auxiliary=cfg.tta.update_auxiliary,
        update_task_norm_only=cfg.tta.update_task_norm_only,
        forward_mode=cfg.tta.forward_mode,
        log_aux_metrics=cfg.logging.log_aux_metrics,
        num_classes=cfg.model.discriminative.num_classes,
        sample_log_dir=getattr(cfg.logging, "sample_log_dir", None),
        enable_sample_logging=getattr(cfg.logging, "enable_sample_logging", False),
    )

    # Save initial model state before any TTA
    tta_module.save_initial_state()

    # Setup logging
    loggers = setup_logging(cfg)

    # Create dataloaders
    task_dataloaders = create_dataloaders(cfg)
    total_tasks = len(task_dataloaders)

    # Run TTA on each task
    print("\n" + "=" * 80)
    print("Starting TTA...")
    print("=" * 80)

    all_task_results = {}

    for task_idx, (task_name, task_dataloader) in enumerate(task_dataloaders):
        print(f"\n{'='*80}")
        print(f"Task {task_idx + 1}/{total_tasks}: {task_name}")
        print(f"{'='*80}")

        task_callback = TaskSwitchCallback(
            task_name=task_name,
            task_idx=task_idx,
            total_tasks=total_tasks,
            continual=cfg.tta.continual,
        )

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
            enable_model_summary=(task_idx == 0),
            enable_checkpointing=cfg.trainer.enable_checkpointing,
            logger=loggers,
            deterministic=cfg.trainer.deterministic,
            benchmark=cfg.trainer.benchmark,
            fast_dev_run=cfg.trainer.fast_dev_run,
            callbacks=[task_callback],
        )

        trainer.fit(tta_module, train_dataloaders=task_dataloader)

        final_acc = tta_module.get_final_accuracy()
        all_task_results[task_name] = {
            "top1": final_acc["top1"],
            "top5": final_acc.get("top5", final_acc["top1"]),  # For few classes, top5 may equal top1
        }

        if loggers:
            for logger in loggers:
                if isinstance(logger, WandbLogger):
                    logger.experiment.log(
                        {
                            f"task_results/{task_name}/top1": final_acc["top1"],
                            f"task_results/{task_name}/top5": final_acc.get("top5", final_acc["top1"]),
                        }
                    )

        print(f"Completed task: {task_name}")

    # Print summary
    print("\n" + "=" * 80)
    print("All tasks completed!")
    print("=" * 80)

    if all_task_results:
        top1_scores = [v["top1"] for v in all_task_results.values()]
        mean_top1 = sum(top1_scores) / len(top1_scores)

        print("\n" + "=" * 80)
        print("FINAL RESULTS SUMMARY")
        print("=" * 80)
        print(f"{'Task':<40} {'Top-1 (%)':<12}")
        print("-" * 52)
        for task_name, acc in all_task_results.items():
            print(f"{task_name:<40} {acc['top1']:<12.2f}")
        print("-" * 52)
        print(f"{'MEAN':<40} {mean_top1:<12.2f}")
        print("=" * 80)

        if loggers:
            for logger in loggers:
                if isinstance(logger, WandbLogger):
                    logger.experiment.log(
                        {
                            "summary/mean_top1": mean_top1,
                            "summary/num_tasks": len(all_task_results),
                        }
                    )

    if loggers:
        for logger in loggers:
            if hasattr(logger, "finalize"):
                logger.finalize("success")


if __name__ == "__main__":
    main()
