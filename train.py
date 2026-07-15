#!/usr/bin/env python3
"""
Training script for the mCGCNN model.
"""

import random
import argparse
import logging
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from torch.optim.lr_scheduler import MultiStepLR

# importing custom modules
from mcgcnn.model import MagneticCrystalGraphConvNet
from mcgcnn.graph_builder import PrecomputedGraphDataset, joint_crystal_collate
from mcgcnn.train_utils import Normalizer, MCGCNNTrainer


def set_seed(seed=42):
    """Enforces total reproducibility across all random generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Force deterministic CuDNN operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(log_dir):
    """Configures persistent logging to both terminal and a file."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "training.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def parse_args():
    """Handles all configurations via command-line arguments."""
    parser = argparse.ArgumentParser(description="mCGCNN Training Script")
    
    parser.add_argument("--data_dir", type=str, required=True, 
                        help="Path to the precomputed dataset directory")
    parser.add_argument("--out_dir", type=str, default="./outputs", 
                        help="Directory to save logs, splits, and model checkpoints")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--weight_decay", type=float, default=5e-3)
    parser.add_argument("--lr_milestones", nargs='+', type=int, default=[80, 120], 
                        help="Epochs at which to drop the learning rate (e.g., --lr_milestones 80 120)")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Global random seed for reproducibility")
    
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Setup paths & logging
    out_dir = Path(args.out_dir).resolve()
    logger = setup_logging(out_dir)
    
    # 2. Lock execution environment for reproducibility
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info("==================================================")
    logger.info("--- mCGCNN Pipeline Initialization ---")
    logger.info(f"Running execution environment on: {device}")
    logger.info(f"Output directory locked to: {out_dir}")
    logger.info("==================================================")

    # 3. Load dataset and perform dynamic split
    data_dir = Path(args.data_dir).resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    logger.info(f"Loading master precomputed dataset from {data_dir}...")
    master_dataset = PrecomputedGraphDataset(str(data_dir))
    total_size = len(master_dataset)

    # 80/10/10 train/val/test split
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size

    logger.info(f"Splitting data: {train_size} Train | {val_size} Val | {test_size} Test")

    # Split using fixed random seed generator (redundant but safe)
    generator = torch.Generator().manual_seed(args.seed)
    train_set, val_set, test_set = random_split(
        master_dataset, 
        [train_size, val_size, test_size], 
        generator=generator
    )

    # Save indices to disk in the output directory
    split_path = out_dir / "dataset_splits.pt"
    split_indices = {
        "train": train_set.indices,
        "val": val_set.indices,
        "test": test_set.indices
    }
    torch.save(split_indices, split_path)
    logger.info(f"-> Successfully locked and saved data split indices to '{split_path.name}'")

    # 4. Instantiate all three loaders
    loader_kwargs = {
        "batch_size": args.batch_size,
        "collate_fn": joint_crystal_collate,
        "pin_memory": True
    }
    
    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **loader_kwargs)

    # 5. Initialize target normalizer
    logger.info("Extracting training target distribution for Regression Normalizer...")
    train_targets = []
    for batch in train_loader:
        train_targets.append(batch[5])

    train_targets = torch.cat(train_targets, dim=0)
    normalizer = Normalizer(train_targets)
    logger.info(f"-> Normalizer Active: Mean = {normalizer.mean:.4f}, Std = {normalizer.std:.4f}")

    # 6. Dynamically build the mcgcnn architecture
    logger.info("Analyzing graph dimensions...")
    sample_batch = next(iter(train_loader))

    orig_atom_fea_len = sample_batch[0][0].shape[-1]
    cg_nbr_fea_len = sample_batch[0][1].shape[-1]    
    orig_mag_fea_len = sample_batch[1][0].shape[-1]  
    mag_nbr_fea_len = sample_batch[1][1].shape[-1]   

    logger.info(f"  - Crystal Node Feature Dim: {orig_atom_fea_len}")
    logger.info(f"  - Magnetic Node Feature Dim: {orig_mag_fea_len}")
    logger.info(f"  - Crystal Edge Feature Dim:  {cg_nbr_fea_len}")
    logger.info(f"  - Magnetic Edge Feature Dim: {mag_nbr_fea_len}")

    model = MagneticCrystalGraphConvNet(
        orig_atom_fea_len=orig_atom_fea_len,
        orig_mag_fea_len=orig_mag_fea_len,
        cg_nbr_fea_len=cg_nbr_fea_len,
        mag_nbr_fea_len=mag_nbr_fea_len,
        atom_fea_len=64,
        n_conv=3,
        n_conv_mag=3,
        h_fea_len=256,
        n_h=2,
        dropout_rate=0.1,
        classification=False
    )

    # 7. Optimizer, Scheduler, and Loss function
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=args.lr_milestones, gamma=0.1)
    criterion = nn.MSELoss()

    # 8. Launch the trainer run
    trainer = MCGCNNTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        normalizer=normalizer,
        device=device
    )

    logger.info("==================================================")
    logger.info("              STARTING TRAINING LOOP              ")
    logger.info("==================================================")

    model_save_path = out_dir / "best_mcgcnn.pt"
    trainer.fit(train_loader, val_loader, epochs=args.epochs, save_path=str(model_save_path))

    logger.info("==================================================")
    logger.info("              FINAL MODEL EVALUATION              ")
    logger.info("==================================================")
    logger.info("Loading best stored model weights...")

    checkpoint = torch.load(model_save_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    final_test_loss = trainer.evaluate_epoch(test_loader, desc="Test")
    logger.info(f">>> Final Holdout Test Set Loss (MSE): {final_test_loss:.6f} <<<")
    logger.info("Pipeline routine complete. Ready for external prediction tasks.")


if __name__ == "__main__":
    main()
