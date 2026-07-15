#!/usr/bin/env python3
"""
Prediction script for the mCGCNN model.
"""

import argparse
import logging
from pathlib import Path

import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset

from mcgcnn.model import MagneticCrystalGraphConvNet
from mcgcnn.graph_builder import PrecomputedGraphDataset, joint_crystal_collate
from mcgcnn.train_utils import Normalizer


def setup_logging(log_dir):
    """Configures persistent logging to both terminal and a file."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "prediction.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def parse_args():
    """Handles all configurations via command-line arguments."""
    parser = argparse.ArgumentParser(description="mCGCNN Production Prediction Script")
    
    parser.add_argument("--data_dir", type=str, required=True, 
                        help="Path to the precomputed dataset directory (e.g., sample_dataset/processed_graphs)")
    parser.add_argument("--model_path", type=str, default="./outputs/best_mcgcnn.pt", 
                        help="Path to the saved model checkpoint")
    parser.add_argument("--split_path", type=str, default="./outputs/dataset_splits.pt", 
                        help="Path to the saved dataset splits file")
    parser.add_argument("--out_dir", type=str, default="./outputs/predictions", 
                        help="Directory to save prediction CSVs and logs")
    parser.add_argument("--batch_size", type=int, default=128)
    
    # Ablation control
    parser.add_argument("--disable_magnetic", action="store_true",
                        help="Set this flag if predicting with a model trained WITHOUT the magnetic subgraph.")
    
    return parser.parse_args()


def load_checkpoint_and_normalizer(model_path, model, device, logger):
    """Safely loads model weights and recovers the Normalizer state."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # 1. Extract model state dictionary
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    logger.info(f"Successfully loaded model weights from {model_path.name}")

    # 2. Extract normalizer state dictionary
    normalizer = None
    if isinstance(checkpoint, dict) and "normalizer_state_dict" in checkpoint:
        normalizer = Normalizer(torch.tensor([0.0, 1.0]))
        normalizer.load_state_dict(checkpoint["normalizer_state_dict"])
        logger.info(f"-> Successfully loaded Normalizer: Mean = {normalizer.mean:.4f}, Std = {normalizer.std:.4f}")
    
    return model, normalizer


def predict(model, loader, normalizer, device):
    """Runs inference over a dataloader and returns a DataFrame of results."""
    model.eval()

    all_preds = []
    all_targets = []
    all_ids = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting", leave=False):
            # Unpack the exact 7 elements returned by joint_crystal_collate
            standard_batch, magnetic_batch, mag_to_struct_idx, cg_batch_idx, mag_batch_idx, targets, cif_ids = batch

            # Unpack standard graph features
            cg_atom_fea, cg_nbr_fea, cg_nbr_idx = standard_batch
            standard_data = (
                cg_atom_fea.to(device),
                cg_nbr_fea.to(device),
                cg_nbr_idx.to(device),
                cg_batch_idx.to(device)
            )

            # Unpack magnetic graph features
            mag_atom_fea, mag_nbr_fea, mag_nbr_idx = magnetic_batch
            magnetic_data = (
                mag_atom_fea.to(device),
                mag_nbr_fea.to(device),
                mag_nbr_idx.to(device),
                mag_batch_idx.to(device)
            )

            mag_to_struct_idx = mag_to_struct_idx.to(device)

            # Forward pass
            output = model(standard_data, magnetic_data, mag_to_struct_idx)

            # Denormalize predictions
            preds = normalizer.denorm(output.cpu())

            all_preds.extend(preds.view(-1).tolist())
            all_targets.extend(targets.view(-1).tolist())
            all_ids.extend(cif_ids)

    return pd.DataFrame({
        "cif_id": all_ids,
        "target": all_targets,
        "prediction": all_preds
    })


def main():
    args = parse_args()

    # 1. Setup paths & logging
    out_dir = Path(args.out_dir).resolve()
    logger = setup_logging(out_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info("==================================================")
    logger.info("--- mCGCNN Predictor Initialization ---")
    logger.info(f"Running execution environment on: {device}")
    logger.info(f"Ablation Status -> 'use_magnetic': {not args.disable_magnetic}")
    logger.info("==================================================")

    # 2. Load the master dataset
    data_dir = Path(args.data_dir).resolve()
    logger.info(f"Loading master dataset from {data_dir}...")
    master_dataset = PrecomputedGraphDataset(str(data_dir))

    # 3. Reconstruct exact splits
    split_path = Path(args.split_path).resolve()
    if split_path.exists():
        logger.info(f"-> Found existing split locks in '{split_path.name}'. Reconstructing subsets...")
        split_indices = torch.load(split_path, map_location="cpu", weights_only=False)
        
        train_set = Subset(master_dataset, split_indices["train"])
        val_set = Subset(master_dataset, split_indices["val"])
        test_set = Subset(master_dataset, split_indices["test"])
        
        logger.info(f"-> Subsets matched: {len(train_set)} Train | {len(val_set)} Val | {len(test_set)} Test")
    else:
        raise FileNotFoundError(
            f"Could not find '{split_path}'. To run a mathematically valid prediction, "
            f"please ensure dataset_splits.pt exists and the path is correct."
        )

    # 4. Create loaders
    loader_kwargs = {"batch_size": args.batch_size, "shuffle": False, "collate_fn": joint_crystal_collate, "pin_memory": True}
    train_loader = DataLoader(train_set, **loader_kwargs)
    val_loader = DataLoader(val_set, **loader_kwargs)
    test_loader = DataLoader(test_set, **loader_kwargs)

    # 5. Extract sizes from master dataset to initialize model
    standard_graph, magnetic_graph, _, _, _ = master_dataset[0]
    orig_atom_fea_len = standard_graph[0].shape[-1]
    cg_nbr_fea_len = standard_graph[1].shape[-1]
    orig_mag_fea_len = magnetic_graph[0].shape[-1]
    mag_nbr_fea_len = magnetic_graph[1].shape[-1]

    # Initialize the architecture
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
    ).to(device)

    # 6. Load model weights and Normalizer state safely
    model_path = Path(args.model_path).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")
        
    model, normalizer = load_checkpoint_and_normalizer(model_path, model, device, logger)

    if normalizer is None:
        raise ValueError("Normalizer state dict not found in checkpoint. This script requires a checkpoint saved with Normalizer weights.")

    # 7. Generate and save prediction CSVs
    logger.info("Generating predictions for Training Set...")
    train_results = predict(model, train_loader, normalizer, device)
    train_results.to_csv(out_dir / "train_predictions.csv", index=False)

    logger.info("Generating predictions for Validation Set...")
    val_results = predict(model, val_loader, normalizer, device)
    val_results.to_csv(out_dir / "val_predictions.csv", index=False)

    logger.info("Generating predictions for Test Set...")
    test_results = predict(model, test_loader, normalizer, device)
    test_results.to_csv(out_dir / "test_predictions.csv", index=False)

    logger.info(f"Predictions successfully written to: '{out_dir}'")
    logger.info("==================================================")
    logger.info("                 PIPELINE COMPLETE                ")
    logger.info("==================================================")


if __name__ == "__main__":
    main()
