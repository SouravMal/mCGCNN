#!/usr/bin/env python3
"""
Inference script for new, unknown data using a Pretrained mCGCNN model.
"""

import argparse
import logging
from pathlib import Path

import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader

from mcgcnn.model import MagneticCrystalGraphConvNet
from mcgcnn.graph_builder import PrecomputedGraphDataset, joint_crystal_collate
from mcgcnn.train_utils import Normalizer


def setup_logging(log_dir):
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "inference.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="mCGCNN Pretrained Model Inference")
    parser.add_argument("--data_dir", type=str, required=True, 
                        help="Path to the user's precomputed dataset directory")
    parser.add_argument("--model_path", type=str, default="./pretrained_models/mcgcnn_pretrained_tot_mom_v1.pt", 
                        help="Path to the pretrained model checkpoint")
    parser.add_argument("--out_dir", type=str, default="./inference_results", 
                        help="Directory to save the final predictions CSV")
    parser.add_argument("--batch_size", type=int, default=128)
    return parser.parse_args()


def load_checkpoint_and_normalizer(model_path, model, device, logger):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    logger.info(f"Successfully loaded pretrained model weights from {model_path.name}")

    normalizer = None
    if isinstance(checkpoint, dict) and "normalizer_state_dict" in checkpoint:
        normalizer = Normalizer(torch.tensor([0.0, 1.0]))
        normalizer.load_state_dict(checkpoint["normalizer_state_dict"])
        logger.info(f"-> Normalizer recovered: Mean = {normalizer.mean:.4f}, Std = {normalizer.std:.4f}")
    
    return model, normalizer


def main():
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    logger = setup_logging(out_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info("==================================================")
    logger.info("--- mCGCNN Pretrained Inference Initialization ---")
    logger.info("==================================================")

    data_dir = Path(args.data_dir).resolve()
    logger.info(f"Loading user dataset from {data_dir}...")
    dataset = PrecomputedGraphDataset(str(data_dir))
    
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, 
                        collate_fn=joint_crystal_collate, pin_memory=True)

    # Extract dimensions
    standard_graph, magnetic_graph, _, _, _ = dataset[0]
    orig_atom_fea_len = standard_graph[0].shape[-1]
    cg_nbr_fea_len = standard_graph[1].shape[-1]
    orig_mag_fea_len = magnetic_graph[0].shape[-1]
    mag_nbr_fea_len = magnetic_graph[1].shape[-1]

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

    model_path = Path(args.model_path).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Pretrained model not found at: {model_path}")
        
    model, normalizer = load_checkpoint_and_normalizer(model_path, model, device, logger)
    if normalizer is None:
        raise ValueError("Normalizer state dict missing. The pretrained model must contain normalizer weights.")

    model.eval()
    all_preds, all_ids = [], []

    logger.info("Running inference on unknown materials...")
    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting", leave=False):
            standard_batch, magnetic_batch, mag_to_struct_idx, cg_batch_idx, mag_batch_idx, _, cif_ids = batch

            standard_data = (standard_batch[0].to(device), standard_batch[1].to(device), standard_batch[2].to(device), cg_batch_idx.to(device))
            magnetic_data = (magnetic_batch[0].to(device), magnetic_batch[1].to(device), magnetic_batch[2].to(device), mag_batch_idx.to(device))
            
            output = model(standard_data, magnetic_data, mag_to_struct_idx.to(device))
            preds = normalizer.denorm(output.cpu())

            all_preds.extend(preds.view(-1).tolist())
            all_ids.extend(cif_ids)

    df_results = pd.DataFrame({"cif_id": all_ids, "predicted_tot_mom_mub": all_preds})
    out_csv = out_dir / "pretrained_predictions.csv"
    df_results.to_csv(out_csv, index=False)
    
    logger.info(f"Inference complete! Results saved to {out_csv}")

if __name__ == "__main__":
    main()
