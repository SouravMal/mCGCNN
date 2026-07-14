#!/usr/bin/env python3
"""
Preprocess crystal structures into graph objects for mCGCNN.
"""

import torch
import argparse
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import multiprocessing as mp
from multiprocessing import Pool
from mcgcnn.graph_builder import JointCrystalDataset

# global variables (one copy per worker process)
worker_dataset = None
worker_output_dir = None

def init_worker(root_dir, csv_path, atom_json, mag_json, target_prop, output_dir):
    """Initialize one dataset object per worker process."""
    global worker_dataset, worker_output_dir
    worker_dataset = JointCrystalDataset(root_dir, csv_path, atom_json, mag_json, target_prop)
    worker_output_dir = output_dir

def process_and_save(idx):
    """Process one crystal and save the graph to disk."""
    try:
        standard_graph, magnetic_graph, mag_to_struct_idx, target, cif_id = worker_dataset[idx]
        save_path = worker_output_dir / f"{cif_id}.pt"
        
        torch.save({
            "standard_graph": standard_graph,
            "magnetic_graph": magnetic_graph,
            "mag_to_struct_idx": mag_to_struct_idx,
            "target": target,
            "cif_id": cif_id,
        }, save_path)
        
        return True, cif_id
    except Exception as e:
        return False, f"Sample {idx}: {type(e).__name__}: {e}"

def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess CIF files into graph objects for mCGCNN.")
    parser.add_argument("root_dir", help="Dataset directory containing dataset.csv, atom_init.json, magnetic_atom_init.json, and CIF files.")
    parser.add_argument("--target", default="tot_mom_mub", help="Target property column in dataset.csv (default: tot_mom_mub).")
    parser.add_argument("--output", default="processed_graphs", help="Output directory (created inside the dataset directory).")
    parser.add_argument("--workers", type=int, default=mp.cpu_count(), help="Number of worker processes (default: all available CPU cores).")
    return parser.parse_args()

def main():
    args = parse_args()
    root_dir = Path(args.root_dir).resolve()
    if not root_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found:\n{root_dir}")
    csv_path = root_dir / "dataset.csv"
    atom_json = root_dir / "atom_init.json"
    mag_json = root_dir / "magnetic_atom_init.json"
    
    output_dir = root_dir / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset CSV not found:\n{csv_path}")
    if not atom_json.exists():
        raise FileNotFoundError(f"atom_init.json not found:\n{atom_json}")
    if not mag_json.exists():
        raise FileNotFoundError(f"magnetic_atom_init.json not found:\n{mag_json}")

    num_items = len(pd.read_csv(csv_path))

    print("=" * 70)
    print("mCGCNN Graph Preprocessing")
    print("=" * 70)
    print(f"Dataset directory : {root_dir}")
    print(f"Dataset CSV       : {csv_path.name}")
    print(f"Target property   : {args.target}")
    print(f"Output directory  : {output_dir}")
    print(f"Workers           : {args.workers}")
    print(f"Total structures  : {num_items}")
    print("=" * 70)

    init_args = (str(root_dir), str(csv_path), str(atom_json), str(mag_json), args.target, output_dir)
    
    with Pool(processes=args.workers, initializer=init_worker, initargs=init_args) as pool:
        results = list(tqdm(
            pool.imap_unordered(process_and_save, range(num_items)),
            total=num_items, desc="Processing", unit="graph"
        ))

    failures = [msg for success, msg in results if not success]
    success = len(results) - len(failures)

    print("\n" + "=" * 70)
    print("Processing Summary")
    print("=" * 70)
    print(f"Successfully processed : {success}")
    print(f"Failed                 : {len(failures)}")

    if failures:
        print("\nFirst 10 failures:")
        for err in failures[:10]:
            print(f"  - {err}")

    print("=" * 70)
    print("Finished.")

if __name__ == "__main__":
    main()
