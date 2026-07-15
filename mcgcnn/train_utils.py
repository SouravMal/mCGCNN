import time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm

class Normalizer:
    """Normalizes target variables for stable regression training."""
    def __init__(self, tensor):
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean, 'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


class MCGCNNTrainer:
    def __init__(self, model, optimizer, scheduler, criterion, normalizer, device):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.normalizer = normalizer
        self.device = device

    def unpack_batch(self, batch):
        """Helper to cleanly unpack the complex mCGCNN batch and move it to the GPU."""
        standard_batch, magnetic_batch, mag_to_struct_idx, cg_batch_idx, mag_batch_idx, targets, cif_ids = batch

        # Unpack Standard Graph
        cg_atom_fea = standard_batch[0].to(self.device)
        cg_nbr_fea = standard_batch[1].to(self.device)
        cg_nbr_idx = standard_batch[2].to(self.device)
        cg_batch_idx = cg_batch_idx.to(self.device)
        standard_data = (cg_atom_fea, cg_nbr_fea, cg_nbr_idx, cg_batch_idx)

        # Unpack Magnetic Subgraph
        mag_atom_fea = magnetic_batch[0].to(self.device)
        mag_nbr_fea = magnetic_batch[1].to(self.device)
        mag_nbr_idx = magnetic_batch[2].to(self.device)
        mag_batch_idx = mag_batch_idx.to(self.device)
        magnetic_data = (mag_atom_fea, mag_nbr_fea, mag_nbr_idx, mag_batch_idx)

        # Move indices and targets
        mag_to_struct_idx = mag_to_struct_idx.to(self.device)
        targets = targets.to(self.device)

        return standard_data, magnetic_data, mag_to_struct_idx, targets

    def train_epoch(self, loader, epoch, num_epochs):
        self.model.train()
        running_loss = 0.0

        loop = tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch in loop:
            # 1. Unpack
            std_data, mag_data, mag_to_struct, targets = self.unpack_batch(batch)
            
            # 2. Normalize targets
            targets_normed = self.normalizer.norm(targets)

            # 3. Forward Pass
            output = self.model(std_data, mag_data, mag_to_struct)
            loss = self.criterion(output, targets_normed)

            # 4. Backward Pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 5. Track Metrics
            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        return running_loss / len(loader)

    def evaluate_epoch(self, loader, desc="Val"):
        self.model.eval()
        running_loss = 0.0

        with torch.no_grad():
            for batch in loader:
                std_data, mag_data, mag_to_struct, targets = self.unpack_batch(batch)
                
                targets_normed = self.normalizer.norm(targets)
                
                output = self.model(std_data, mag_data, mag_to_struct)
                loss = self.criterion(output, targets_normed)
                
                running_loss += loss.item()

        return running_loss / len(loader)

    def fit(self, train_loader, val_loader, epochs, save_path="best_mcgcnn.pt"):
        best_val_loss = float('inf')
        history = {"epoch": [], "train_loss": [], "val_loss": []}
        
        start_time = time.time()

        for epoch in range(epochs):
            # Train and Validate
            train_loss = self.train_epoch(train_loader, epoch, epochs)
            val_loss = self.evaluate_epoch(val_loader, desc="Val")
            
            # Step the learning rate scheduler
            self.scheduler.step()

            # Save History
            history["epoch"].append(epoch)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            # Checkpoint the best model
            log_msg = f"Epoch {epoch+1}: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                
                # Save both the model weights AND the normalizer states!
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'normalizer_state_dict': self.normalizer.state_dict()
                }, save_path)
                log_msg += " --> [Best Model Saved]"
                
            print(log_msg)

        print(f"\nTraining completed in {(time.time() - start_time):.2f}s")
        
        # Save logs to CSV
        pd.DataFrame(history).to_csv("mcgcnn_training_log.csv", index=False)
        print("Training history saved to 'mcgcnn_training_log.csv'")
        
        return history
