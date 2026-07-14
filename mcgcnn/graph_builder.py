import os
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from pymatgen.core.structure import Structure
import warnings


warnings.filterwarnings("ignore", category=UserWarning, module="pymatgen")


class GaussianExpander:
    """Converts a distance scalar into a vector using Gaussian bins."""
    def __init__(self, dmin=0, dmax=8, step=0.2, var=None):
        self.filter = torch.arange(dmin, dmax + step, step)
        self.var = var if var else step

    def expand(self, distances):
        return torch.exp(-((distances.unsqueeze(-1) - self.filter) ** 2) / self.var**2)

class FourierAngleExpansion:
    """
    Expands an angle \theta into a Fourier basis: 
    [cos(\theta), cos(2\theta), ..., cos(K\theta)]
    """
    def __init__(self, K=8, is_degrees=True):
        self.K = K
        self.is_degrees = is_degrees
        # Create a tensor of frequencies [1, 2, ..., K]
        self.frequencies = torch.arange(1, K + 1, dtype=torch.float32)

    def expand(self, angles):
        """
        angles: Tensor of shape (N,) or (N, 1) containing angles.
        Returns: Tensor of shape (N, K)
        """
        if self.is_degrees:
            angles = torch.deg2rad(angles)
        angles = angles.unsqueeze(-1) 
        k_theta = angles * self.frequencies.to(angles.device)
        return torch.cos(k_theta)



class AtomEmbedder:
    """Handles mapping elements to their standard and magnetic feature vectors."""
    def __init__(self, json_path, mag_json_path=None):
        # Load standard structural embeddings
        with open(json_path) as f:
            data = json.load(f)
        self.embeddings = {
            int(k): torch.tensor(v, dtype=torch.float)
            for k, v in data.items()
        }
        self.dim = len(list(self.embeddings.values())[0])

        # Load and concatenate magnetic embeddings (if provided)
        self.mag_embeddings = None
        self.mag_dim = self.dim

        if mag_json_path and os.path.exists(mag_json_path):
            with open(mag_json_path) as f:
                mag_data = json.load(f)

            self.mag_embeddings = {}
            for k in self.embeddings.keys():
                orig_feat = self.embeddings[int(k)]

                if str(k) in mag_data:
                    mag_feat = torch.tensor(mag_data[str(k)], dtype=torch.float)
                else:
                    # Fallback to zero vector if an element is somehow missing
                    mag_feat = torch.zeros(7, dtype=torch.float)

                # Concatenate the standard + magnetic vectors (e.g., 92 + 7 = 99 dimensions)
                self.mag_embeddings[int(k)] = torch.cat([orig_feat, mag_feat])

            self.mag_dim = len(list(self.mag_embeddings.values())[0])

    def get_features(self, atomic_numbers):
        """Returns standard features for the bulk crystal graph."""
        return torch.stack([self.embeddings[int(n)] for n in atomic_numbers])

    def get_mag_features(self, atomic_numbers):
        """Returns concatenated features for the magnetic subgraph."""
        if self.mag_embeddings is None:
            return self.get_features(atomic_numbers)
        return torch.stack([self.mag_embeddings[int(n)] for n in atomic_numbers])


class JointCrystalDataset(Dataset):
    def __init__(self,
                 root_dir,
                 csv_path,
                 atom_init_path,
                 mag_atom_init_path,
                 target_column="ms_tesla",
                 max_cg_neighbors=12,
                 cg_radius=8.0,
                 max_mag_neighbors=8,
                 mag_radius=5.0,
                 ):

        self.root_dir = root_dir
        self.target_column = target_column

        # Decoupled Hyperparameters
        self.max_cg_neighbors = max_cg_neighbors
        self.cg_radius = cg_radius
        self.max_mag_neighbors = max_mag_neighbors
        self.mag_radius = mag_radius

        self.df = pd.read_csv(csv_path)

        # Initialize embedder
        self.embedder = AtomEmbedder(atom_init_path, mag_atom_init_path)

        # Initialize Specific Expanders
        self.cg_dist_expander = GaussianExpander(dmin=0, dmax=cg_radius, step=0.2)
        self.mag_dist_expander = GaussianExpander(dmin=0, dmax=mag_radius, step=0.2)
        self.angle_expander = FourierAngleExpansion(K=8) 

        self.anions = {
            1, 6, 14, 7, 15, 33, 51, 8, 16, 34, 52, 9, 17, 35, 53
        }

        # Edge Feature Lengths (Unchanged for CG, updated for Mag to include ligand dim)
        self.cg_nbr_fea_len = len(self.cg_dist_expander.filter)
        self.mag_nbr_fea_len = (
            (len(self.mag_dist_expander.filter) * 3) +
            8 +
            self.embedder.dim  
        )

    def __len__(self):
        return len(self.df)

    def is_magnetic_site(self, site):
        return (site.specie.is_transition_metal or
                site.specie.is_lanthanoid or
                site.specie.is_actinoid)

    def get_shortest_bridge_features(self, struct, idx_i, idx_j, bridge_elements):
        """Finds the common bridge ligand that minimizes the total M1-L + M2-L distance."""
        if idx_i == idx_j:
            return 0.0, 0.0, 0.0, 0 # Added 0 for dummy ligand Z

        # Look slightly beyond the magnetic radius to catch bridging ligands
        nbrs_i = struct.get_neighbors(struct[idx_i], r=self.mag_radius + 1.0)
        nbrs_j = struct.get_neighbors(struct[idx_j], r=self.mag_radius + 1.0)

        dict_i = {n.index: n.nn_distance for n in nbrs_i if n.specie.number in bridge_elements}
        dict_j = {n.index: n.nn_distance for n in nbrs_j if n.specie.number in bridge_elements}

        common_bridges = set(dict_i.keys()).intersection(set(dict_j.keys()))
        common_bridges.discard(idx_i)
        common_bridges.discard(idx_j)

        if not common_bridges:
            return 0.0, 0.0, 0.0, 0 # Direct M-M bond, no ligand

        # Minimize the sum of the distances
        best_ligand = min(common_bridges, key=lambda b_idx: dict_i[b_idx] + dict_j[b_idx])

        dist_M1_L = dict_i[best_ligand]
        dist_M2_L = dict_j[best_ligand]
        angle = struct.get_angle(idx_i, best_ligand, idx_j)
        ligand_z = struct[best_ligand].specie.number # Extract the atomic number

        if not np.isfinite(angle):
            angle = 0.0

        return dist_M1_L, dist_M2_L, angle, ligand_z

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        cif_id = str(row["material_id"])
        target = row[self.target_column]
        struct = Structure.from_file(os.path.join(self.root_dir, f"{cif_id}.cif"))

        # standard crystal graph
        atomic_numbers = [site.specie.number for site in struct]
        atom_fea = self.embedder.get_features(atomic_numbers)

        all_nbrs = struct.get_all_neighbors(self.cg_radius, include_index=True)
        cg_nbr_idx = torch.zeros((len(struct), self.max_cg_neighbors), dtype=torch.long)
        cg_nbr_dist = torch.full((len(struct), self.max_cg_neighbors), self.cg_radius + 1.0)

        for i, nbrs in enumerate(all_nbrs):
            nbrs = sorted(nbrs, key=lambda x: x[1])[:self.max_cg_neighbors]
            for j, (_, dist, nx_idx, _) in enumerate(nbrs):
                cg_nbr_idx[i, j] = nx_idx
                cg_nbr_dist[i, j] = dist

        cg_nbr_fea = self.cg_dist_expander.expand(cg_nbr_dist)
        standard_graph = (atom_fea, cg_nbr_fea, cg_nbr_idx)

        # magnetic subgraph 
        tm_elements, anion_elements, intermetallic_elements = set(), set(), set()

        for site in struct:
            z = site.specie.number
            if self.is_magnetic_site(site):
                tm_elements.add(z)
            elif z in self.anions:
                anion_elements.add(z)
            else:
                intermetallic_elements.add(z)

        # Guaranteed to have at least one TM based on dataset constraints
        tm_indices = [i for i, site in enumerate(struct) if site.specie.number in tm_elements]

        # Cross-Coupling Lookup Tensor
        mag_to_struct_idx = torch.LongTensor(tm_indices)

        bridge_elements = anion_elements if anion_elements else (intermetallic_elements if intermetallic_elements else tm_elements)
        tm_index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(tm_indices)}

        tm_atomic_numbers = [struct[i].specie.number for i in tm_indices]
        mag_atom_fea = self.embedder.get_mag_features(tm_atomic_numbers)

        n_tm = len(tm_indices)
        mag_nbr_idx = torch.zeros((n_tm, self.max_mag_neighbors), dtype=torch.long)
        mag_nbr_dist = torch.full((n_tm, self.max_mag_neighbors), self.mag_radius + 1.0)

        mag_ligand_dist1 = torch.full((n_tm, self.max_mag_neighbors), self.mag_radius + 1.0)
        mag_ligand_dist2 = torch.full((n_tm, self.max_mag_neighbors), self.mag_radius + 1.0)
        mag_angles = torch.zeros((n_tm, self.max_mag_neighbors))

        # tensor for the ligand embedding vector
        mag_ligand_fea = torch.zeros((n_tm, self.max_mag_neighbors, self.embedder.dim), dtype=torch.float)

        for i, tm_idx in enumerate(tm_indices):
            all_neighbors = struct.get_neighbors(struct[tm_idx], r=self.mag_radius)
            tm_neighbors = [n for n in all_neighbors if self.is_magnetic_site(n) and n.index in tm_index_map and n.index != tm_idx]
            tm_neighbors = sorted(tm_neighbors, key=lambda x: x.nn_distance)[:self.max_mag_neighbors]

            for j, nbr in enumerate(tm_neighbors):
                mag_nbr_idx[i, j] = tm_index_map[nbr.index]
                mag_nbr_dist[i, j] = nbr.nn_distance

                d1, d2, angle, ligand_z = self.get_shortest_bridge_features(
                    struct, tm_idx, nbr.index, bridge_elements
                )

                mag_ligand_dist1[i, j] = d1
                mag_ligand_dist2[i, j] = d2
                mag_angles[i, j] = angle

                # fetch and store ligand embedding 
                if ligand_z > 0: # ensures direct M-M bonds just stay as zeros
                    mag_ligand_fea[i, j] = self.embedder.get_features([ligand_z])[0]

        # Expand continuously
        mag_dist_fea = self.mag_dist_expander.expand(mag_nbr_dist)
        mag_l1_fea = self.mag_dist_expander.expand(mag_ligand_dist1)
        mag_l2_fea = self.mag_dist_expander.expand(mag_ligand_dist2)
        mag_angle_fea = self.angle_expander.expand(mag_angles)

        # Concatenate (now includes mag_ligand_fea)
        mag_nbr_fea = torch.cat(
            [mag_dist_fea, mag_l1_fea, mag_l2_fea, mag_angle_fea, mag_ligand_fea],
            dim=-1
        )
        mag_nbr_fea = torch.nan_to_num(mag_nbr_fea, nan=0.0, posinf=0.0, neginf=0.0)

        magnetic_graph = (mag_atom_fea, mag_nbr_fea, mag_nbr_idx)

        return standard_graph, magnetic_graph, mag_to_struct_idx, torch.tensor([float(target)]), cif_id


def save_joint_graphs_to_disk(dataset, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    print(f"Converting {len(dataset)} CIFs to Joint Graph Tensors...")
    for i in tqdm(range(len(dataset))):
        # Update unpacking to catch the new mag_to_struct_idx
        standard_graph, magnetic_graph, mag_to_struct_idx, target, cif_id = dataset[i]

        save_path = os.path.join(out_dir, f"{cif_id}.pt")
        
        # Add the mapping tensor to your saved dictionaries
        torch.save({
            "standard_graph": standard_graph,
            "magnetic_graph": magnetic_graph,
            "mag_to_struct_idx": mag_to_struct_idx,
            "target": target,
            "cif_id": cif_id
        }, save_path)


def joint_crystal_collate(batch):
    # Standard Graph trackers
    batch_atom_fea, batch_nbr_fea, batch_nbr_idx = [], [], []
    cg_batch_idx = []
    cg_base_idx = 0

    # Magnetic Graph trackers
    batch_mag_atom_fea, batch_mag_nbr_fea, batch_mag_nbr_idx = [], [], []
    mag_batch_idx = []
    mag_base_idx = 0

    # Cross-coupling & targets
    batch_mag_to_struct_idx = []
    batch_target, batch_ids = [], []

    # Use enumerate to keep track of the current graph's ID in the batch (0, 1, 2...)
    for graph_idx, (standard_graph, magnetic_graph, mag_to_struct, target, cif_id) in enumerate(batch):
        
        # Unpack graphs
        atom_fea, nbr_fea, nbr_idx = standard_graph
        m_atom_fea, m_nbr_fea, m_nbr_idx = magnetic_graph

        # crystal graph processing 
        n_atoms = atom_fea.shape[0]
        batch_atom_fea.append(atom_fea)
        batch_nbr_fea.append(nbr_fea)
        batch_nbr_idx.append(nbr_idx + cg_base_idx) # Shift edge indices
        
        # create a flat 1D tensor for efficient scatter pooling
        # e.g., if n_atoms is 4 and graph_idx is 0 -> [0, 0, 0, 0]
        cg_batch_idx.append(torch.full((n_atoms,), graph_idx, dtype=torch.long))

        # magnetic subgraph processing
        n_mag_atoms = m_atom_fea.shape[0]
        batch_mag_atom_fea.append(m_atom_fea)
        batch_mag_nbr_fea.append(m_nbr_fea)
        batch_mag_nbr_idx.append(m_nbr_idx + mag_base_idx) # Shift edge indices
        
        # create flat 1D tensor for magnetic scatter pooling
        mag_batch_idx.append(torch.full((n_mag_atoms,), graph_idx, dtype=torch.long))

        # Shift the Cross-Coupling Index 
        # The magnetic atoms in this graph need to point to the structural atoms in this batched graph
        batch_mag_to_struct_idx.append(mag_to_struct + cg_base_idx)

        # Targets and IDs 
        batch_target.append(target)
        batch_ids.append(cif_id)

        # Update base indices for the next iteration
        cg_base_idx += n_atoms
        mag_base_idx += n_mag_atoms

    # Group the tensors into tuples for clean unpacking in the training loop
    standard_batch = (
        torch.cat(batch_atom_fea, dim=0),
        torch.cat(batch_nbr_fea, dim=0),
        torch.cat(batch_nbr_idx, dim=0)
    )

    magnetic_batch = (
        torch.cat(batch_mag_atom_fea, dim=0),
        torch.cat(batch_mag_nbr_fea, dim=0),
        torch.cat(batch_mag_nbr_idx, dim=0)
    )

    return (
        standard_batch, 
        magnetic_batch, 
        torch.cat(batch_mag_to_struct_idx, dim=0), 
        torch.cat(cg_batch_idx, dim=0), 
        torch.cat(mag_batch_idx, dim=0), 
        torch.stack(batch_target, dim=0), 
        batch_ids
    )


class PrecomputedGraphDataset(Dataset):
    """A lightning-fast dataset that just loads pre-built tensors from disk."""
    def __init__(self, data_dir):
        self.data_dir = data_dir
        # Get a list of all .pt files in the directory
        self.pt_files = [f for f in os.listdir(data_dir) if f.endswith('.pt')]

    def __len__(self):
        return len(self.pt_files)

    def __getitem__(self, idx):
        # Load the dictionary we saved earlier
        file_path = os.path.join(self.data_dir, self.pt_files[idx])
        data = torch.load(file_path)

        # Return exactly what the collate_fn expects!
        return (
            data["standard_graph"],
            data["magnetic_graph"],
            data["mag_to_struct_idx"],
            data["target"],
            data["cif_id"]
        )



