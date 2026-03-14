"""
Drag Force Dataset for PyTorch Geometric
Parses the drag_dataset_84k.pt file and creates train/val splits based on unique meshes.
"""

import torch
from torch_geometric.data import Data, Dataset
from typing import List, Optional, Tuple
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import numpy as np

class DragForceDataset(Dataset):
    """
    Dataset for drag force prediction on different mesh geometries.

    Args:
        root: Root directory containing the dataset
        split: One of 'train', 'val', or 'all'
        validation_meshes: List of mesh numbers to use for validation
        transform: Optional transform to apply to each data sample
        pre_transform: Optional pre-transform to apply once during processing
        full_dataset: Optional pre-loaded dataset (skips disk load + normalization pass)
        feature_min: Optional per-feature min tensor (shape [num_features]) from training data
        feature_max: Optional per-feature max tensor (shape [num_features]) from training data
    """

    def __init__(
        self,
        root: str,
        split: str = 'train',
        validation_meshes: Optional[List[int]] = None,
        transform=None,
        pre_transform=None,
        feature_min: Optional[torch.Tensor] = None,
        feature_max: Optional[torch.Tensor] = None,
    ):
        self.split = split
        assert split in ['train', 'val', 'all'], "split must be 'train', 'val', or 'all'"

        if validation_meshes is None:
            validation_meshes = [0, 1, 2, 3, 4]
        self.validation_meshes = set(validation_meshes)

        self.data_path = Path(root) / 'drag_dataset_84k.pt'
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset not found at {self.data_path}")

        print(f"Loading dataset from {self.data_path}...")
        self.full_dataset = torch.load(self.data_path, weights_only=False)

        if pre_transform is not None:
            print("Applying pre_transform...")
            print(f"global_features shape: {self.full_dataset[0].global_features.shape}")
            print(f"x shape: {self.full_dataset[0].x.shape}")
            print("Data fields before pre_transform:", self.full_dataset[0].keys())
            for d in self.full_dataset:
                global_feats = d.global_features[0, [0, 1, 2, 6, 7]]  # [5]
                d.x_global = global_feats
            self.full_dataset = [pre_transform(d) for d in tqdm(self.full_dataset)]

            for d in self.full_dataset:
                num_nodes = d.x.shape[0]
                global_broadcasted = d.x_global.unsqueeze(0).expand(num_nodes, -1)  # [num_nodes, 5]
                d.x_raw = torch.cat([d.x, global_broadcasted], dim=1)  # [num_nodes, x_dim + 5]

            print(f"x_raw shape after concat: {self.full_dataset[0].x_raw.shape}")

        if not hasattr(self.full_dataset[0], 'mesh_number'):
            print("Adding mesh_number field to dataset...")
            self._add_mesh_numbers()

        # Split indices must exist before computing min/max
        self._create_split_indices()

        # Always compute normalization stats from training data only.
        # For val, feature_min/max are injected after instantiation (see utils.py).
        if feature_min is not None and feature_max is not None:
            self.feature_min = feature_min
            self.feature_max = feature_max
            self._apply_minmax_normalization()
        elif pre_transform is not None:
            # This is the train dataset — compute stats and normalize
            self.feature_min, self.feature_max = self._compute_minmax(self.train_indices)
            self._apply_minmax_normalization()

        super().__init__(root, transform, pre_transform)

    # ------------------------------------------------------------------
    # Min-max normalization
    # ------------------------------------------------------------------

    def _compute_minmax(self, indices: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute per-feature min/max across all nodes in the given indices."""
        print("Computing per-feature min/max from training data...")
        num_features = self.full_dataset[indices[0]].x_raw.shape[1]
        feat_min = torch.full((num_features,), float('inf'))
        feat_max = torch.full((num_features,), float('-inf'))

        for idx in tqdm(indices, desc="Scanning features"):
            x = self.full_dataset[idx].x_raw  # [num_nodes, num_features]
            feat_min = torch.minimum(feat_min, x.min(dim=0).values)
            feat_max = torch.maximum(feat_max, x.max(dim=0).values)

        constant_mask = (feat_max - feat_min) == 0
        if constant_mask.any():
            print(
                f"  Warning: {constant_mask.sum().item()} constant feature(s) detected "
                f"(indices: {constant_mask.nonzero(as_tuple=True)[0].tolist()}). "
                "These will be mapped to 0."
            )

        print(f"  feature_min: {feat_min}")
        print(f"  feature_max: {feat_max}")
        return feat_min, feat_max

    def _apply_minmax_normalization(self):
        """Normalize x_raw -> x_normalized using training min/max. Keeps x_raw intact."""
        print("Applying min-max normalization to all samples...")
        scale = self.feature_max - self.feature_min
        safe_scale = scale.clone()
        safe_scale[safe_scale == 0] = 1.0

        for d in tqdm(self.full_dataset, desc="Normalizing"):
            d.x_normalized = (d.x_raw - self.feature_min) / safe_scale
            d.x_normalized[:, scale == 0] = 0.0

        print(f"x_normalized shape: {self.full_dataset[0].x_normalized.shape}")

    
    def _add_mesh_numbers(self):
        """Identify unique meshes and add mesh_number field to each Data object."""
        unique_meshes = []
        mesh_assignments = []
        
        for i, data in enumerate(tqdm(self.full_dataset, desc="Identifying unique meshes")):
            pos = data.pos
            
            # Check if this mesh matches any existing unique mesh
            found_match = False
            for mesh_idx, unique_pos in enumerate(unique_meshes):
                # First check if shapes match
                if pos.shape == unique_pos.shape:
                    # Then check if positions are close
                    if torch.allclose(pos, unique_pos, rtol=1e-5, atol=1e-8):
                        mesh_assignments.append(mesh_idx)
                        found_match = True
                        break
            
            # If no match found, this is a new unique mesh
            if not found_match:
                unique_meshes.append(pos)
                mesh_assignments.append(len(unique_meshes) - 1)
        
        print(f"\nFound {len(unique_meshes)} unique meshes")
        
        # Add mesh_number to each data object
        for i, data in enumerate(self.full_dataset):
            data.mesh_number = mesh_assignments[i]
        
        # Print distribution
        print("\nMesh distribution:")
        from collections import Counter
        mesh_counts = Counter(mesh_assignments)
        for mesh_num in sorted(mesh_counts.keys()):
            num_nodes = unique_meshes[mesh_num].shape[0]
            print(f"  Mesh {mesh_num} ({num_nodes} nodes): {mesh_counts[mesh_num]} samples")
    
    def _create_split_indices(self):
        """Create train/val split indices based on mesh numbers."""
        self.train_indices = []
        self.val_indices = []
        
        for idx, data in enumerate(self.full_dataset):
            if data.mesh_number in self.validation_meshes:
                self.val_indices.append(idx)
            else:
                self.train_indices.append(idx)
        
        print(f"\nDataset split:")
        print(f"  Training samples: {len(self.train_indices)}")
        print(f"  Validation samples: {len(self.val_indices)}")
        print(f"  Validation meshes: {sorted(self.validation_meshes)}")

    def _compute_minmax(self, indices: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute per-feature min and max across all nodes in the given split indices.
        Statistics are always derived from training data to prevent leakage.

        Returns:
            feature_min: Tensor of shape [num_features]
            feature_max: Tensor of shape [num_features]
        """
        print("Computing per-feature min/max from training data...")
        num_features = self.full_dataset[indices[0]].x_raw.shape[1]
        feat_min = torch.full((num_features,), float('inf'))
        feat_max = torch.full((num_features,), float('-inf'))

        for idx in tqdm(indices, desc="Scanning features"):
            x = self.full_dataset[idx].x_raw  # [num_nodes, num_features]
            feat_min = torch.minimum(feat_min, x.min(dim=0).values)
            feat_max = torch.maximum(feat_max, x.max(dim=0).values)

        # Sanity-check for constant features (would cause division by zero)
        constant_mask = (feat_max - feat_min) == 0
        if constant_mask.any():
            print(
                f"  Warning: {constant_mask.sum().item()} constant feature(s) detected "
                f"(indices: {constant_mask.nonzero(as_tuple=True)[0].tolist()}). "
                "These will be mapped to 0."
            )

        print(f"  feature_min: {feat_min}")
        print(f"  feature_max: {feat_max}")
        return feat_min, feat_max

    def _apply_minmax_normalization(self):
        """
        Normalise x_raw in-place for every sample in the full dataset.
        Stores the result in data.x_normalized.

        Formula:  x_norm = (x - min) / (max - min)
        Constant features (max == min) are set to 0.
        """
        print("Applying min-max normalization to all samples...")
        scale = self.feature_max - self.feature_min          # [num_features]
        safe_scale = scale.clone()
        safe_scale[safe_scale == 0] = 1.0                    # avoid division by zero

        for d in tqdm(self.full_dataset, desc="Normalizing"):
            d.x_normalized = (d.x_raw - self.feature_min) / safe_scale
            # Zero out any feature that had no range (constant across training set)
            d.x_normalized[:, scale == 0] = 0.0

        print(f"x_normalized shape: {self.full_dataset[0].x_normalized.shape}")
    
    def len(self) -> int:
        """Return the number of samples in the current split."""
        if self.split == 'train':
            return len(self.train_indices)
        elif self.split == 'val':
            return len(self.val_indices)
        else:  # 'all'
            return len(self.full_dataset)
    
    def get(self, idx: int) -> Data:
        """Get a data sample by index."""
        if self.split == 'train':
            actual_idx = self.train_indices[idx]
        elif self.split == 'val':
            actual_idx = self.val_indices[idx]
        else:  # 'all'
            actual_idx = idx
        
        data = self.full_dataset[actual_idx]
        
        if self.transform is not None:
            data = self.transform(data)
        
        return data
    
    def get_mesh_statistics(self) -> dict:
        """Get statistics about meshes in the current split."""
        if self.split == 'train':
            indices = self.train_indices
        elif self.split == 'val':
            indices = self.val_indices
        else:
            indices = range(len(self.full_dataset))
        
        mesh_data = defaultdict(list)
        for idx in indices:
            data = self.full_dataset[idx]
            mesh_data[data.mesh_number].append({
                'num_nodes': data.pos.shape[0],
                'num_edges': data.edge_index.shape[1],
                'num_faces': data.face.shape[1],
                'y': data.y.item() if data.y.numel() == 1 else data.y.tolist(),
                'species': data.species_label,
            })
        
        # Aggregate statistics
        stats = {}
        for mesh_num, samples in mesh_data.items():
            stats[mesh_num] = {
                'num_samples': len(samples),
                'num_nodes': samples[0]['num_nodes'],
                'num_edges': samples[0]['num_edges'],
                'num_faces': samples[0]['num_faces'],
                'y_mean': np.mean([s['y'] for s in samples]),
                'y_std': np.std([s['y'] for s in samples]),
                'species': set([s['species'] for s in samples]),
            }
        
        return stats


def test_dataset():
    """Test the dataset loading and splitting."""
    # Test with different validation mesh selections
    print("=" * 80)
    print("Testing DragForceDataset")
    print("=" * 80)
    
    # Use first 5 meshes for validation
    dataset_train = DragForceDataset(
        root='../',
        split='train',
        validation_meshes=[0, 1, 2, 3, 4]
    )
    
    dataset_val = DragForceDataset(
        root='../',
        split='val',
        validation_meshes=[0, 1, 2, 3, 4]
    )

    print(len(dataset_train), "training meshes")
    print(len(dataset_val), "validation meshes")
    
    print("\n" + "=" * 80)
    print("Training set sample:")
    sample = dataset_train[0]
    print(f"  pos shape: {sample.pos.shape}")
    print(f"  edge_index shape: {sample.edge_index.shape}")
    print(f"  x shape: {sample.x.shape}")
    print(f"  y: {sample.y}")
    print(f"  mesh_number: {sample.mesh_number}")
    print(f"  global_features shape: {sample.global_features.shape}")
    print(f"  species: {sample.species_label}")
    
    print("\n" + "=" * 80)
    print("Validation set sample:")
    sample = dataset_val[0]
    print(f"  pos shape: {sample.pos.shape}")
    print(f"  edge_index shape: {sample.edge_index.shape}")
    print(f"  x shape: {sample.x.shape}")
    print(f"  y: {sample.y}")
    print(f"  mesh_number: {sample.mesh_number}")
    print(f"  global_features shape: {sample.global_features.shape}")
    print(f"  species: {sample.species_label}")
    
    print("\n" + "=" * 80)
    print("Training set mesh statistics:")
    train_stats = dataset_train.get_mesh_statistics()
    for mesh_num in sorted(train_stats.keys())[:5]:  # Show first 5
        stats = train_stats[mesh_num]
        print(f"  Mesh {mesh_num}:")
        print(f"    Samples: {stats['num_samples']}")
        print(f"    Nodes: {stats['num_nodes']}")
        print(f"    y: {stats['y_mean']:.4f} ± {stats['y_std']:.4f}")
    
    print("\n" + "=" * 80)
    print("Validation set mesh statistics:")
    val_stats = dataset_val.get_mesh_statistics()
    for mesh_num in sorted(val_stats.keys()):
        stats = val_stats[mesh_num]
        print(f"  Mesh {mesh_num}:")
        print(f"    Samples: {stats['num_samples']}")
        print(f"    Nodes: {stats['num_nodes']}")
        print(f"    y: {stats['y_mean']:.4f} ± {stats['y_std']:.4f}")


if __name__ == "__main__":
    test_dataset()
