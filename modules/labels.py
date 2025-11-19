import torch
import torch.nn as nn
from typing import Optional, Literal

class LabelAggregator:
    """
    Args:
        num_nodes: Total number of nodes in graph
        num_classes: Number of target classes
        window: Window size for moving average (default: 7)
        noise_factor: Noise scaling factor γ for regularization (default: 0.0)
        mode: Aggregation strategy ('moving_average', 'historical_average', 'persistent_forecast')
    """
    
    def __init__(
        self,
        num_nodes: int,
        num_classes: int,
        window: int = 7,
        noise_factor: float = 0.0,
        mode: Literal['moving_average', 'historical_average', 'persistent_forecast'] = 'moving_average'
    ):
        self.num_nodes = num_nodes
        self.num_classes = num_classes
        self.window = window
        self.noise_factor = noise_factor
        self.mode = mode
        
        # Tensor for sliding window average (used for MA and HA)
        # Shape: (num_nodes, num_classes)
        self.average = torch.zeros(num_nodes, num_classes, dtype=torch.float)
        
        # Flags to track which nodes have been initialized
        self.initialized = torch.zeros(num_nodes, dtype=torch.bool)
        
        # For HA mode: track number of observations per node
        self.nodes_count = torch.zeros(num_nodes, dtype=torch.int)
        
        # For PF mode: store last observed labels
        self.updated_nodes = torch.zeros(num_nodes, dtype=torch.bool)
        
    def update(self, node_ids: torch.Tensor, labels: torch.Tensor):
        """
        Update pseudo-labels for observed nodes.
        
        Args:
            node_ids: Tensor of node indices with new observations
            labels: Tensor of ground-truth labels (one-hot or probability distributions)
        """
        # Mark nodes as having been updated
        self.updated_nodes[node_ids] = True
        
        # Determine which nodes are already initialized
        mask = self.initialized[node_ids]
        
        # Update initialized nodes according to chosen mode
        if self.mode == 'historical_average':
            self.nodes_count[node_ids[mask]] += 1
            self.average[node_ids[mask]] = (
                self.average[node_ids[mask]] * (self.nodes_count[node_ids[mask]][:, None] - 1) + 
                labels[mask]
            ) / self.nodes_count[node_ids[mask]][:, None]
            
        elif self.mode == 'moving_average':
            self.average[node_ids[mask]] = (
                (self.window - 1) / self.window * self.average[node_ids[mask]] + 
                (1 / self.window) * labels[mask]
            )
            
        elif self.mode == 'persistent_forecast':
            self.average[node_ids[mask]] = labels[mask]
        
        # Initialize new nodes with first observation
        self.average[node_ids[~mask]] = labels[~mask]
        self.initialized[node_ids] = True
        
    def reset_state(self):
        """Reset all stored state"""
        self.average = torch.zeros_like(self.average)
        self.initialized = torch.zeros_like(self.initialized)
        self.updated_nodes = torch.zeros_like(self.updated_nodes)
        self.nodes_count = torch.zeros_like(self.nodes_count)
        
    def to(self, device: torch.device):
        """Move all tensors to specified device"""
        self.average = self.average.to(device)
        self.initialized = self.initialized.to(device)
        self.updated_nodes = self.updated_nodes.to(device)
        self.nodes_count = self.nodes_count.to(device)
        return self
        
    def get_node_label(self, node_ids: torch.Tensor) -> torch.Tensor:
        """
        Get pseudo-labels for specified nodes with optional noise injection.
        
        Args:
            node_ids: Tensor of node indices to retrieve labels for
            
        Returns:
            Pseudo-labels tensor of shape (len(node_ids), num_classes)
        """
        srcs = torch.where(self.updated_nodes)[0]
        
        srcs = srcs[torch.isin(srcs, node_ids)]
        
        labels = self.average[srcs]
        
        noise = torch.rand_like(labels) * self.noise_factor
        
        noise = noise - noise.mean(dim=-1, keepdim=True)
        
        labels = labels + noise
            
        return srcs, labels

