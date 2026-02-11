"""
3DOF Robotic Arm Controller
Maps fused multimodal embeddings to robotic arm position and gripper force
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple


class RoboticArmController3DOF(nn.Module):
    """
    Decodes 512-dim fused embeddings to 3DOF robotic arm position and gripper force
    
    Output Space:
    - X (horizontal): -0.5 to 0.5 meters (left-right)
    - Y (vertical):    0.0 to 1.0 meters (down-up)
    - Z (depth):       0.0 to 1.0 meters (near-far / along-arm)
    - Force:           0.0 to 100.0 % (gripper strength)
    """
    
    def __init__(
        self,
        embedding_dim: int = 512,
        hidden_dim: int = 256,
        device: str = "cpu"
    ):
        """
        Initialize 3DOF Robotic Arm Controller
        
        Args:
            embedding_dim: Input embedding dimension (512 for our system)
            hidden_dim: Hidden layer dimension for decoder
            device: Device to run on
        """
        super().__init__()
        self.device = device
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Decoder network: 512-dim embedding → 4-dim output (X, Y, Z, force)
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 4)  # [x, y, z, force]
        ).to(device)
        
        # Initialize weights
        self._initialize_weights()
        
        # Register workspace bounds
        self._register_workspace_bounds()
    
    def _initialize_weights(self):
        """Initialize decoder weights"""
        for module in self.decoder:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                nn.init.constant_(module.bias, 0.0)
    
    def _register_workspace_bounds(self):
        """Register robotic arm workspace bounds"""
        # Position bounds (meters)
        self.pos_min = torch.tensor([-0.5, 0.0, 0.0], device=self.device, dtype=torch.float32)
        self.pos_max = torch.tensor([0.5, 1.0, 1.0], device=self.device, dtype=torch.float32)
        
        # Force bounds (percentage)
        self.force_min = 0.0
        self.force_max = 100.0
    
    def decode(self, fused_embedding: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Decode 512-dim fused embedding to robotic commands
        
        Args:
            fused_embedding: (batch_size, 512) tensor
        
        Returns:
            Dictionary containing:
            - 'position': (batch_size, 3) - X, Y, Z coordinates in meters
            - 'force': (batch_size, 1) - Gripper force 0-100%
            - 'raw': (batch_size, 4) - Raw decoder output before normalization
        """
        batch_size = fused_embedding.size(0)
        
        # Get raw decoder output
        raw_output = self.decoder(fused_embedding)  # (batch, 4)
        
        # Split position and force
        position_raw = raw_output[:, :3]   # (batch, 3)
        force_raw = raw_output[:, 3:4]     # (batch, 1)
        
        # Normalize position to workspace bounds
        position = self._normalize_position(position_raw)  # (batch, 3)
        
        # Normalize force to [0, 100]%
        force = torch.sigmoid(force_raw) * 100.0  # (batch, 1)
        
        return {
            'position': position,
            'force': force,
            'raw': raw_output,
            'position_raw': position_raw,
            'force_raw': force_raw
        }
    
    def _normalize_position(self, position_raw: torch.Tensor) -> torch.Tensor:
        """
        Normalize raw position to workspace bounds using tanh mapping
        
        Args:
            position_raw: (batch, 3) unbounded tensor
        
        Returns:
            position: (batch, 3) bounded to workspace
        """
        # Map through tanh: unbounded → [-1, 1]
        position_norm = torch.tanh(position_raw)  # (batch, 3) in [-1, 1]
        
        # Scale from [-1, 1] to [pos_min, pos_max]
        # Formula: scaled = (normalized + 1) / 2 * (max - min) + min
        position = (position_norm + 1.0) / 2.0 * (self.pos_max - self.pos_min) + self.pos_min
        
        return position
    
    def forward(self, fused_embedding: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass - decode embedding to robotic commands
        
        Args:
            fused_embedding: (batch_size, 512) fused embedding
        
        Returns:
            Dictionary with robotic arm commands
        """
        return self.decode(fused_embedding)
    
    def compute_joint_angles(
        self,
        position: torch.Tensor,
        use_approximation: bool = True
    ) -> torch.Tensor:
        """
        Compute estimated joint angles from 3D position (approximate)
        
        For a simple 3DOF arm:
        - Joint 1 (base rotation): atan2(x, z)
        - Joint 2 (shoulder): related to y
        - Joint 3 (elbow): related to reach distance
        
        Args:
            position: (batch_size, 3) - X, Y, Z coordinates
            use_approximation: If True, use simple geometric approximation
        
        Returns:
            joint_angles: (batch_size, 3) - angles in radians
        """
        batch_size = position.size(0)
        
        if use_approximation:
            # Simple approximation for 3DOF robot
            # Joint 1: Base rotation (around Y axis)
            theta1 = torch.atan2(position[:, 0], position[:, 2])  # (batch,)
            
            # Joint 2: Vertical reach (pitch)
            reach_xy = torch.sqrt(position[:, 0]**2 + position[:, 2]**2)  # (batch,)
            theta2 = torch.atan2(position[:, 1], reach_xy)  # (batch,)
            
            # Joint 3: Wrist pitch (estimated from position magnitude)
            position_magnitude = torch.norm(position, dim=1)  # (batch,)
            theta3 = torch.atan2(position[:, 1], position_magnitude + 1e-6)  # (batch,)
            
            joint_angles = torch.stack([theta1, theta2, theta3], dim=1)  # (batch, 3)
        else:
            # Placeholder for more complex IK
            joint_angles = torch.zeros(batch_size, 3, device=self.device)
        
        return joint_angles
    
    def compute_reachability(self, position: torch.Tensor) -> torch.Tensor:
        """
        Compute reachability score (0-1) for given position
        1.0 = easily reachable, 0.0 = unreachable
        
        Args:
            position: (batch_size, 3) - X, Y, Z coordinates
        
        Returns:
            reachability: (batch_size,) - reachability scores 0-1
        """
        # Check if position is within workspace bounds
        in_bounds = (
            (position[:, 0] >= self.pos_min[0]) & (position[:, 0] <= self.pos_max[0]) &
            (position[:, 1] >= self.pos_min[1]) & (position[:, 1] <= self.pos_max[1]) &
            (position[:, 2] >= self.pos_min[2]) & (position[:, 2] <= self.pos_max[2])
        ).float()
        
        # Compute distance from center of workspace (sweet spot)
        workspace_center = (self.pos_min + self.pos_max) / 2
        workspace_range = self.pos_max - self.pos_min
        
        # Normalized distance from center
        distance_from_center = torch.abs(position - workspace_center) / (workspace_range / 2)
        distance_score = torch.exp(-torch.norm(distance_from_center, dim=1) / 2.0)
        
        # Reachability = in_bounds_indicator * distance_score
        reachability = in_bounds * distance_score
        
        return reachability


class RoboticArmTrajectory:
    """
    Utility for generating and analyzing robotic arm trajectories
    """
    
    @staticmethod
    def interpolate_trajectory(
        start_pos: np.ndarray,
        end_pos: np.ndarray,
        num_steps: int,
        method: str = 'linear'
    ) -> np.ndarray:
        """
        Generate trajectory between two positions
        
        Args:
            start_pos: (3,) starting position
            end_pos: (3,) ending position
            num_steps: Number of steps in trajectory
            method: 'linear', 'cubic', or 'sigmoid'
        
        Returns:
            trajectory: (num_steps, 3) positions along path
        """
        t = np.linspace(0, 1, num_steps)
        
        if method == 'linear':
            trajectory = np.outer(1 - t, start_pos) + np.outer(t, end_pos)
        
        elif method == 'cubic':
            # Cubic easing in/out
            t_eased = 3*t**2 - 2*t**3
            trajectory = np.outer(1 - t_eased, start_pos) + np.outer(t_eased, end_pos)
        
        elif method == 'sigmoid':
            # Sigmoid smoothing
            t_smooth = 1 / (1 + np.exp(-10 * (t - 0.5)))
            trajectory = np.outer(1 - t_smooth, start_pos) + np.outer(t_smooth, end_pos)
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return trajectory
    
    @staticmethod
    def compute_trajectory_smoothness(trajectory: np.ndarray) -> float:
        """
        Compute smoothness of trajectory (lower is smoother)
        Uses second derivative (acceleration)
        
        Args:
            trajectory: (num_steps, 3) trajectory
        
        Returns:
            smoothness: float, lower values indicate smoother trajectories
        """
        # Compute second derivative (acceleration)
        if len(trajectory) < 3:
            return 0.0
        
        diffs = np.diff(trajectory, axis=0)
        accel = np.diff(diffs, axis=0)
        smoothness = np.mean(np.linalg.norm(accel, axis=1))
        
        return smoothness
