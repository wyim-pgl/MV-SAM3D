"""
Pose Optimization: Direct optimization in aligned space

This module optimizes SAM3D pose (scale, rotation, translation) to align with
DA3 point clouds using the complete transformation chain from canonical to aligned space.

Key features:
- Mask-based object extraction from DA3 scene
- Chamfer Distance optimization with regularization
- Early stopping and adaptive learning rates
- Optional mask erosion to remove edge artifacts
"""
import numpy as np
import torch
import torch.nn.functional as F
from pytorch3d.ops import knn_points
from pytorch3d.transforms import quaternion_to_matrix
from typing import Dict, List, Optional, Tuple
from loguru import logger
import trimesh

# Check for opencv (required for mask erosion)
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    logger.warning("opencv-python not found. Mask erosion will be disabled.")
    logger.warning("Install with: pip install opencv-python")


# =============================================================================
# Utility Functions
# =============================================================================

def project_points_to_frame(points_world: np.ndarray, K: np.ndarray, w2c: np.ndarray, H: int, W: int):
    """
    Project 3D points (world space) to image plane.
    
    Args:
        points_world: (N, 3)
        K: (3, 3) intrinsics
        w2c: (4, 4) world-to-camera
        H, W: image dimensions
        
    Returns:
        uv: (N, 2) pixel coordinates
        valid: (N,) boolean mask
        depth: (N,) depth values
    """
    N = len(points_world)
    
    # To camera space
    points_homo = np.hstack([points_world, np.ones((N, 1))])
    points_cam = (w2c @ points_homo.T).T[:, :3]
    
    # Project to image
    points_img = (K @ points_cam.T).T
    depth = points_img[:, 2]
    uv = points_img[:, :2] / (depth[:, None] + 1e-8)
    
    # Check validity
    valid = (depth > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    
    return uv, valid, depth


def extract_object_pointcloud_from_scene(
    scene_glb_path: str,
    da3_npz_path: str,
    masks: List[np.ndarray],
    mask_threshold: int = 128,
    depth_tolerance: float = 0.1,
    max_points: int = 100000,
    mask_erosion_kernel: int = 3,
    size_tolerance_factor: Optional[float] = 0.5,
) -> np.ndarray:
    """
    Extract object points from DA3 scene.glb using masks (back-projection method).
    
    This function uses mask back-projection to identify which points in the DA3 scene
    belong to the target object. It projects each scene point to all views, checks if
    it falls within the mask, and verifies depth consistency.
    
    Args:
        scene_glb_path: Path to DA3 scene.glb
        da3_npz_path: Path to DA3 output npz
        masks: List of binary masks (one per view)
        mask_threshold: Threshold for object detection (0-255)
        depth_tolerance: Relative depth matching tolerance (e.g., 0.1 = 10%)
        max_points: Maximum points to return (downsampling if exceeded)
        mask_erosion_kernel: Erosion kernel size (pixels), removes mask edge errors
            - 0: No erosion
            - 3: Recommended (removes 1-2px edge errors)
            - 5: More conservative (removes 2-3px edge errors)
        size_tolerance_factor: Caps the depth tolerance at this fraction of the
            object's apparent size in each view (equivalent mask diameter x
            median mask depth / focal length). A purely relative tolerance is
            larger than small objects themselves and pulls in surface points
            behind them, which inflates the optimized scale. None disables
            the cap (previous behavior).
        
    Returns:
        object_points: (N, 3) in aligned space
    """
    from tqdm import tqdm
    
    logger.info("Extracting object pointcloud from scene.glb...")
    
    # Load scene
    scene = trimesh.load(scene_glb_path)
    
    # Find point cloud in scene
    point_cloud = None
    for name, geom in scene.geometry.items():
        if isinstance(geom, trimesh.points.PointCloud) or (
            hasattr(geom, 'vertices') and len(geom.vertices) > 100000
        ):
            point_cloud = geom
            break
    
    if point_cloud is None:
        raise ValueError("No point cloud found in scene.glb")
    
    points_aligned = point_cloud.vertices.copy()
    logger.info(f"  Total points in scene: {len(points_aligned)}")
    
    # Get alignment matrix
    alignment_matrix = None
    if hasattr(scene, 'metadata') and scene.metadata:
        alignment_matrix = scene.metadata.get('hf_alignment')
    
    if alignment_matrix is None:
        raise ValueError("No hf_alignment found in scene metadata")
    
    alignment_matrix = np.array(alignment_matrix)
    
    # Transform to world space
    alignment_inv = np.linalg.inv(alignment_matrix)
    points_world = trimesh.transform_points(points_aligned, alignment_inv)
    
    # Load DA3 data
    da3_data = np.load(da3_npz_path)
    depth = da3_data['depth']
    intrinsics = da3_data['intrinsics']
    extrinsics = da3_data['extrinsics']
    
    N, H, W = depth.shape
    
    # Check mask count vs frame count
    num_masks = len(masks) if masks else 0
    if num_masks < N:
        logger.warning(f"  Only {num_masks} masks provided for {N} DA3 frames. "
                      f"Only frames with masks will be used for point extraction.")
    
    # Back-project to identify object points
    is_object = np.zeros(len(points_world), dtype=bool)

    # Per-frame binary masks at DA3 resolution, absolute depth tolerance caps
    # from the object's apparent size.
    frame_masks = {}
    for frame_idx in range(min(N, len(masks) if masks is not None else 0)):
        mask = masks[frame_idx]
        if mask is None:
            continue
        if mask.shape != (H, W):
            from PIL import Image
            mask = np.array(Image.fromarray(mask).resize((W, H), Image.NEAREST))
        mask_binary = mask > mask_threshold
        if not mask_binary.any():
            continue
        if mask_erosion_kernel > 0 and HAS_CV2:
            kernel = np.ones((mask_erosion_kernel, mask_erosion_kernel), np.uint8)
            mask_check = cv2.erode(mask_binary.astype(np.uint8), kernel, iterations=1) > 0
        else:
            mask_check = mask_binary
        size_cap = None
        if size_tolerance_factor is not None:
            K = intrinsics[frame_idx]
            focal = 0.5 * (float(K[0, 0]) + float(K[1, 1]))
            mask_depth = depth[frame_idx][mask_binary]
            mask_depth = mask_depth[np.isfinite(mask_depth) & (mask_depth > 0)]
            if mask_depth.size and focal > 0:
                apparent_size = 2.0 * np.sqrt(mask_binary.sum() / np.pi) * float(np.median(mask_depth)) / focal
                size_cap = size_tolerance_factor * apparent_size
                logger.info(f"  Frame {frame_idx}: apparent object size {apparent_size:.4f}, "
                            f"depth tolerance capped at {size_cap:.4f}")
        frame_masks[frame_idx] = (mask_check, size_cap)
    
    batch_size = 50000
    num_batches = (len(points_world) + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(num_batches), desc="Extracting object points"):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(points_world))
        batch_points = points_world[start_idx:end_idx]
        
        for frame_idx in range(N):
            if frame_idx not in frame_masks:
                continue
            mask_check, size_cap = frame_masks[frame_idx]
            K = intrinsics[frame_idx]
            ext = extrinsics[frame_idx]
            depth_map = depth[frame_idx]

            # Convert to 4x4
            if ext.shape == (3, 4):
                w2c = np.eye(4)
                w2c[:3, :4] = ext
            else:
                w2c = ext

            # Project
            uv, valid, depth_proj = project_points_to_frame(batch_points, K, w2c, H, W)
            if not np.any(valid):
                continue
            u = np.clip(uv[valid, 0].astype(int), 0, W-1)
            v = np.clip(uv[valid, 1].astype(int), 0, H-1)

            # Depth verification: relative tolerance, capped by object size
            actual_depth = depth_map[v, u]
            tolerance = depth_tolerance * actual_depth
            if size_cap is not None:
                tolerance = np.minimum(tolerance, size_cap)
            depth_match = np.abs(actual_depth - depth_proj[valid]) < tolerance
            visible_indices = np.where(valid)[0][depth_match]
            in_mask = mask_check[v[depth_match], u[depth_match]]
            is_object[start_idx + visible_indices[in_mask]] = True

    # Filter object points (in aligned space)
    object_points = points_aligned[is_object]
    
    logger.info(f"  Object points extracted: {len(object_points)} ({len(object_points)/len(points_aligned)*100:.1f}%)")
    
    # Downsample if needed
    if len(object_points) > max_points:
        indices = np.random.choice(len(object_points), max_points, replace=False)
        object_points = object_points[indices]
        logger.info(f"  Downsampled to {max_points} points")
    
    return object_points


def widest_mask_diameter_px(masks: List[Optional[np.ndarray]], mask_threshold: int = 128) -> float:
    """Largest equivalent-circle diameter, in native mask pixels, over all views."""
    widest = 0.0
    for mask in masks or []:
        if mask is None:
            continue
        area = int((np.asarray(mask) > mask_threshold).sum())
        widest = max(widest, 2.0 * float(np.sqrt(area / np.pi)))
    return widest


# =============================================================================
# Pose Optimizer
# =============================================================================

class PoseOptimizer:
    """
    Pose optimizer that works in aligned space.
    
    This optimizer transforms the canonical mesh through the complete SAM3D
    transformation chain to aligned space, then optimizes the pose parameters
    (scale, rotation, translation) to minimize Chamfer Distance with target points.
    
    Transformation chain:
        canonical (Z-up) 
        → Y-up rotation
        → scale
        → rotation (quaternion)
        → translation
        → PyTorch3D to CV
        → alignment matrix
        → aligned space
    """
    
    def __init__(
        self,
        canonical_mesh_path: str,
        initial_pose: Dict[str, np.ndarray],
        target_points: np.ndarray,
        alignment_matrix: np.ndarray,
        device: str = 'cuda',
        optimize_scale: bool = False,
    ):
        """
        Args:
            canonical_mesh_path: Path to canonical mesh (result.glb)
            initial_pose: Initial pose with keys 'scale', 'rotation', 'translation'
            target_points: (N, 3) target point cloud in aligned space
            alignment_matrix: (4, 4) hf_alignment matrix from DA3
            device: 'cuda' or 'cpu'
            optimize_scale: If True, optimize scale; if False, keep scale fixed (default: False)
        """
        self.optimize_scale = optimize_scale
        self.device = device
        
        # Load canonical mesh
        logger.info("Loading canonical mesh...")
        sam3d_scene = trimesh.load(canonical_mesh_path)
        
        # Extract canonical vertices from GLB
        if isinstance(sam3d_scene, trimesh.Scene):
            source_vertices = None
            for name, geom in sam3d_scene.geometry.items():
                if hasattr(geom, 'vertices') and hasattr(geom, 'faces'):
                    source_vertices = geom.vertices
                    logger.info(f"  Found mesh: {name} ({len(source_vertices)} vertices)")
                    break
            if source_vertices is None:
                raise ValueError("No mesh found in canonical GLB")
        else:
            source_vertices = sam3d_scene.vertices
        
        # Sample points from mesh
        if len(source_vertices) > 50000:
            indices = np.random.choice(len(source_vertices), 50000, replace=False)
            source_points = source_vertices[indices]
        else:
            source_points = source_vertices
        
        self.source_canonical = torch.from_numpy(source_points).float().to(device)
        logger.info(f"  Source points (canonical): {len(self.source_canonical)}")
        
        # Target points
        self.target_points = torch.from_numpy(target_points).float().to(device)
        logger.info(f"  Target points (aligned): {len(self.target_points)}")
        
        # Alignment matrix
        self.alignment_matrix = torch.from_numpy(alignment_matrix).float().to(device)
        
        # Z-up to Y-up rotation
        self.R_zup_to_yup = torch.tensor([
            [1,  0,  0],
            [0,  0, -1],
            [0,  1,  0]
        ], dtype=torch.float32, device=device)
        
        # PyTorch3D to CV
        self.p3d_to_cv = torch.diag(torch.tensor([-1.0, -1.0, 1.0], device=device))
        
        # Initialize optimizable parameters
        scale = initial_pose['scale']
        rotation = initial_pose['rotation']
        translation = initial_pose['translation']
        
        if len(scale.shape) > 1:
            scale = scale.flatten()
        if len(rotation.shape) > 1:
            rotation = rotation.flatten()
        if len(translation.shape) > 1:
            translation = translation.flatten()
        
        # Use log scale for better optimization
        # If optimize_scale=False, this will be a fixed tensor (not Parameter)
        scale_value = torch.log(torch.tensor(scale[0], dtype=torch.float32, device=device))
        if optimize_scale:
            self.log_scale = torch.nn.Parameter(scale_value)
        else:
            # Keep as regular tensor (not Parameter), so it won't be optimized
            self.log_scale = scale_value
        
        # Quaternion (wxyz)
        self.quat = torch.nn.Parameter(
            torch.tensor(rotation, dtype=torch.float32, device=device)
        )
        
        # Translation
        self.translation = torch.nn.Parameter(
            torch.tensor(translation, dtype=torch.float32, device=device)
        )
        
        # Store initial values for regularization
        self.initial_log_scale = self.log_scale.data.clone()
        self.initial_quat = self.quat.data.clone()
        self.initial_translation = self.translation.data.clone()
        
        logger.info("[PoseOptimizer] Initialized:")
        logger.info(f"  Initial scale: {scale[0]:.4f} {'(optimizable)' if optimize_scale else '(fixed)'}")
        logger.info(f"  Initial rotation (wxyz): {rotation}")
        logger.info(f"  Initial translation: {translation}")
    
    def transform_to_aligned_space(self) -> torch.Tensor:
        """
        Apply complete transformation chain: canonical → aligned space.
        
        Returns:
            points: (N, 3) in aligned space
        """
        points = self.source_canonical.clone()
        
        # 1. Z-up to Y-up
        points = points @ self.R_zup_to_yup.T
        
        # 2. Scale
        scale = torch.exp(self.log_scale)
        points = points * scale
        
        # 3. Rotation
        quat_normalized = F.normalize(self.quat, p=2, dim=0)
        R = quaternion_to_matrix(quat_normalized.unsqueeze(0))[0]
        points = points @ R
        
        # 4. Translation
        points = points + self.translation
        
        # 5. PyTorch3D to CV
        points = points @ self.p3d_to_cv.T
        
        # 6. Apply alignment
        points_homo = torch.cat([points, torch.ones((len(points), 1), device=self.device)], dim=1)
        points = (self.alignment_matrix @ points_homo.T).T[:, :3]
        
        return points
    
    def compute_loss(self) -> Tuple[torch.Tensor, float]:
        """
        Compute unsquared, one-sided Chamfer loss without pairwise matrices.
        
        Returns:
            total_loss: Loss value for backprop
            cd: Chamfer distance value
        """
        source_aligned = self.transform_to_aligned_space()
        
        # Chunked cdist still retains every N x M matrix for backward. KNN
        # stores only nearest indices; differentiate the selected Euclidean
        # distances to preserve the original (unsquared) target -> source loss.
        with torch.no_grad():
            nearest = knn_points(
                self.target_points.unsqueeze(0), source_aligned.unsqueeze(0), K=1
            ).idx[0, :, 0]
        cd_loss = torch.linalg.vector_norm(
            self.target_points - source_aligned[nearest], dim=1
        ).mean()
        
        # Regularization to prevent large deviations
        quat_reg = 0.001 * (self.quat - self.initial_quat).pow(2).sum()
        trans_reg = 0.001 * (self.translation - self.initial_translation).pow(2).sum()
        
        # Only add scale regularization if we're optimizing scale
        if self.optimize_scale:
            scale_reg = 0.001 * (self.log_scale - self.initial_log_scale).pow(2)
            total_loss = cd_loss + scale_reg + quat_reg + trans_reg
        else:
            total_loss = cd_loss + quat_reg + trans_reg
        
        return total_loss, cd_loss.item()
    
    def optimize(
        self, 
        num_iterations: int = 300, 
        lr: float = 0.01, 
        early_stopping: bool = True, 
        patience: int = 50,
        scale_warmup_iterations: int = 100,
    ) -> Dict:
        """
        Run optimization.
        
        Args:
            num_iterations: Maximum number of iterations
            lr: Base learning rate
            early_stopping: Enable early stopping if CD stops improving
            patience: Number of iterations to wait before stopping
            scale_warmup_iterations: With optimize_scale, keep scale fixed for
                this many iterations so rotation and translation converge
                first. Otherwise a mesh initialized far from its target grows
                until its shell reaches the target points (issue #15: a 40 mm
                ball placed 77 mm away was scaled x3.4).
        
        Returns:
            history: Dictionary with optimization history
        """
        # Adaptive learning rate based on initial CD
        with torch.no_grad():
            _, initial_cd = self.compute_loss()
        logger.info(f"  Initial CD: {initial_cd:.6f}")
        
        # If initial CD is very small, use smaller learning rates
        if initial_cd < 0.01:
            lr_scale_factor = 0.5
            logger.warning(f"  Initial CD is very small ({initial_cd:.6f}), reducing learning rates by 50%")
        else:
            lr_scale_factor = 1.0
        
        # Build parameter groups based on what we're optimizing
        param_groups = [
            {'params': [self.quat], 'lr': lr * lr_scale_factor},
            {'params': [self.translation], 'lr': lr * 5 * lr_scale_factor},
        ]
        
        # Only add scale to optimizer if optimize_scale is True
        if self.optimize_scale:
            param_groups.insert(0, {'params': [self.log_scale], 'lr': lr * 10 * lr_scale_factor})
            logger.info("  Optimizing: scale + rotation + translation")
        else:
            logger.info("  Optimizing: rotation + translation (scale fixed)")
        
        optimizer = torch.optim.Adam(param_groups)

        # Keep scale fixed while rotation and translation converge, but never
        # for more than a third of the run so scale still gets optimized.
        warmup = min(scale_warmup_iterations, num_iterations // 3) if self.optimize_scale else 0
        # Halve every group's rate each 100 steps, counting the scale group's
        # steps from the end of its warm-up so it starts at its full rate.
        schedules = [lambda i: 0.5 ** (i // 100)] * len(param_groups)
        if self.optimize_scale:
            schedules[0] = lambda i: 0.5 ** (max(0, i - warmup) // 100)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedules)
        
        history = {
            'loss': [],
            'cd': [],
            'scale': [],
        }
        
        # Early stopping variables
        best_cd = float('inf')
        best_params = None
        patience_counter = 0
        
        from tqdm import tqdm
        
        if warmup:
            logger.info(f"  Scale fixed for the first {warmup} iterations (rotation + translation first)")

        with tqdm(total=num_iterations, desc="Optimizing pose") as pbar:
            for i in range(num_iterations):
                if warmup and i == warmup:
                    patience_counter = 0  # the scale phase starts its own patience window
                optimizer.zero_grad()
                loss, cd = self.compute_loss()
                loss.backward()
                if i < warmup:
                    self.log_scale.grad = None  # Adam skips parameters without gradients
                optimizer.step()
                scheduler.step()
                
                # Record
                scale = torch.exp(self.log_scale).item()
                history['loss'].append(loss.item())
                history['cd'].append(cd)
                history['scale'].append(scale)
                
                # Early stopping logic
                if early_stopping:
                    if cd < best_cd:
                        best_cd = cd
                        best_params = {
                            'log_scale': self.log_scale.data.clone(),
                            'quat': self.quat.data.clone(),
                            'translation': self.translation.data.clone(),
                        }
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    
                    if patience_counter >= patience and i >= warmup:
                        logger.warning(f"  Early stopping at iteration {i} (no improvement for {patience} steps)")
                        # Restore best parameters
                        if best_params is not None:
                            self.log_scale.data = best_params['log_scale']
                            self.quat.data = best_params['quat']
                            self.translation.data = best_params['translation']
                            logger.info(f"  Restored best parameters (CD={best_cd:.6f})")
                        pbar.update(num_iterations - i)
                        break
                
                if i % 10 == 0:
                    pbar.set_postfix({
                        'loss': f'{loss.item():.6f}',
                        'cd': f'{cd:.6f}',
                        'scale': f'{scale:.4f}'
                    })
                    pbar.update(10)
        
        logger.info(f"✓ Optimization finished. Final loss: {history['loss'][-1]:.6f}, Best CD: {best_cd:.6f}")
        
        return history
    
    def get_optimized_pose(self) -> Dict[str, np.ndarray]:
        """
        Get optimized pose parameters.
        
        Returns:
            Dictionary with 'scale', 'rotation', 'translation'
        """
        with torch.no_grad():
            scale = torch.exp(self.log_scale).item()
            quat = F.normalize(self.quat, p=2, dim=0).cpu().numpy()
            trans = self.translation.cpu().numpy()
        
        return {
            'scale': np.array([scale, scale, scale]),
            'rotation': quat,
            'translation': trans,
        }
