"""
SAM3 多物体分割模块
"""
import os
import sys
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from PIL import Image
from loguru import logger


def select_instance_indices(masks, scores, expected_count, iou_threshold=0.5):
    """Select distinct, nonempty instances by score; never pad missing objects."""
    if not isinstance(expected_count, (int, np.integer)) or expected_count < 1:
        raise ValueError("Expected count must be a positive integer")
    masks = np.asarray(masks, dtype=bool)
    scores = np.asarray(scores)
    if masks.ndim != 3 or scores.shape != (len(masks),) or not np.isfinite(scores).all():
        raise ValueError("Expected masks (N,H,W) and finite scores (N,)")
    selected = []
    for index in np.argsort(-scores, kind='stable'):
        candidate = masks[index]
        if not candidate.any():
            continue
        if any(np.logical_and(candidate, masks[j]).sum() /
               np.logical_or(candidate, masks[j]).sum() > iou_threshold
               for j in selected):
            continue
        selected.append(int(index))
        if len(selected) == expected_count:
            return selected
    raise ValueError(f"Expected {expected_count} distinct instances, found {len(selected)}")


class SAM3MultiObjectSegmenter:
    """SAM3 多物体分割器"""

    def __init__(
        self,
        checkpoint_path: Path = None,
        confidence_threshold: float = 0.1,
        sam3_root: Optional[Path] = None,
    ):
        """
        初始化 SAM3 模型

        Args:
            checkpoint_path: SAM3 checkpoint path. Defaults to $SAM3_CHECKPOINT.
            confidence_threshold: 置信度阈值
            sam3_root: Path to a SAM 3 checkout. Only needed when sam3 is not
                installed in this environment. Defaults to $SAM3_ROOT.
        """
        # Upstream hard-coded the original author's paths under /mnt/workspace,
        # so this class could not run anywhere else. Both locations are now
        # configurable and an installed sam3 package is preferred.
        if checkpoint_path is None and os.environ.get("SAM3_CHECKPOINT"):
            checkpoint_path = Path(os.environ["SAM3_CHECKPOINT"]).expanduser()
        if checkpoint_path is None:
            raise ValueError(
                "No SAM3 checkpoint given. Pass --sam3_checkpoint, or set "
                "$SAM3_CHECKPOINT to the sam3.pt you downloaded from "
                "https://huggingface.co/facebook/sam3"
            )
        checkpoint_path = Path(checkpoint_path).expanduser()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"SAM3 checkpoint not found: {checkpoint_path}")

        self.checkpoint_path = checkpoint_path
        self.confidence_threshold = confidence_threshold

        if sam3_root is None and os.environ.get("SAM3_ROOT"):
            sam3_root = os.environ["SAM3_ROOT"]
        if sam3_root is not None:
            sam3_root = Path(sam3_root).expanduser().resolve()
            if sam3_root.is_dir() and str(sam3_root) not in sys.path:
                sys.path.insert(0, str(sam3_root))

        try:
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
        except ImportError as e:
            raise ImportError(
                "Could not import sam3. Install it into this environment or point "
                "--sam3_root / $SAM3_ROOT at a SAM 3 checkout. Preprocessing is only "
                "needed to generate masks; if you already have RGBA masks you can "
                "skip it entirely."
            ) from e
        
        # 加载模型
        logger.info(f"Loading SAM3 model from: {checkpoint_path}")
        model = build_sam3_image_model(
            checkpoint_path=str(checkpoint_path),
            load_from_HF=False
        )
        self.processor = Sam3Processor(model, confidence_threshold=confidence_threshold)
        logger.success("✓ SAM3 model loaded")
    
    def segment_object_multiview(
        self,
        images_dir: Path,
        object_name: str,
        text_prompt: str,
        output_dir: Path,
        expected_count: int = 1,
    ) -> Dict:
        """
        对多个视角的图像分割同一个物体
        
        Args:
            images_dir: 图像目录
            object_name: 物体名称
            text_prompt: SAM3 文本提示词
            output_dir: 输出目录（将创建 object_name/ 子目录）
        
        Returns:
            Dict with status and info
        """
        logger.info(f"\n[Segmenting: {object_name}]")
        logger.info(f"  Prompt: '{text_prompt}'")
        
        # 获取所有图像（使用自然数字排序，确保 2.png 排在 10.png 前面）
        def natural_sort_key(p):
            try:
                return (0, int(p.stem), p.stem)
            except ValueError:
                return (1, 0, p.stem)
        
        image_files = sorted(
            list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg")),
            key=natural_sort_key
        )
        if not image_files:
            return {'success': False, 'error': 'No images found'}
        
        logger.info(f"  Processing {len(image_files)} views...")
        
        # 创建物体的 mask 目录
        mask_dir = output_dir / object_name
        mask_dir.mkdir(parents=True, exist_ok=True)
        
        # 对每个视角进行分割
        success_count = 0
        failed_views = []
        view_reports = []
        
        for i, img_path in enumerate(image_files):
            try:
                # 读取图像
                image = Image.open(img_path).convert('RGB')
                
                # SAM3 分割（按照SAM4D的方式）
                with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                    inference_state = self.processor.set_image(image)
                    output = self.processor.set_text_prompt(
                        state=inference_state,
                        prompt=text_prompt
                    )
                
                masks = output["masks"]
                scores = output["scores"]
                
                masks_np = masks.detach().cpu().numpy() if torch.is_tensor(masks) else np.asarray(masks)
                scores_np = scores.float().detach().cpu().numpy() if torch.is_tensor(scores) else np.asarray(scores)
                masks_np = masks_np[:, 0]  # SAM 3 returns (N,1,H,W).
                selected = select_instance_indices(masks_np, scores_np, expected_count)
                mask_np = np.logical_or.reduce(masks_np[selected])
                selected_scores = scores_np[selected].tolist()
                
                # 确保 mask 和原图尺寸一致
                if mask_np.shape != (image.size[1], image.size[0]):
                    mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8))
                    mask_pil = mask_pil.resize(image.size, Image.NEAREST)
                    mask_np = np.array(mask_pil) / 255.0
                
                # 创建 RGBA 格式的 mask（与 example 一致）
                # RGB 保留原图颜色，Alpha 作为 mask
                image_np = np.array(image)
                mask_bool = mask_np > 0.5
                
                # 创建 RGBA 图像
                rgba_mask = np.zeros((image_np.shape[0], image_np.shape[1], 4), dtype=np.uint8)
                # mask 区域：保留原图 RGB，alpha=255
                rgba_mask[mask_bool, :3] = image_np[mask_bool]
                rgba_mask[mask_bool, 3] = 255
                # 非 mask 区域：RGB=0，alpha=0（透明）
                rgba_mask[~mask_bool, :3] = 0
                rgba_mask[~mask_bool, 3] = 0
                
                # 保存为 RGBA PNG（使用原图文件名，确保mask和image名称一致）
                mask_path = mask_dir / f"{img_path.stem}.png"
                Image.fromarray(rgba_mask, 'RGBA').save(mask_path)
                
                # 计算 mask 面积（使用 alpha 通道）
                area_ratio = np.sum(rgba_mask[:, :, 3] > 0) / (rgba_mask.shape[0] * rgba_mask.shape[1])
                
                view_reports.append({'image': img_path.name, 'candidate_count': len(scores_np),
                                     'selected_count': len(selected), 'scores': selected_scores,
                                     'area_ratio': float(area_ratio)})
                logger.info(f"  View {i}: ✓ (instances={len(selected)}, area={area_ratio*100:.1f}%)")
                success_count += 1
                
            except Exception as e:
                # A failed rerun must not leave this view's old mask usable.
                (mask_dir / f"{img_path.stem}.png").unlink(missing_ok=True)
                logger.error(f"  View {i}: Failed - {e}")
                import traceback
                traceback.print_exc()
                failed_views.append(i)
                view_reports.append({'image': img_path.name, 'error': str(e)})
        
        # 总结
        logger.info(f"  Result: {success_count}/{len(image_files)} views segmented")
        if failed_views:
            logger.warning(f"  Failed views: {failed_views}")
        
        return {
            'success': success_count == len(image_files),
            'expected_count': expected_count,
            'text_prompt': text_prompt,
            'confidence_threshold': self.confidence_threshold,
            'views': view_reports,
            'object_name': object_name,
            'total_views': len(image_files),
            'success_views': success_count,
            'failed_views': failed_views,
            'mask_dir': mask_dir,
        }
