"""
SAM3 多物体分割模块
"""
import os
import sys
from contextlib import nullcontext
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from PIL import Image
from loguru import logger


def select_instance_indices(masks, scores, expected_count, iou_threshold=0.5, report=None):
    """Fixed-count selection; optional diagnostics survive insufficient detections.

    Select-all is intentionally unsupported: None, zero and bool are not counts.
    """
    if isinstance(expected_count, (bool, np.bool_)) or not isinstance(expected_count, (int, np.integer)) or expected_count < 1:
        raise ValueError("Expected count must be a positive integer")
    masks = np.asarray(masks, dtype=bool)
    scores = np.asarray(scores)
    if masks.ndim != 3 or scores.shape != (len(masks),) or not np.isfinite(scores).all():
        raise ValueError("Expected masks (N,H,W) and finite scores (N,)")
    selected, dropped = [], []
    for index in np.argsort(-scores, kind='stable'):
        index = int(index)
        candidate = masks[index]
        if not candidate.any():
            reason = 'empty'
        elif any(np.logical_and(candidate, masks[j]).sum() /
                 np.logical_or(candidate, masks[j]).sum() > iou_threshold
                 for j in selected):
            reason = 'duplicate'
        elif len(selected) == expected_count:
            reason = 'count_limit'
        else:
            selected.append(index)
            continue
        dropped.append({'index': index, 'score': float(scores[index]), 'reason': reason})
    if report is not None:
        report.update(candidate_count=len(scores), selected_count=len(selected),
                      selected_indices=selected, scores=scores[selected].tolist(),
                      dropped_candidates=dropped)
    if len(selected) != expected_count:
        raise ValueError(f"Expected {expected_count} distinct instances, found {len(selected)}")
    return selected


def select_instance_candidates(masks, scores, max_candidates, iou_threshold=0.5):
    """Keep distinct, nonempty SAM3 proposals without requiring every view to
    contain the same number. Cross-view tracking resolves identities later.
    """
    if (isinstance(max_candidates, bool)
            or not isinstance(max_candidates, (int, np.integer))
            or max_candidates < 1):
        raise ValueError('max_candidates must be a positive integer')
    masks = np.asarray(masks, dtype=bool)
    scores = np.asarray(scores)
    if masks.ndim != 3 or scores.shape != (len(masks),) or not np.isfinite(scores).all():
        raise ValueError('Expected masks (N,H,W) and finite scores (N,)')
    selected, dropped = [], []
    for index in np.argsort(-scores, kind='stable'):
        index = int(index)
        candidate = masks[index]
        if not candidate.any():
            reason = 'empty'
        elif any(np.logical_and(candidate, masks[j]).sum() /
                 np.logical_or(candidate, masks[j]).sum() > iou_threshold
                 for j in selected):
            reason = 'duplicate'
        elif len(selected) == max_candidates:
            reason = 'count_limit'
        else:
            selected.append(index)
            continue
        dropped.append({'index': index, 'score': float(scores[index]), 'reason': reason})
    return selected, dropped


def validate_object_name(object_name):
    """Object outputs must be a simple, nonreserved directory name."""
    if (not isinstance(object_name, str) or not object_name.strip()
            or object_name != object_name.strip() or object_name in {'.', '..'}
            or any(c in object_name for c in '/\\\\:\x00')
            or object_name.casefold() in {'images', 'segmentation_report.json'}):
        raise ValueError(f"Unsafe or reserved object name: {object_name!r}")


def reject_source_aliases(destinations, sources):
    """Preflight the complete batch before writing or removing any output."""
    for destination in destinations:
        for source in sources:
            if (destination.resolve() == source.resolve()
                    or (destination.exists() and source.exists() and destination.samefile(source))):
                raise ValueError(f"Output aliases source image: {destination} -> {source}")


def preflight_mask_outputs(image_files, output_dir, object_name):
    validate_object_name(object_name)
    mask_dir = output_dir / object_name
    reject_source_aliases([mask_dir], {p.parent for p in image_files})
    destinations = [mask_dir / f'{p.stem}.png' for p in image_files]
    if len(set(destinations)) != len(destinations):
        raise ValueError("Source image stems must be unique for mask export")
    reject_source_aliases(destinations, image_files)
    return mask_dir


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

    def segment_instances_multiview(
        self,
        images_dir: Path,
        object_name: str,
        text_prompt: str,
        output_dir: Path,
        expected_count: int,
        tracking_views: Dict[str, Dict],
        object_names: Optional[List[str]] = None,
    ) -> Dict:
        """Track separate same-prompt detections in the shared DA3 scene frame.

        SAM3 is evaluated once per input view. Candidate masks are associated
        across views by their DA3 pointmap positions, then written separately
        under deterministic scene-position IDs. Unobserved views get explicit
        empty RGBA masks so downstream multi-object inference can keep a
        consistent image set.
        """
        from preprocessing.instance_tracking import track_instances

        def natural_sort_key(path):
            try:
                return (0, int(path.stem), path.stem)
            except ValueError:
                return (1, 0, path.stem)

        image_files = sorted(
            list(images_dir.glob('*.png')) + list(images_dir.glob('*.jpg'))
            + list(images_dir.glob('*.jpeg')), key=natural_sort_key)
        if len(image_files) < 2:
            raise ValueError('Automatic instance tracking requires at least two views')
        if set(path.stem for path in image_files) != set(tracking_views):
            raise ValueError('DA3 tracking views must match input images by filename stem')
        if object_names is None:
            object_names = [f'{object_name}_{i:02d}' for i in range(1, expected_count + 1)]
        if len(object_names) != expected_count or len(set(object_names)) != expected_count:
            raise ValueError('object_names must provide one unique output name per tracked instance')

        output_dirs = [preflight_mask_outputs(image_files, output_dir, name)
                       for name in object_names]
        candidates_by_view, masks_by_view, view_reports = {}, {}, {}
        use_cuda = torch.cuda.is_available()
        inference_context = torch.autocast('cuda', dtype=torch.bfloat16) if use_cuda else nullcontext()

        for view_index, image_path in enumerate(image_files):
            image = Image.open(image_path).convert('RGB')
            context = tracking_views[image_path.stem]
            pointmap = np.asarray(context.get('pointmap'), dtype=float)
            camera_to_scene = np.asarray(context.get('camera_to_scene'), dtype=float)
            if (pointmap.ndim != 3 or pointmap.shape[-1] != 3
                    or camera_to_scene.shape != (4, 4)
                    or not np.isfinite(camera_to_scene).all()):
                raise ValueError(f'{image_path.name}: invalid DA3 tracking context')

            with torch.inference_mode(), inference_context:
                inference_state = self.processor.set_image(image)
                output = self.processor.set_text_prompt(state=inference_state, prompt=text_prompt)
            raw_masks, raw_scores = output['masks'], output['scores']
            masks = raw_masks.detach().cpu().numpy() if torch.is_tensor(raw_masks) else np.asarray(raw_masks)
            scores = raw_scores.float().detach().cpu().numpy() if torch.is_tensor(raw_scores) else np.asarray(raw_scores)
            if masks.ndim == 4 and masks.shape[1] == 1:
                masks = masks[:, 0]
            selected, dropped = select_instance_candidates(
                masks, scores, max_candidates=max(16, expected_count * 4))
            candidate_rows, candidate_masks = [], {}
            for candidate_index in selected:
                mask = np.asarray(masks[candidate_index], dtype=bool)
                pointmap_mask = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize(
                    (pointmap.shape[1], pointmap.shape[0]), Image.Resampling.NEAREST)) > 0
                valid = pointmap_mask & np.isfinite(pointmap).all(axis=-1) & (pointmap[..., 2] > 0)
                points = pointmap[valid]
                if len(points) < 3:
                    continue
                if len(points) > 4096:
                    points = points[np.linspace(0, len(points) - 1, 4096, dtype=int)]
                points_h = np.column_stack((points, np.ones(len(points))))
                scene_points = (camera_to_scene @ points_h.T).T[:, :3]
                scene_points = scene_points[np.isfinite(scene_points).all(axis=1)]
                if len(scene_points) < 3:
                    continue
                center = np.median(scene_points, axis=0)
                bounds = np.percentile(scene_points, [5, 95], axis=0)
                radius = max(float(np.linalg.norm(bounds[1] - bounds[0]) / 2), 1e-4)
                candidate_rows.append({
                    'candidate_index': int(candidate_index), 'center': center,
                    'radius': radius, 'score': float(scores[candidate_index]),
                })
                # Store compressed binary masks once; avoid keeping RGBA images
                # or full source frames in memory across the complete capture.
                candidate_masks[int(candidate_index)] = mask.astype(np.uint8)
            candidates_by_view[image_path.stem] = candidate_rows
            masks_by_view[image_path.stem] = candidate_masks
            view_reports[image_path.stem] = {
                'image': image_path.name, 'candidate_count': len(scores),
                'selected_count': len(candidate_rows), 'selected_indices': [
                    row['candidate_index'] for row in candidate_rows],
                'dropped_candidates': dropped,
            }

        tracking = track_instances(candidates_by_view, expected_count)
        name_by_track = {f'instance_{index:02d}': name
                         for index, name in enumerate(object_names, 1)}
        for track_id, track in tracking['instances'].items():
            mask_dir = output_dir / name_by_track[track_id]
            mask_dir.mkdir(parents=True, exist_ok=True)
            track['object_name'] = name_by_track[track_id]
            for image_path in image_files:
                stem = image_path.stem
                candidate = tracking['instances'][track_id]['views'][stem]
                image = Image.open(image_path).convert('RGB')
                rgba = np.zeros((image.height, image.width, 4), dtype=np.uint8)
                if candidate is not None:
                    mask = masks_by_view[stem][candidate['candidate_index']]
                    mask = np.asarray(Image.fromarray(mask * 255).resize(
                        image.size, Image.Resampling.NEAREST)) > 0
                    rgb = np.asarray(image)
                    rgba[mask, :3] = rgb[mask]
                    rgba[mask, 3] = 255
                Image.fromarray(rgba, 'RGBA').save(mask_dir / f'{stem}.png')
                view_reports[stem].setdefault('instances', {})[track_id] = {
                    'object_name': name_by_track[track_id],
                    'selected_count': int(candidate is not None),
                    'selected_indices': [] if candidate is None else [candidate['candidate_index']],
                    'success': True, 'occluded': candidate is None,
                }

        return {
            'success': True, 'object_name': object_name,
            'text_prompt': text_prompt, 'expected_count': expected_count,
            'total_views': len(image_files), 'success_views': len(image_files),
            'views': [view_reports[path.stem] for path in image_files],
            'tracking': tracking,
            'objects': [
                {'object_name': name_by_track[track_id], 'success': True,
                 'expected_count': 1, 'failed_views': [], 'instance_tracking': True,
                 'success_views': track['visible_views'], 'total_views': len(image_files),
                 'views': [view_reports[path.stem]['instances'][track_id]
                           | {'image': path.name} for path in image_files]}
                for track_id, track in tracking['instances'].items()
            ],
        }
    
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
        mask_dir = preflight_mask_outputs(image_files, output_dir, object_name)
        if not image_files:
            return {'success': False, 'error': 'No images found'}
        
        logger.info(f"  Processing {len(image_files)} views...")
        
        # 创建物体的 mask 目录
        mask_dir.mkdir(parents=True, exist_ok=True)
        
        # 对每个视角进行分割
        success_count = 0
        failed_views = []
        view_reports = []
        
        for i, img_path in enumerate(image_files):
            view_report = {'image': img_path.name, 'text_prompt': text_prompt,
                           'confidence_threshold': self.confidence_threshold,
                           'expected_count': expected_count, 'candidate_count': None,
                           'selected_count': 0, 'selected_indices': [], 'scores': [],
                           'dropped_candidates': [], 'success': False}
            view_reports.append(view_report)
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
                view_report['candidate_count'] = len(scores_np)
                selected = select_instance_indices(masks_np, scores_np, expected_count, report=view_report)
                mask_np = np.logical_or.reduce(masks_np[selected])
                
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
                
                view_report.update(success=True, area_ratio=float(area_ratio))
                logger.info(f"  View {i}: ✓ (instances={len(selected)}, area={area_ratio*100:.1f}%)")
                success_count += 1
                
            except Exception as e:
                # A failed rerun must not leave this view's old mask usable.
                (mask_dir / f"{img_path.stem}.png").unlink(missing_ok=True)
                logger.error(f"  View {i}: Failed - {e}")
                import traceback
                traceback.print_exc()
                failed_views.append(i)
                view_report['error'] = str(e)
        
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
