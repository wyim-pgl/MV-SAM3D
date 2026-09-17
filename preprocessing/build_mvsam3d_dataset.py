#!/usr/bin/env python3
"""
MV-SAM3D 数据预处理主脚本

使用方法：
    # 单场景处理
    python preprocessing/build_mvsam3d_dataset.py \
        --input data/dog_cat_table \
        --objects dog,cat,table \
        --run_da3
    
    # 批量处理
    python preprocessing/build_mvsam3d_dataset.py \
        --batch \
        --scenes dog_cat_table,dog_table,dog_cat_table_sofa
"""
import sys
import json
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict
from loguru import logger

# 导入本地模块
from data_organizer import organize_images
from sam3_segmenter import (
    SAM3MultiObjectSegmenter, preflight_mask_outputs, reject_source_aliases,
)


def run_da3(scene_dir: Path) -> bool:
    """
    运行 DA3 处理
    
    Args:
        scene_dir: 场景目录（包含 images/）
    
    Returns:
        True if successful
    """
    logger.info(f"\n[Running DA3]")
    
    images_dir = scene_dir / "images"
    if not images_dir.exists():
        logger.error("  images/ directory not found")
        return False
    
    scene_name = scene_dir.name
    output_path = Path("da3_outputs") / scene_name / "da3_output.npz"
    
    # 检查是否已经处理过
    if output_path.exists():
        logger.info(f"  DA3 output already exists: {output_path}")
        logger.info(f"  Skipping DA3 processing")
        return True
    
    # 运行 DA3（注意参数名称是 --image_dir 和 --output_dir）
    output_dir = output_path.parent
    cmd = [
        "python", "scripts/run_da3.py",
        "--image_dir", str(images_dir),
        "--output_dir", str(output_dir),
    ]
    
    logger.info(f"  Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.success(f"✓ DA3 completed: {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"  DA3 failed: {e}")
        logger.error(f"  stdout: {e.stdout}")
        logger.error(f"  stderr: {e.stderr}")
        return False


def process_scene(
    scene_dir: Path,
    objects: List[str],
    prompts: Dict[str, str] = None,
    run_da3_flag: bool = False,
    sam3_checkpoint: Path = None,
    sam3_root: Path = None,
    counts: Dict[str, int] = None,
    confidence_threshold: float = 0.1,
) -> Dict:
    """
    处理单个场景
    
    Args:
        scene_dir: 场景目录
        objects: 物体名称列表
        prompts: 物体名称到prompt的映射（可选，默认用物体名称作为prompt）
        run_da3_flag: 是否运行 DA3
        sam3_checkpoint: SAM3 checkpoint 路径
    
    Returns:
        Dict with processing result
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing scene: {scene_dir.name}")
    logger.info(f"Objects: {objects}")
    logger.info(f"{'='*60}")
    
    result = {
        'scene': scene_dir.name,
        'success': False,
        'steps': {},
    }
    
    # Check every destination for every object before even invalidating the report.
    # Include original loose images and existing organized images in alias checks.
    images_dir = scene_dir / 'images'
    originals = [p for p in scene_dir.iterdir()
                 if p.suffix.lower() in {'.png', '.jpg', '.jpeg'} and p.is_file()]
    existing = list(images_dir.glob('*.png')) + list(images_dir.glob('*.jpg'))
    sources = originals + existing
    planned = [images_dir / f'{i}.png' for i in range(len(originals))] if originals else existing
    if originals:
        reject_source_aliases(planned, sources)
    report_path = scene_dir / 'segmentation_report.json'
    reject_source_aliases([report_path], sources)
    if not objects:
        raise ValueError('At least one object is required')
    for obj_name in objects:
        mask_dir = preflight_mask_outputs(planned, scene_dir, obj_name)
        reject_source_aliases([mask_dir / f'{p.stem}.png' for p in planned], sources)

    # Invalidate any earlier successful report before starting a new run.
    report_path.write_text(json.dumps(result, indent=2))

    # Step 1: 数据组织
    logger.info("\n[Step 1/3] Organizing images...")
    org_result = organize_images(scene_dir)
    result['steps']['organize'] = org_result
    
    if not org_result['success']:
        logger.error("  Failed to organize images")
        report_path.write_text(json.dumps(result, indent=2, default=str))
        return result
    
    images_dir = scene_dir / "images"
    
    # Step 2: SAM3 分割
    logger.info("\n[Step 2/3] SAM3 segmentation...")
    
    # 初始化 SAM3
    try:
        segmenter = SAM3MultiObjectSegmenter(
            checkpoint_path=sam3_checkpoint,
            sam3_root=sam3_root,
            confidence_threshold=confidence_threshold
        )
    except Exception as exc:
        result['error'] = str(exc)
        report_path.write_text(json.dumps(result, indent=2, default=str))
        logger.error(f"SAM 3 initialization failed: {exc}")
        return result
    
    # 对每个物体进行分割
    seg_results = []
    for obj_name in objects:
        # 确定 prompt
        if prompts and obj_name in prompts:
            prompt = prompts[obj_name]
        else:
            prompt = obj_name
        
        # 分割
        try:
            seg_result = segmenter.segment_object_multiview(
                images_dir=images_dir,
                object_name=obj_name,
                text_prompt=prompt,
                output_dir=scene_dir,
                expected_count=counts.get(obj_name, 1) if counts else 1,
            )
        except Exception as exc:
            seg_result = {'object_name': obj_name, 'success': False, 'error': str(exc)}
        seg_results.append(seg_result)
    
    result['steps']['segmentation'] = seg_results
    
    # 检查是否所有物体都成功分割
    all_success = all(r['success'] for r in seg_results)
    if not all_success:
        logger.warning("  Some objects failed to segment")
    
    # Step 3: DA3 处理
    if run_da3_flag and all_success:
        logger.info("\n[Step 3/3] DA3 processing...")
        da3_success = run_da3(scene_dir)
        result['steps']['da3'] = {'success': da3_success}
    else:
        logger.info("\n[Step 3/3] DA3 skipped (requires --run_da3 and masks for every view)")
        result['steps']['da3'] = {'success': None, 'skipped': True}
    
    # 最终结果
    result['success'] = all_success and (not run_da3_flag or result['steps']['da3']['success'] is True)
    report_path.write_text(json.dumps(result, indent=2, default=str))
    
    logger.info(f"\n{'='*60}")
    if result['success']:
        logger.success(f"✓ Scene '{scene_dir.name}' processed successfully!")
    else:
        logger.warning(f"⚠ Scene '{scene_dir.name}' completed with warnings")
    logger.info(f"{'='*60}")
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="MV-SAM3D Dataset Preprocessing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # 输入
    parser.add_argument("--input", type=str, help="Scene directory (e.g., data/dog_cat_table)")
    parser.add_argument("--objects", type=str, help="Comma-separated object names (e.g., dog,cat,table)")
    parser.add_argument("--prompts", help="Comma-separated text prompts matching --objects (single scene)")
    parser.add_argument("--counts", help="Comma-separated positive instance counts matching --objects (single scene)")
    parser.add_argument("--confidence_threshold", type=float, default=0.1,
                        help="SAM 3 detection threshold in [0,1] (default: 0.1)")
    
    # 批量处理
    parser.add_argument("--batch", action="store_true", help="Batch processing mode")
    parser.add_argument("--scenes", type=str, help="Comma-separated scene names for batch mode")
    parser.add_argument("--data_dir", type=str, default="data", help="Base data directory for batch mode")
    
    # DA3
    parser.add_argument("--run_da3", action="store_true", help="Run DA3 after segmentation")
    
    # SAM3
    parser.add_argument("--sam3_checkpoint", type=str, default=None,
                        help="SAM3 checkpoint path (default: $SAM3_CHECKPOINT)")
    parser.add_argument("--sam3_root", type=str, default=None,
                        help="Path to a SAM 3 checkout, if sam3 is not installed "
                             "in this environment (default: $SAM3_ROOT)")
    
    args = parser.parse_args()
    if not 0 <= args.confidence_threshold <= 1:
        parser.error('--confidence_threshold must be in [0,1]')
    if args.batch and (args.prompts is not None or args.counts is not None):
        parser.error('--prompts and --counts require single-scene --input mode')
    
    # 批量处理模式
    if args.batch:
        if not args.scenes:
            logger.error("--scenes required for batch mode")
            sys.exit(1)
        
        scene_names = [s.strip() for s in args.scenes.split(',')]
        data_dir = Path(args.data_dir)
        
        logger.info(f"Batch processing {len(scene_names)} scenes...")
        
        results = []
        for scene_name in scene_names:
            scene_dir = data_dir / scene_name
            if not scene_dir.exists():
                logger.error(f"Scene directory not found: {scene_dir}")
                continue
            
            # 交互式输入物体列表
            logger.info(f"\n[Scene: {scene_name}]")
            objects_input = input(f"Enter objects for {scene_name} (comma-separated): ").strip()
            objects = [o.strip() for o in objects_input.split(',') if o.strip()]
            
            if not objects:
                logger.warning(f"No objects specified for {scene_name}, skipping")
                continue
            
            # 处理
            result = process_scene(
                scene_dir=scene_dir,
                objects=objects,
                run_da3_flag=args.run_da3,
                sam3_checkpoint=Path(args.sam3_checkpoint) if args.sam3_checkpoint else None,
                sam3_root=Path(args.sam3_root) if args.sam3_root else None,
                confidence_threshold=args.confidence_threshold,
            )
            results.append(result)
        
        # 总结
        logger.info(f"\n{'='*60}")
        logger.info(f"BATCH PROCESSING SUMMARY")
        logger.info(f"{'='*60}")
        success_count = sum(1 for r in results if r['success'])
        logger.info(f"Total scenes: {len(results)}")
        logger.info(f"✓ Success: {success_count}")
        logger.info(f"⚠ Warnings: {len(results) - success_count}")
        
    # 单场景处理模式
    else:
        if not args.input:
            logger.error("--input required for single scene mode")
            sys.exit(1)
        
        if not args.objects:
            logger.error("--objects required for single scene mode")
            sys.exit(1)
        
        scene_dir = Path(args.input)
        if not scene_dir.exists():
            logger.error(f"Scene directory not found: {scene_dir}")
            sys.exit(1)
        
        objects = [o.strip() for o in args.objects.split(',') if o.strip()]
        if not objects or len(set(objects)) != len(objects):
            parser.error('--objects must contain unique, nonempty names')
        prompts = [p.strip() for p in args.prompts.split(',')] if args.prompts is not None else objects
        try:
            counts = [int(c.strip()) for c in args.counts.split(',')] if args.counts is not None else [1] * len(objects)
        except ValueError:
            parser.error('--counts must contain positive integers')
        if len(prompts) != len(objects) or not all(prompts):
            parser.error('--prompts must contain one nonempty prompt per object')
        if len(counts) != len(objects) or any(c < 1 for c in counts):
            parser.error('--counts must contain one positive integer per object')
        
        result = process_scene(
            scene_dir=scene_dir,
            objects=objects,
            run_da3_flag=args.run_da3,
            sam3_checkpoint=Path(args.sam3_checkpoint) if args.sam3_checkpoint else None,
            sam3_root=Path(args.sam3_root) if args.sam3_root else None,
            prompts=dict(zip(objects, prompts)),
            counts=dict(zip(objects, counts)),
            confidence_threshold=args.confidence_threshold,
        )
        
        if not result['success']:
            sys.exit(1)


if __name__ == "__main__":
    main()
