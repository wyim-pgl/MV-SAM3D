"""Issue #2 validation only: SAM 1 with manually supplied boxes/points.
Not automatic text segmentation. Each bearing is segmented separately, then unioned.
Run from MV-SAM3D root in the existing mvsam3d environment.
"""
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import SamModel, SamProcessor

root = Path('data/issue2_cup_bearings')
# Coordinates are in the original downloaded images, without resizing.
prompts = [
    {'cup': [15, 4, 375, 420], 'bearings': [[171,501],[176,524],[177,548],[178,572],[179,596],[182,621],[187,646]], 'r': 13},
    {'cup': [17,178,253,478], 'bearings': [[292,446],[309,446],[326,447],[344,448],[361,448],[378,449],[394,449],[411,450]], 'r': 10},
    {'cup': [150,294,501,667], 'bearings': [[94,125],[107,146],[119,166],[130,187],[141,208],[151,227],[162,244]], 'r': 12},
]
processor = SamProcessor.from_pretrained('facebook/sam-vit-huge', local_files_only=True)
model = SamModel.from_pretrained('facebook/sam-vit-huge', local_files_only=True).to('cuda').eval()
report = []
preview = Image.new('RGB', (3*502, 2*668), 'white')
for i, spec in enumerate(prompts):
    image = Image.open(root/'images'/f'{i}.png').convert('RGB')
    w,h = image.size
    r = spec['r']
    boxes = [spec['cup']] + [[max(0,x-r), max(0,y-r), min(w,x+r), min(h,y+r)] for x,y in spec['bearings']]
    inputs = processor(image, input_boxes=[boxes], return_tensors='pt').to('cuda')
    with torch.inference_mode():
        pred = model(**inputs, multimask_output=True)
    masks = processor.image_processor.post_process_masks(pred.pred_masks.cpu(), inputs['original_sizes'].cpu(), inputs['reshaped_input_sizes'].cpu())[0].numpy()
    scores = pred.iou_scores.cpu().numpy()[0]
    chosen = [masks[j, scores[j].argmax()].astype(bool) for j in range(len(boxes))]
    for row, (obj,mask) in enumerate([('red_cup',chosen[0]),('ball_bearings',np.logical_or.reduce(chosen[1:]))]):
        assert mask.any() and not mask.all()
        folder = root/obj; folder.mkdir(exist_ok=True)
        rgba = np.dstack([np.asarray(image), mask.astype(np.uint8)*255])
        Image.fromarray(rgba).save(folder/f'{i}.png')
        overlay = np.asarray(image).copy()
        overlay[mask] = (overlay[mask]*0.45 + np.array([0,255,0])*0.55).astype(np.uint8)
        vis = Image.fromarray(overlay)
        ImageDraw.Draw(vis).text((5,5), f'{obj} view {i}', fill='yellow')
        preview.paste(vis, (i*502,row*668))
        report.append({'view':i,'object':obj,'foreground_pixels':int(mask.sum()),'area_fraction':float(mask.mean())})
    print('view', i, 'scores', scores.max(axis=1).tolist(), flush=True)
root.joinpath('sam1_prompt_report.json').write_text(json.dumps({'method':'SAM1 manual boxes, per-bearing union', 'prompts':prompts, 'masks':report}, indent=2))
preview.save(root/'mask_overlays.jpg')
print('Masks and preview saved:',root,flush=True)
