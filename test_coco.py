import cv2
import argparse
import json
import numpy as np
from tqdm import tqdm
from os.path import exists
import os

from segment_anything import sam_model_registry
from automatic_mask_generator import SamAutomaticMaskGenerator
import matplotlib.pyplot as plt




parser = argparse.ArgumentParser(description="Few Shot Counting Evaluation code")
parser.add_argument("-dp", "--data_path", type=str, default='/data/counte/', help="Path to the coco dataset")
parser.add_argument("-ts", "--test_split", type=str, default='val2017', choices=["val2017"], help="what data split to evaluate on")
parser.add_argument("-mt", "--model_type", type=str, default="vit_h", help="model type")
parser.add_argument("-mp",  "--model_path", type=str, default="/home/teddy/segment-anything/sam_vit_h_4b8939.pth", help="path to trained model")
parser.add_argument("-v",  "--viz", type=bool, default=True, help="wether to visualize")
parser.add_argument("-d",   "--device", default='0', help='assign device')
args = parser.parse_args()

data_path = args.data_path
anno_file = data_path + 'annotations_trainval2017/annotations/instances_val2017.json'
im_dir = data_path + 'val2017'


if not exists(anno_file) or not exists(im_dir):
    print("Make sure you set up the --data-path correctly.")
    print("Current setting is {}, but the image dir and annotation file do not exist.".format(args.data_path))
    print("Aborting the evaluation")
    exit(-1)

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    for ann in sorted_anns:
        x0, y0, w, h = ann['bbox']
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
        ax.scatter([x0+w//2], [y0+h//2], color='green', marker='*', s=10, edgecolor='white', linewidth=1.25)


debug = True
os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()
device = 'cuda'
sam = sam_model_registry[args.model_type](checkpoint=args.model_path)
sam.to(device=device)


mask_generator = SamAutomaticMaskGenerator(
                                model=sam,
                                min_mask_region_area=25
                                )

with open(anno_file) as f:
    annotations = json.load(f)
    
images = sorted(annotations['images'],key=lambda x:x['file_name'])

prepared_json = {}
for i in images:
    prepared_json[i['file_name']] = {
        "H":i['height'],
        "W":i['width'],
        "boxes":{},
        # "category_ids":[],
    }
for i in annotations['annotations']:
    im_id = str(i['image_id'])
    prezero = 12 - len(im_id)
    im_id = '0'*prezero + im_id + ".jpg"
    if i["category_id"] in prepared_json[im_id]["boxes"]:
        prepared_json[im_id]["boxes"][i["category_id"]].append(i['bbox'])
    else:
        prepared_json[im_id]["boxes"][i["category_id"]] = []
        prepared_json[im_id]["boxes"][i["category_id"]].append(i['bbox'])

im_ids = []
for i in prepared_json.keys():
    im_ids.append(i)


cnt = 0
folds = [
    [1,5,9,14,18,22,27,33,37,41,46,50,54,58,62,67,74,78,82,87],
    [2,6,10,15,19,23,28,34,38,42,47,51,55,59,63,70,75,79,84,88],
    [3,7,11,16,20,24,31,35,39,43,48,52,56,60,64,72,76,80,85,89],
    [4,8,13,17,21,25,32,36,40,44,49,53,57,61,65,73,77,81,86,90],
]
SAE = [0,0,0,0]  # sum of absolute errors
SSE = [0,0,0,0]  # sum of square errors

print("Evaluation on {} data".format(args.test_split))

# logs = []


pbar = tqdm(im_ids)
# err_list = []
for im_id in pbar:
    category_id = list(prepared_json[im_id]['boxes'].keys())
    
    image = cv2.imread('{}/{}'.format(im_dir, im_id))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # log = []
    # log.append(im_id)
    
    for id in category_id:
        boxes = prepared_json[im_id]['boxes'][id]
        
        input_boxes = list()
        x1, y1 = boxes[0][0],boxes[0][1]
        x2, y2 = boxes[0][0] + boxes[0][2],boxes[0][1] + boxes[0][3]
        input_boxes.append([x1, y1, x2, y2])
        
        masks = mask_generator.generate(image, input_boxes)
        
        if args.viz:
            if not exists('viz'):
                os.mkdir('viz')
            plt.figure(figsize=(10,10))
            plt.imshow(image)
            show_anns(masks)
            plt.axis('off')
            plt.savefig('viz/{}_{}.jpg'.format(im_id[0:-4],id))
            plt.close()
        
        gt_cnt = len(boxes)
        pred_cnt = len(masks)
        err = abs(gt_cnt - pred_cnt)
        log.append("\n{},gt_cnt:{},pred_cnt:{}".format(id,gt_cnt,pred_cnt))
        if id in folds[0]:
            SAE[0] += err
            SSE[0] += err**2
        elif id in folds[1]:
            SAE[1] += err
            SSE[1] += err**2
        elif id in folds[2]:
            SAE[2] += err
            SSE[2] += err**2
        elif id in folds[3]: 
            SAE[3] += err
            SSE[3] += err**2

    cnt = cnt + 1
    # logs.append(log)
    pbar.set_description('fold1: {:6.2f}, fold2: {:6.2f}, fold3: {:6.2f}, fold4: {:6.2f},'.\
                        format(SAE[0]/cnt,SAE[1]/cnt,SAE[2]/cnt,SAE[3]/cnt))
    
print('On {} data, fold1 MAE: {:6.2f}, RMSE: {:6.2f}\n \
    fold2 MAE: {:6.2f}, RMSE: {:6.2f}\n \
    fold3 MAE: {:6.2f}, RMSE: {:6.2f}\n \
    fold4 MAE: {:6.2f}, RMSE: {:6.2f}\n \
    '.format(args.test_split,SAE[0]/cnt,(SSE[0]/cnt)**0.5,SAE[1]/cnt,(SSE[1]/cnt)**0.5,SAE[2]/cnt,(SSE[2]/cnt)**0.5,SAE[3]/cnt,(SSE[3]/cnt)**0.5))