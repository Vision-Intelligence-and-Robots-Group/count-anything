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
parser.add_argument("-dp", "--data_path", type=str, default='/data/counte/', help="Path to the FSC147 dataset")
parser.add_argument("-ts", "--test_split", type=str, default='val', choices=["val_PartA","val_PartB","test_PartA","test_PartB","test", "val"], help="what data split to evaluate on")
parser.add_argument("-mt", "--model_type", type=str, default="vit_h", help="model type")
parser.add_argument("-mp",  "--model_path", type=str, default="/home/teddy/segment-anything/sam_vit_h_4b8939.pth", help="path to trained model")
parser.add_argument("-v",  "--viz", type=bool, default=True, help="wether to visualize")
parser.add_argument("-d",   "--device", default='0', help='assign device')
args = parser.parse_args()

data_path = args.data_path
anno_file = data_path + 'annotation_FSC147_384.json'
data_split_file = data_path + 'Train_Test_Val_FSC_147.json'
im_dir = data_path + 'images_384_VarV2'


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

with open(data_split_file) as f:
    data_split = json.load(f)


cnt = 0
SAE = 0  # sum of absolute errors
SSE = 0  # sum of square errors

print("Evaluation on {} data".format(args.test_split))
im_ids = data_split[args.test_split]

# with open("err.json") as f:
#     im_ids = json.load(f)


pbar = tqdm(im_ids)
# err_list = []
for im_id in pbar:
    anno = annotations[im_id]
    bboxes = anno['box_examples_coordinates']
    dots = np.array(anno['points'])

    image = cv2.imread('{}/{}'.format(im_dir, im_id))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    input_boxes = list()
    for bbox in bboxes:
        x1, y1 = bbox[0][0], bbox[0][1]
        x2, y2 = bbox[2][0], bbox[2][1]
        input_boxes.append([x1, y1, x2, y2])
    
    masks = mask_generator.generate(image, input_boxes)
    if args.viz:
        if not exists('viz'):
            os.mkdir('viz')
        plt.figure(figsize=(10,10))
        plt.imshow(image)
        show_anns(masks)
        plt.axis('off')
        plt.savefig('viz/{}'.format(im_id))
        plt.close()

    gt_cnt = dots.shape[0]
    pred_cnt = len(masks)
    cnt = cnt + 1
    err = abs(gt_cnt - pred_cnt)
    SAE += err
    SSE += err**2
    # if err / gt_cnt > 0.7:
    #     err_list.append(im_id)

    pbar.set_description('{:<8}: actual-predicted: {:6d}, {:6.1f}, error: {:6.1f}. Current MAE: {:5.2f}, RMSE: {:5.2f}'.\
                         format(im_id, gt_cnt, pred_cnt, abs(pred_cnt - gt_cnt), SAE/cnt, (SSE/cnt)**0.5))

print('On {} data, MAE: {:6.2f}, RMSE: {:6.2f}'.format(args.test_split, SAE/cnt, (SSE/cnt)**0.5))
# with open('err.json', "w") as f:
#     json.dump(err_list, f)