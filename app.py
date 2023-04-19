from PIL import Image, ImageDraw
import cv2
import gradio as gr
import torch
from segment_anything import sam_model_registry
from automatic_mask_generator import SamAutomaticMaskGenerator

device = 'cuda'
sam = sam_model_registry['vit_h'](checkpoint='./sam_vit_h_4b8939.pth')
sam.to(device=device)


mask_generator = SamAutomaticMaskGenerator(
                                model=sam,
                                min_mask_region_area=25
                                )

def binarize(x):
    return (x != 0).astype('uint8') * 255

def draw_box(boxes=[], img=None):
    if len(boxes) == 0 and img is None:
        return None

    if img is None:
        img = Image.new('RGB', (512, 512), (255, 255, 255))
    colors = ["red", "olive", "blue", "green", "orange", "brown", "cyan", "purple"]
    draw = ImageDraw.Draw(img)
    # print(boxes)
    for bid, box in enumerate(boxes):
        draw.rectangle([box[0], box[1], box[2], box[3]], outline=colors[bid % len(colors)], width=4)
    return img


def draw_pred_box(boxes=[], img=None):
    if len(boxes) == 0 and img is None:
        return None

    if img is None:
        img = Image.new('RGB', (512, 512), (255, 255, 255))
    colors = "green"
    draw = ImageDraw.Draw(img)
    # print(boxes)
    for bid, box in enumerate(boxes):
        draw.rectangle([box[0], box[1], box[2], box[3]], outline=colors, width=4)
    return img


def debug(input_img):
    mask = input_img["mask"]
    mask = mask[..., 0]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for contour in contours:
        y1, y2 = contour[:, 0, 1].min(), contour[:, 0, 1].max()
        x1, x2 = contour[:, 0, 0].min(), contour[:, 0, 0].max()
        boxes.append([x1, y1, x2, y2])
    draw_image = draw_box(boxes, Image.fromarray(input_img["image"]))

    masks = mask_generator.generate(input_img["image"], boxes)
    pred_cnt = len(masks)
    pred_bboxes = []
    for i in masks:
        x0, y0, w, h = i['bbox']
        pred_bboxes.append([x0, y0, x0+w, y0+h])
    pred_image = draw_pred_box(pred_bboxes, Image.fromarray(input_img["image"]))
    return [draw_image, pred_image, "Count: {}".format(pred_cnt)]

description = """<p style="text-align: center; font-weight: bold;">
    <span style="font-size: 28px">Count Anything</span>
    <br>
    <span style="font-size: 18px" id="paper-info">
        [<a href=" " target="_blank">Project Page</a>]
        [<a href=" " target="_blank">Paper</a>]
        [<a href="https://github.com/Vision-Intelligence-and-Robots-Group/count-anything" target="_blank">GitHub</a>]
    </span>
</p>
"""

run = gr.Interface(
    debug,
    gr.Image(shape=[512, 512], source="upload", tool="sketch").style(height=500, width=500),
    [gr.Image(), gr.Image(), gr.Text()],
    description = description
)

run.launch()