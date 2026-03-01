import os
import argparse
import torch
import cv2
import numpy as np

from torchvision import transforms
from models.fast_scnn import get_fast_scnn
from PIL import Image

parser = argparse.ArgumentParser(
    description='Predict segmentation result from a given image')
parser.add_argument('--model', type=str, default='fast_scnn',
                    help='model name (default: fast_scnn)')
parser.add_argument('--dataset', type=str, default='citys',
                    help='dataset name (default: citys)')
parser.add_argument('--weights-folder', default='./weights',
                    help='Directory for saving checkpoint models')
parser.add_argument('--input-pic', type=str,
                    default='./datasets/citys/leftImg8bit/test/berlin/berlin_000000_000019_leftImg8bit.png',
                    help='path to the input picture')
parser.add_argument('--outdir', default='./test_result', type=str,
                    help='path to save the predict result')

parser.add_argument('--cpu', dest='cpu', action='store_true')
parser.set_defaults(cpu=False)

args = parser.parse_args()


def demo():
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    # output folder
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # image transform (for model)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    # Load image for model
    pil_img = Image.open(args.input_pic).convert('RGB')
    pil_img = pil_img.resize((1024, 512))
    image = transform(pil_img).unsqueeze(0).to(device)

    # Load model
    model = get_fast_scnn(
        args.dataset,
        pretrained=True,
        root=args.weights_folder,
        map_cpu=args.cpu
    ).to(device)

    print('Finished loading model!')
    model.eval()

    # Inference
    with torch.no_grad():
        outputs = model(image)

    # Prediction mask
    pred = torch.argmax(outputs[0], 1).squeeze(0).cpu().numpy()

    # -------- Drivable Area Overlay (ROAD ONLY) --------
    road_class_id = 0  # Cityscapes: road = 0

    # Load original image for visualization
    orig = cv2.imread(args.input_pic)
    orig = cv2.resize(orig, (1024, 512))

    road_mask = (pred == road_class_id)

    overlay = orig.copy()
    overlay[road_mask] = [0, 255, 0]  # Green color for drivable area

    alpha = 0.5
    vis = cv2.addWeighted(orig, 1 - alpha, overlay, alpha, 0)

    outname = os.path.splitext(os.path.split(args.input_pic)[-1])[0] + '_road_overlay.png'
    cv2.imwrite(os.path.join(args.outdir, outname), vis)

    print(f"✅ Saved drivable area overlay to: {os.path.join(args.outdir, outname)}")


if __name__ == '__main__':
    demo()