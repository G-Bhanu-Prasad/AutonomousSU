import os
import argparse
import cv2
import numpy as np
import torch

from PIL import Image
from torchvision import transforms
from models.fast_scnn import get_fast_scnn


def parse_args():
    parser = argparse.ArgumentParser(description="Stable Road Detection with Motion Compensation")
    parser.add_argument("--dataset", type=str, default="citys")
    parser.add_argument("--weights-folder", default="./weights")
    parser.add_argument("--input-video", type=str, required=True)
    parser.add_argument("--out-video", type=str, default="./output.mp4")
    parser.add_argument("--cpu", action="store_true")

    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=512)

    parser.add_argument("--road-class-id", type=int, default=0)
    parser.add_argument("--prob-threshold", type=float, default=0.45)
    parser.add_argument("--min-area", type=int, default=1500)

    parser.add_argument("--overlay-alpha", type=float, default=0.4)
    parser.add_argument("--temporal-alpha", type=float, default=0.9)

    return parser.parse_args()


def build_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


# POSTPROCESSING
def get_largest_component(mask):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask

    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    output = np.zeros_like(mask)
    output[labels == largest] = 255
    return output


def fill_and_smooth(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask

    largest = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(largest)

    output = np.zeros_like(mask)
    cv2.drawContours(output, [hull], -1, 255, -1)
    return output


def process_mask(prob):
    prob = cv2.GaussianBlur(prob, (7, 7), 0)
    mask = (prob > 0.45).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, 2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, 1)

    mask = get_largest_component(mask)

    if cv2.countNonZero(mask) < 1500:
        return np.zeros_like(mask)

    mask = fill_and_smooth(mask)

    return mask


# OVERLAY
def overlay(frame, mask, alpha_val=0.4):
    green = np.zeros_like(frame)
    green[:, :, 1] = 255

    soft = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (15, 15), 0)
    alpha = (soft * alpha_val)[:, :, None]

    out = frame * (1 - alpha) + green * alpha
    return out.astype(np.uint8)


# MAIN
def main():
    args = parse_args()

    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    model = get_fast_scnn(
        args.dataset,
        pretrained=True,
        root=args.weights_folder,
        map_cpu=args.cpu
    ).to(device)

    model.eval()
    print("✅ Model loaded on", device)

    transform = build_transform()

    cap = cv2.VideoCapture(args.input_video)
    if not cap.isOpened():
        print("❌ Cannot open video")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 20

    out = cv2.VideoWriter(
        args.out_video,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (args.width, args.height)
    )

    prev_gray = None
    prev_mask = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (args.width, args.height))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        tensor = transform(pil).unsqueeze(0).to(device)

        with torch.no_grad():
            out_model = model(tensor)

            if isinstance(out_model, (list, tuple)):
                out_model = out_model[0]

            prob = torch.softmax(out_model, dim=1)[0, args.road_class_id].cpu().numpy()

        mask = process_mask(prob)

        # MOTION COMPENSATION
        if prev_gray is not None and prev_mask is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray,
                None,
                0.5, 3, 15, 3, 5, 1.2, 0
            )

            dx = np.mean(flow[..., 0])
            dy = np.mean(flow[..., 1])

            M = np.float32([[1, 0, dx], [0, 1, dy]])

            prev_mask = cv2.warpAffine(
                prev_mask,
                M,
                (mask.shape[1], mask.shape[0]),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT
            )

        # TEMPORAL SMOOTHING
        if prev_mask is None:
            prev_mask = mask.astype(np.float32)

        smoothed = (
            args.temporal_alpha * prev_mask +
            (1 - args.temporal_alpha) * mask.astype(np.float32)
        )

        smoothed = cv2.GaussianBlur(smoothed, (9, 9), 0)
        _, final_mask = cv2.threshold(smoothed.astype(np.uint8), 127, 255, cv2.THRESH_BINARY)

        prev_mask = smoothed
        prev_gray = gray

        vis = overlay(frame, final_mask, args.overlay_alpha)

        out.write(vis)

    cap.release()
    out.release()

    print("✅ Output saved:", args.out_video)


if __name__ == "__main__":
    main()