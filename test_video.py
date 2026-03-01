import os
import argparse
import torch
import cv2
import numpy as np

from torchvision import transforms
from models.fast_scnn import get_fast_scnn
from PIL import Image


parser = argparse.ArgumentParser(description='Predict drivable area from video')
parser.add_argument('--model', type=str, default='fast_scnn')
parser.add_argument('--dataset', type=str, default='citys')
parser.add_argument('--weights-folder', default='./weights')
parser.add_argument('--input-video', type=str, default='0',
                    help='path to input video file or 0 for webcam')
parser.add_argument('--out-video', type=str, default='./test_result/output.mp4',
                    help='path to save output video')
parser.add_argument('--cpu', action='store_true')
args = parser.parse_args()


def demo_video():
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    # Image transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    # Load model
    model = get_fast_scnn(
        args.dataset,
        pretrained=True,
        root=args.weights_folder,
        map_cpu=args.cpu
    ).to(device)

    print("✅ Model loaded")
    model.eval()

    # Video source
    cap = cv2.VideoCapture(0 if args.input_video == "0" else args.input_video)
    if not cap.isOpened():
        print("❌ Error opening video source")
        return

    # Output video writer
    os.makedirs(os.path.dirname(args.out_video), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 20
    out = cv2.VideoWriter(args.out_video, fourcc, fps, (1024, 512))

    road_class_id = 0
    previous_mask = None   # ✅ Proper temporal smoothing storage

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame, (1024, 512))
        pil_img = Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
        image = transform(pil_img).unsqueeze(0).to(device)

        # ---------- Inference ----------
        with torch.no_grad():
            outputs = model(image)
            probs = torch.softmax(outputs[0], dim=0).cpu().numpy()
            road_prob = probs[road_class_id]

            confidence_threshold = 0.6
            road_mask = (road_prob > confidence_threshold).astype(np.uint8) * 255

        # ---------- Strong Morphology Cleanup ----------
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

        # ---------- Remove Small Components ----------
        min_area = 5000
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        clean_mask = np.zeros_like(mask)

        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                clean_mask[labels == i] = 255

        # ---------- Keep Only Largest Road Region ----------
        contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        final_mask = np.zeros_like(clean_mask)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)

            epsilon = 0.02 * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)
            hull = cv2.convexHull(approx)

            cv2.fillPoly(final_mask, [hull], 255)

        # ---------- Temporal Smoothing ----------
        if previous_mask is None:
            previous_mask = final_mask.copy()

        alpha = 0.7
        final_mask = cv2.addWeighted(final_mask, alpha,
                                     previous_mask, 1 - alpha, 0)

        previous_mask = final_mask.copy()
        _, final_mask = cv2.threshold(final_mask, 127, 255, cv2.THRESH_BINARY)

        # ---------- Overlay ----------
        overlay = frame_resized.copy()

        fill_color = (0, 255, 0)
        border_color = (0, 200, 255)

        overlay[final_mask == 255] = fill_color

        contours_draw, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours_draw, -1, border_color, thickness=4)

        vis = cv2.addWeighted(frame_resized, 0.6, overlay, 0.4, 0)

        cv2.imshow("Fast-SCNN Clean Drivable Area", vis)
        out.write(vis)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("✅ Output saved to:", args.out_video)


if __name__ == '__main__':
    demo_video()