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
parser.add_argument('--input-video', type=str, default=0,  # 0 = webcam
                    help='path to input video file or 0 for webcam')
parser.add_argument('--out-video', type=str, default='./test_result/output.mp4',
                    help='path to save output video')
parser.add_argument('--cpu', action='store_true')
args = parser.parse_args()


def demo_video():
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    # Transform
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

    # Video writer
    os.makedirs(os.path.dirname(args.out_video), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 20
    out = cv2.VideoWriter(args.out_video, fourcc, fps, (1024, 512))

    road_class_id = 0  # Cityscapes road

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame to model input size
        frame_resized = cv2.resize(frame, (1024, 512))
        pil_img = Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))

        image = transform(pil_img).unsqueeze(0).to(device)

        # Inference
        with torch.no_grad():
            outputs = model(image)
            pred = torch.argmax(outputs[0], 1).squeeze(0).cpu().numpy()

        # Drivable area mask
        road_mask = (pred == road_class_id)

        overlay = frame_resized.copy()
        overlay[road_mask] = [0, 255, 0]

        vis = cv2.addWeighted(frame_resized, 0.6, overlay, 0.4, 0)

        cv2.imshow("Fast-SCNN Drivable Area", vis)
        out.write(vis)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("✅ Output video saved to:", args.out_video)


if __name__ == '__main__':
    demo_video()
