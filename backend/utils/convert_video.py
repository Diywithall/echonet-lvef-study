import cv2
import os
import argparse
import numpy as np

def convert_to_echonet_format(input_path, output_path):
    """
    Standardizes video to 112x112, 50fps (or original), and masks 
    to match EchoNet-Dynamics training data preparation.
    """
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found")
        return

    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Define codec and writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (112, 112))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Square Crop (Center)
        size = min(width, height)
        start_x = (width - size) // 2
        start_y = (height - size) // 2
        frame = frame[start_y:start_y+size, start_x:start_x+size]

        # 2. Resize to 112x112
        frame = cv2.resize(frame, (112, 112), interpolation=cv2.INTER_CUBIC)

        out.write(frame)

    cap.release()
    out.release()
    print(f"Conversion complete: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert echocardiogram videos to EchoNet format")
    parser.add_argument("input", help="Path to input video (.mp4, .avi)")
    parser.add_argument("output", help="Path to output video (.avi)")
    args = parser.parse_args()
    
    convert_to_echonet_format(args.input, args.output)
