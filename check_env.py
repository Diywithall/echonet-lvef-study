import sys
print(sys.version)
print(sys.path)
try:
    import torch
    print("torch:", torch.__version__)
except ImportError as e:
    print("torch not found:", e)
try:
    import cv2
    print("cv2:", cv2.__version__)
except ImportError as e:
    print("cv2 not found:", e)
