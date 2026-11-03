import sys
sys.path.insert(0, r'c:\Users\sarda\Downloads\pocus\pocus\backend\venv\Lib\site-packages')
print(sys.version)
try:
    import torch
    print("torch:", torch.__version__)
except ImportError as e:
    print("torch FAIL:", e)
try:
    import cv2
    print("cv2:", cv2.__version__)
except ImportError as e:
    print("cv2 FAIL:", e)
try:
    import numpy as np
    print("numpy:", np.__version__)
except ImportError as e:
    print("numpy FAIL:", e)
