import torch
import sys 
sys.path.remove(r'/home/dell/.local/lib/python3.8/site-packages')
# print(sys.path)
# print(torch.cuda.is_available())
import cv2

print(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
