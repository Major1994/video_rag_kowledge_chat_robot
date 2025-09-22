from PIL import Image
import requests

from transformers import CLIPProcessor, CLIPModel
import torch

import pickle

import cv2
import numpy as np

import os

from tqdm import tqdm

model_name="clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name).to("cuda:0")
processor = CLIPProcessor.from_pretrained(model_name)

##提取视频关键帧
def extract_keyframes(video_path, topK=5):
    cap = cv2.VideoCapture(video_path)
    keyframes = []
    ret, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    results=[]
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        #每一帧的灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 计算帧间差异（直方图或像素差值）
        diff = cv2.absdiff(prev_gray, gray)
        diff_mean = np.mean(diff)
        results.append([frame,diff_mean])
        prev_gray = gray  
    cap.release()
    results=sorted(results,key=lambda s:s[1],reverse=True)[0:topK]
    return [s[0] for s in results]

#提取图像向量特征
def normal(vector):
    ss=float(sum([s**2 for s in vector])**0.5)
    return [float(s)/ss for s in vector]

def extract_img_vector(image):  
    #图像部分
    img_model=model.vision_model
    inputs = processor(images=image, return_tensors="pt", padding=True).to("cuda:0")
    img_vector=img_model(**inputs).pooler_output[0]
    img_vector=model.visual_projection(img_vector)
    img_vector= normal(img_vector)
    return img_vector

def extract_text_vector(text):
    #文字部分
    inputs = processor(text=[text],  return_tensors="pt", padding=True).to("cuda:0")
    text_model=model.text_model
    text_vector= text_model(**inputs).pooler_output[0]
    text_vector=model.text_projection(text_vector)
    text_vector= normal(text_vector)
    return text_vector


def extract_video_frame_embedding(path):
    results=[]
    type="video" if path.endswith("mp4") else "image"

    if type=="video":
        frames=extract_keyframes(path)
    else:
        frames=[Image.open(path)]
    for img in frames:
        vector=extract_img_vector(img)
        results.append(vector)
    return results


if __name__ == "__main__":
    video_dir = "/root/autodl-tmp/video"
    output_file = "video_frame_embedding.pickle"
    result = {}
    for video_file in tqdm(os.listdir(video_dir)):
        video_path = os.path.join(video_dir, video_file)
        try:
            video_frame_embedding_list = extract_video_frame_embedding(video_path)
            result[video_path] = video_frame_embedding_list
        except Exception as e:
            print(video_path)
            print(e)

    print(result)
    with open(output_file, 'wb') as file:
        pickle.dump(result, file)
    
