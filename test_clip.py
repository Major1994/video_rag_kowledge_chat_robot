from PIL import Image
import requests

from transformers import CLIPProcessor, CLIPModel
import torch

#提取图像向量特征
def normal(vector):
    ss=float(sum([s**2 for s in vector])**0.5)
    return [float(s)/ss for s in vector]

model_name="clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name).to("cuda:0")
processor = CLIPProcessor.from_pretrained(model_name)
#
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



if __name__ == "__main__":
    path = "8562217216959202.png"
    #提取文字对应的向量
    vector_text=extract_text_vector("一辆黑色本田摩托车 背着黑色的布贡迪座椅")
 

    #提取图像对应的向量
    image = Image.open(path)
    vector_img=extract_img_vector(image)
 
    similar=sum([ s1*s2 for s1,s2 in  zip(vector_img,vector_text)])
    print ("similar",similar)

    