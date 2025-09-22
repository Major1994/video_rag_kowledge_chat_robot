import faiss
import pickle
import numpy as np
import json
import os

# 图片向量库
video_frame_embedding_file = "/root/autodl-tmp/video_frame_embedding.pickle"
with open(video_frame_embedding_file,"rb") as inf:
    frame_embedding = pickle.load(inf)

frame_embedding_index = faiss.IndexFlatL2(512)
img_path_dict={}
img_embedding_list = []
for video_path, video_frames in frame_embedding.items():

    for frame_embedding in video_frames:
        img_path_dict[len(img_path_dict)] = video_path
        img_embedding_list.append(frame_embedding)

img_embedding_array = np.array(img_embedding_list)
print(img_embedding_array.shape)
frame_embedding_index.add(img_embedding_array)
faiss.write_index(frame_embedding_index, "img_index.faiss")  
with open("img_path_dict.json","w") as f:
    json.dump(img_path_dict,f,ensure_ascii=False)
    
# 文字向量库
video_description_embedding_file = "/root/autodl-tmp/video_description_embedding.pickle"
with open(video_description_embedding_file,"rb") as inf:
    text_embedding = pickle.load(inf)

text_index = faiss.IndexFlatL2(4096)
text_path_dict={}
text_embedding_list = []

for text_path, text_embedding_value in text_embedding.items():
    # print(text_embedding_value)
    text_path_dict[len(text_path_dict)] = os.path.join("/root/autodl-tmp/video",text_path)
    text_embedding_list.append(text_embedding_value)

text_embedding_array = np.array(text_embedding_list)
print(text_embedding_array.shape)
text_index.add(text_embedding_array)
faiss.write_index(text_index, "text_index.faiss")  
with open("text_path_dict.json","w") as f:
    json.dump(text_path_dict,f,ensure_ascii=False)

# 文字倒排索引库
import os
import jieba
from whoosh.index import create_in
from whoosh.fields import Schema, TEXT, ID
from whoosh.analysis import Tokenizer, Token

video_description_file = "/root/autodl-tmp/video_description.jsonl"

# 创建使用结巴分词的分析器
class JiebaTokenizer(Tokenizer):
     def __call__(self, text, positions=True,**kwargs):
        tokens = jieba.cut(text)
        for token in tokens:
            t = Token()
            t.text = token
            t.pos = kwargs.get("pos", 0)  # 设置位置信息（可选）
            
            yield t

jieba_analyzer = JiebaTokenizer()

# 定义索引结构 
schema = Schema(
    id=ID(unique=True, stored=True), 
    content=TEXT(stored=True,analyzer=jieba_analyzer),
)

# 索引目录
if not os.path.exists("indexdir"):
    os.makedirs("indexdir")

ix = create_in("indexdir", schema)  
writer = ix.writer()
with open(video_description_file) as inf:
    for line in inf:
        line = json.loads(line)
        for video_path, video_descrition in line.items():
            writer.add_document(id=os.path.join("/root/autodl-tmp/video",text_path),content=video_descrition)

writer.commit()
