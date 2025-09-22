import faiss
import numpy as np
import time
 
import jieba
import json

# 文字倒排索引
from whoosh.analysis import Tokenizer, Token
class JiebaTokenizer(Tokenizer):
     def __call__(self, text, positions=True,**kwargs):
        tokens = jieba.cut(text)
        for token in tokens:
            t = Token()
            t.text = token
            t.pos = kwargs.get("pos", 0)  # 设置位置信息（可选）
            
            yield t
from whoosh.qparser import QueryParser
from whoosh.index import open_dir

ix = open_dir("indexdir")
def tfidf_search(query):
    # 创建使用结巴分词的分析器
    words=jieba.cut(query)
    parser = QueryParser("content", schema=ix.schema)
    query = parser.parse(text=" OR ".join(words))
    # 执行搜索
    results=[]
    with ix.searcher() as searcher:
        hits = searcher.search(query)
        for hit in hits:
            results.append([hit["id"],hit.score])
    max_score=max(results,key=lambda s:s[1])[1]
    results=[[id,score/max_score] for id,score in results]
    return results


# 图像向量
img_index = faiss.read_index("img_index.faiss")
with open("img_path_dict.json") as f:
    img_path_dict=json.load(f)

from transformers import CLIPProcessor, CLIPModel
import torch
clip_model = CLIPModel.from_pretrained("clip-vit-base-patch32").to("cuda:0")
clip_processor = CLIPProcessor.from_pretrained("clip-vit-base-patch32")

def extract_text_vector(text):
    #文字部分
    inputs = clip_processor(text=[text],  return_tensors="pt", padding=True).to("cuda:0")
    with torch.no_grad():
        text_model=clip_model.text_model
        text_vector= text_model(**inputs).pooler_output[0]
        text_vector=clip_model.text_projection(text_vector).cpu()

    vector_norm = float(sum([v**2  for v in text_vector])**0.5) 
    text_vector= [(v/vector_norm).cpu().numpy().astype('float32')  for v in text_vector]
    return text_vector

# def extract_text_vector(text):
#     inputs = clip_processor(text=text, return_tensors="pt", padding=True, truncation=True).to("cuda:0")
#     with torch.no_grad():
#         text_features = clip_model.get_text_features(
#             input_ids=inputs["input_ids"],
#             attention_mask=inputs["attention_mask"]
#         )
#         text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
#     return text_features.cpu().numpy().astype('float32')


def clip_search(query):
    vector=extract_text_vector(query)
    distance,ids=img_index.search(np.array([vector]), 20)
    #print (distance)
    result=[[img_path_dict[str(s)],float(1-0.5*(d**2))] for s,d in zip(ids[0],distance[0])]
    return result

# 文字向量
text_index = faiss.read_index("text_index.faiss")
with open("text_path_dict.json") as f:
    text_path_dict=json.load(f)

from sentence_transformers import SentenceTransformer
qwen3_model = SentenceTransformer("/root/autodl-tmp/Qwen3-Embedding-8B")

def extract_qwen3_embedding(query):
    query_embeddings = qwen3_model.encode([query], prompt_name="query",convert_to_numpy=True)[0]
    query_embeddings = query_embeddings.astype("float32")
    vector_norm = float(sum([value**2 for value in query_embeddings]) ** 0.5)
    normaled_embedding = [float(value)/vector_norm for value in query_embeddings]
    
    return normaled_embedding

# def extract_qwen3_embedding(query):
#     # 模型输出已经是归一化向量，无需手动处理
#     embedding = qwen3_model.encode(
#         [query],
#         prompt_name="query",
#         convert_to_numpy=True,
#         normalize_embeddings=False  # 可选：False 表示信任模型输出；True 会再归一化一次
#     )[0]
#     return embedding.astype('float32')  # 确保 float32


def text_search(query):
    vector=extract_qwen3_embedding(query)
    distance,ids=text_index.search(np.array([vector]), 20)

    result=[[text_path_dict[str(s)],float(1-0.5*(d**2))] for s,d in zip(ids[0],distance[0])]
    return result

import feature_qwenvl_describe
import cal_score
def qa(query):

    #多路召回
    item_list1=tfidf_search(query)
    item_list2=clip_search(query)
    item_list3=text_search(query)

    #多路merge
    item_score={}
    for s,score in item_list1+item_list2+item_list3:
        item_score[s]=item_score.get(s,0)+score
    item_score=sorted(item_score.items(),key=lambda s:s[1],reverse=True)[0:3]

    result_score=[]
    for item,score in item_score:
        #获取音频对应的文字
        audio_text="" #id_document_map[item]

        prompt=f"背景信息{audio_text}问题{query} 如果不图片和问题不相关，直接输出不相关，不要输出无关内容"
        #item召回的视频
        #qwen2.5 VL 做问答
        result=feature_qwenvl_describe.extract_video_description(item,prompt)
        #qwen2.5 reranker
        score=cal_score.reranker(query,result)
        result_score.append([result,item,score])
        if score>=0.9:
            break

    answer,item,score=sorted(result_score,key=lambda s:s[-1],reverse=True)[0]

    return answer+"\n"+item

if __name__ == "__main__":
    print(qa("谁的飞机被打了"))
