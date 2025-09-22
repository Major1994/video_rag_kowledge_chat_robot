from sentence_transformers import SentenceTransformer
import json
import pickle

# Load the model
model = SentenceTransformer("/root/autodl-tmp/Qwen3-Embedding-8B")

def extract_qwen_embedding(query):
    query_embeddings = model.encode([query], prompt_name="query")[0]

    vector_norm = float(sum([value**2 for value in query_embeddings]) ** 0.5)
    normaled_embedding = [float(value)/vector_norm for value in query_embeddings]
    
    return normaled_embedding

if __name__ == "__main__":
    result = {}
    output_file= "video_description_embedding.pickle"
    with open("/root/autodl-tmp/video_description.jsonl") as outf:
        for line in outf:
            line = json.loads(line)
            for k,v in line.items():
                text_path = k
                text_description = v
            text_description_embedding = extract_qwen_embedding(text_description)
            result[text_path] = text_description_embedding

    with open(output_file, "wb") as outf:
        pickle.dump(result, outf)
            
            
