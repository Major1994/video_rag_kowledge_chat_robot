# Requires transformers>=4.51.0
# Requires sentence-transformers>=2.7.0

from sentence_transformers import SentenceTransformer

# Load the model
model = SentenceTransformer("/root/autodl-tmp/Qwen3-Embedding-8B")

def extract_qwen_embedding(query):
    query_embeddings = model.encode([query], prompt_name="query")[0]

    vector_norm = float(sum([value**2 for value in query_embeddings]) ** 0.5)
    normaled_embedding = [float(value)/vector_norm for value in query_embeddings]
    
    return normaled_embedding

if __name__ == "__main__":
    text = "我是大大的脑袋"
    embedding= extract_qwen_embedding(text)
    print(embedding)