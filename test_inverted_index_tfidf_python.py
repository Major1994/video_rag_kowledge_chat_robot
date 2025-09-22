# inverted_index_tfidf.py

import jieba
import math
from collections import defaultdict

# ================================
# 1. 示例文档数据
# ================================
documents = {
    "doc1": "猫喜欢玩耍，特别爱抓老鼠",
    "doc2": "狗喜欢跑步，非常忠诚",
    "doc3": "猫和狗都是可爱的宠物",
    "doc4": "大家都爱可爱的动物"
}

# ================================
# 2. 停用词表
# ================================
stop_words = {
    '的', '了', '和', '都', '是', '在', '有', '这', '那', '就',
    '也', '很', '太', '又', '不', '但', '只', '要', '会', '可以',
    '能够', '应该', '一个', '一些', '这种', '那种', '我们', '你们',
    '他们', '她', '他', '它', '我', '你', '着', '过', '得', '地', '吗', '呢', '吧', '啊'
}

# ================================
# 3. 构建倒排索引 + 词频统计（TF）
# ================================
inverted_index = defaultdict(set)
term_freq = defaultdict(dict)  # term_freq[doc_id][word] = 频次
doc_lengths = {}  # 每个文档的有效词数（用于归一化）

print("🔍 分词与词频统计（过滤停用词）：")
for doc_id, content in documents.items():
    words = list(jieba.cut_for_search(content))
    filtered_words = []
    word_count = defaultdict(int)
    
    for word in words:
        word = word.strip()
        if not word or word in stop_words:
            continue
        filtered_words.append(word)
        word_count[word] += 1
    
    # 保存词频
    for word, freq in word_count.items():
        inverted_index[word].add(doc_id)
        term_freq[doc_id][word] = freq
    
    doc_lengths[doc_id] = len(filtered_words)
    print(f"{doc_id}: {filtered_words} (共 {len(filtered_words)} 词)")

# 总文档数
N = len(documents)
print(f"\n📊 总文档数: {N}")

# ================================
# 4. 计算 IDF
# ================================
def compute_idf(word, index, total_docs):
    """计算 IDF = log(N / df)"""
    df = len(index.get(word, set()))  # 包含该词的文档数
    if df == 0:
        return 0
    return math.log(total_docs / df)

# 缓存 IDF 值
idf_values = {}

# ================================
# 5. TF-IDF 搜索与排序
# ================================
def search_tfidf(query_text, index, term_freq, doc_lengths, stop_words_set, documents_dict):
    """搜索并按 TF-IDF 相关性排序"""
    words = list(jieba.cut_for_search(query_text))
    query_words = [w.strip() for w in words if w.strip() and w not in stop_words_set]
    
    if not query_words:
        return []
    
    print(f"🔍 实际查询词: {query_words}")
    
    scores = defaultdict(float)
    
    for word in query_words:
        if word not in index:
            continue
        # 计算 IDF
        if word not in idf_values:
            idf_values[word] = compute_idf(word, index, N)
        idf = idf_values[word]
        
        # 对每个包含该词的文档计算 TF-IDF
        for doc_id in index[word]:
            tf = term_freq[doc_id].get(word, 0)
            # TF 归一化：tf / doc_length
            tf_norm = tf / doc_lengths[doc_id] if doc_lengths[doc_id] > 0 else 0
            # 累加 TF-IDF 分数（多个查询词）
            scores[doc_id] += tf_norm * idf
    
    # 按分数降序排序
    ranked_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    # 返回 (doc_id, score, content)
    return [(doc_id, score, documents_dict[doc_id]) for doc_id, score in ranked_results]

# ================================
# 6. 测试查询
# ================================
print("\n🔍 TF-IDF 搜索测试：")

# 示例1：搜索“猫 爱”
results = search_tfidf("猫 爱", inverted_index, term_freq, doc_lengths, stop_words, documents)
print("\n--- 搜索 '猫 爱' ---")
for i, (doc_id, score, content) in enumerate(results, 1):
    print(f"{i}. [{doc_id}] 分数: {score:.4f} | 内容: {content}")

# 示例2：搜索“可爱 动物”
results2 = search_tfidf("可爱 动物", inverted_index, term_freq, doc_lengths, stop_words, documents)
print("\n--- 搜索 '可爱 动物' ---")
for i, (doc_id, score, content) in enumerate(results2, 1):
    print(f"{i}. [{doc_id}] 分数: {score:.4f} | 内容: {content}")

# 示例3：单个词
results3 = search_tfidf("忠诚", inverted_index, term_freq, doc_lengths, stop_words, documents)
print("\n--- 搜索 '忠诚' ---")
for i, (doc_id, score, content) in enumerate(results3, 1):
    print(f"{i}. [{doc_id}] 分数: {score:.4f} | 内容: {content}")