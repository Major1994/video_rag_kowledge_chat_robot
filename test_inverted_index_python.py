# inverted_index_with_stopwords.py

import jieba
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
# 2. 停用词表（可扩展）
# ================================
# 常见中文停用词
stop_words = {
    '的', '了', '和', '都', '是', '在', '有', '这', '那', '就',
    '也', '很', '太', '又', '不', '但', '只', '都', '要', '会',
    '可以', '能够', '应该', '一个', '一些', '这种', '那种',
    '我们', '你们', '他们', '她', '他', '它', '我', '你',
    '着', '过', '得', '地', '很', '吗', '呢', '吧', '啊', '，'
}

# 可选：从文件加载停用词
# with open("stopwords.txt", encoding="utf-8") as f:
#     stop_words = set(line.strip() for line in f if line.strip())

# ================================
# 3. 构建倒排索引（带停用词过滤）
# ================================
inverted_index = defaultdict(set)

print("🔍 分词详情（过滤后）：")
for doc_id, content in documents.items():
    # 使用 jieba.cut_for_search：更细粒度分词
    words = list(jieba.cut_for_search(content))
    filtered_words = []
    for word in words:
        word = word.strip()
        if len(word) == 0 or word in stop_words:
            continue  # 跳过空串和停用词
        filtered_words.append(word)
        inverted_index[word].add(doc_id)
    
    print(f"{doc_id}: {filtered_words}")

print("\n📊 倒排索引结构（已过滤停用词）：")
for word, doc_ids in sorted(inverted_index.items()):
    print(f"  '{word}' → {sorted(doc_ids)}")

# ================================
# 4. 查询函数（查询时也过滤停用词）
# ================================
def search_and(query_text, index, stop_words_set=None):
    """AND 查询：返回同时包含所有词的文档（过滤停用词）"""
    if stop_words_set is None:
        stop_words_set = set()
    words = list(jieba.cut_for_search(query_text))
    result_docs = None
    valid_words = []  # 记录实际用于查询的词
    for word in words:
        word = word.strip()
        if not word or word in stop_words_set:
            continue
        valid_words.append(word)
        docs = index.get(word, set())
        if result_docs is None:
            result_docs = docs
        else:
            result_docs = result_docs & docs
    print(f"🔍 AND 查询词: {valid_words}")
    return result_docs or set()

def search_or(query_text, index, stop_words_set=None):
    """OR 查询：返回包含任意一个词的文档"""
    if stop_words_set is None:
        stop_words_set = set()
    words = list(jieba.cut_for_search(query_text))
    result_docs = set()
    valid_words = []
    for word in words:
        word = word.strip()
        if not word or word in stop_words_set:
            continue
        valid_words.append(word)
        result_docs |= index.get(word, set())
    print(f"🔍 OR 查询词: {valid_words}")
    return result_docs

# ================================
# 5. 测试查询
# ================================
print("\n🔍 查询测试：")

# AND 查询
res_and = search_and("猫 爱", inverted_index, stop_words)
print(f"同时包含 '猫' 和 '爱' 的文档: {sorted(res_and)}")

# OR 查询
res_or = search_or("猫 爱", inverted_index, stop_words)
print(f"包含 '猫' 或 '爱' 的文档: {sorted(res_or)}")

# 复杂查询（停用词自动过滤）
res_hard = search_and("可爱的 动物", inverted_index, stop_words)
print(f"同时包含 '可爱' 和 '动物' 的文档: {sorted(res_hard)}")