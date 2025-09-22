import os
import jieba
from whoosh.index import create_in
from whoosh.fields import Schema, ID, TEXT
from whoosh.analysis import Tokenizer, Token
from whoosh.qparser import QueryParser

# 自定义 Jieba 分词器
class JiebaTokenizer(Tokenizer):
    def __call__(self, value, positions=False, chars=False,
                 keeporiginal=False, removestops=True,
                 start_pos=0, start_char=0, mode='', **kwargs):
        t = Token(positions, chars, removestops=removestops, mode=mode)
        for pos, word in enumerate(jieba.lcut(value)):
            if positions:
                t.pos = start_pos + pos
            if chars:
                # 简单处理字符位置（可忽略）
                pass
            t.text = word
            yield t

# 创建 analyzer
jieba_analyzer = JiebaTokenizer()

# 定义 schema
schema = Schema(
    id=ID(stored=True),
    content=TEXT(stored=True, analyzer=jieba_analyzer)
)

# 创建索引目录
if not os.path.exists("indexdir"):
    os.mkdir("indexdir")

ix = create_in("indexdir", schema)
writer = ix.writer()

# 示例文档
documents = [
    ("doc1", "猫喜欢玩耍，特别爱抓老鼠"),
    ("doc2", "狗喜欢跑步，非常忠诚"),
    ("doc3", "猫和狗都是可爱的宠物"),
]

for doc_id, content in documents:
    writer.add_document(id=doc_id, content=content)

writer.commit()

# 搜索测试
with ix.searcher() as searcher:
    parser = QueryParser("content", ix.schema)
    query = parser.parse("猫 爱")
    results = searcher.search(query)
    for r in results:
        print(f"ID: {r['id']}, 内容: {r['content']}")