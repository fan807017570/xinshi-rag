import os

MILVUS_HOST = os.environ.get("MILVUS_HOST", "localhost")
MILVUS_PORT = os.environ.get("MILVUS_PORT", "19530")
COLLECTION_NAME = os.environ.get("MILVUS_COLLECTION", "Xinshi_school")

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "/Users/anranfan/core-bank-dp/llm/models/bge-base-zh-v1.5")
EMBEDDING_MODEL_CACHE_DIR = os.environ.get("EMBEDDING_MODEL_CACHE_DIR", "./models")

RERANK_MODEL = os.environ.get("RERANK_MODEL", "/Users/anranfan/core-bank-dp/llm/models/bge-reranker-large")
RERANK_MODEL_CACHE_DIR = os.environ.get("RERANK_MODEL_CACHE_DIR", "models")
RERANK_BATCH_SIZE = max(1, int(os.environ.get("RERANK_BATCH_SIZE", "64")))
RERANK_MAX_LENGTH = max(64, int(os.environ.get("RERANK_MAX_LENGTH", "256")))
RERANK_BACKEND = os.environ.get("RERANK_BACKEND", "torch")
RERANK_CANDIDATE_LIMIT = max(1, int(os.environ.get("RERANK_CANDIDATE_LIMIT", "16")))
RERANK_QUERY_CHAR_LIMIT = max(16, int(os.environ.get("RERANK_QUERY_CHAR_LIMIT", "256")))
RERANK_DOC_CHAR_LIMIT = max(64, int(os.environ.get("RERANK_DOC_CHAR_LIMIT", "800")))
RERANK_SCORE_CACHE_SIZE = max(0, int(os.environ.get("RERANK_SCORE_CACHE_SIZE", "4096")))
ROLE_FILTER_MULTIPLIER = max(1, int(os.environ.get("ROLE_FILTER_MULTIPLIER", "3")))

TOP_K_RETRIEVE = 20
TOP_K_RERANK = 5

# 多轮对话传入 LLM 的历史条数上限（user/assistant 各算一条）
MAX_HISTORY_MESSAGES = 12
