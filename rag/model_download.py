from huggingface_hub import snapshot_download

from rag.config import EMBEDDING_MODEL, RERANK_MODEL

# 指定模型ID和你想存放的本地路径

reranker_model_id = "BAAI/bge-reranker-large"
reranker_local_path = RERANK_MODEL

embedding_model_id = "BAAI/bge-base-zh-v1.5"
embedding_local_path = EMBEDDING_MODEL


# 执行下载
def download_reranker_model():
    print(f"正在下载模型 '{reranker_model_id}' 到 '{reranker_local_path}'...")
    snapshot_download(repo_id=reranker_model_id, local_dir=reranker_local_path, local_dir_use_symlinks=False)
    print("✅ 模型下载完成！")


def download_embedding_text():
    print(f"正在下载模型 '{embedding_model_id}' 到 '{embedding_local_path}'...")
    snapshot_download(repo_id=embedding_model_id, local_dir=embedding_local_path, local_dir_use_symlinks=False)
    print("✅ 模型下载完成！")


if __name__ == "__main__":
    download_reranker_model()
