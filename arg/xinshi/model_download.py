from huggingface_hub import snapshot_download

# 指定模型ID和你想存放的本地路径
model_id = "BAAI/bge-reranker-large"
local_path = "./models/bge-reranker-large"  # 可以换成你喜欢的任何绝对或相对路径

# 执行下载
print(f"正在下载模型 '{model_id}' 到 '{local_path}'...")
snapshot_download(repo_id=model_id, local_dir=local_path, local_dir_use_symlinks=False)

print("✅ 模型下载完成！")