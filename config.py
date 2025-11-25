# Created by erainm on 2025/11/25 11:09.
# IDE：PyCharm
# @Project: image_text_rag
# @File：config
# @Description: 配置文件

import os
from pathlib import Path

# 基础路径配置
BASE_DIR = Path(__file__).parent
DOCUMENTS_DIR = BASE_DIR / "documents"
IMAGES_DIR = BASE_DIR / "images"
UPLOADS_DIR = BASE_DIR / "uploads"
MODELS_DIR = BASE_DIR / "models"
MODEL_CACHE_DIR = BASE_DIR / "model_cache"

# 创建目录
for directory in [DOCUMENTS_DIR, IMAGES_DIR, UPLOADS_DIR, MODELS_DIR, MODEL_CACHE_DIR]:
    directory.mkdir(exist_ok=True)

# 模型配置 - 使用本地模型
MODEL_CONFIG = {
    "text_embedding": {
        "name": "BAAI/bge-base-zh-v1.5",
        "local_path": str(MODELS_DIR / "bge-base-zh-v1.5"),
        "type": "sentence-transformers",
        "dim": 768,
        "use_local": True
    },
    "multimodal": {
        "name": "openai/clip-vit-base-patch16",
        "type": "clip",
        "dim": 512,
        "use_local": False  # 未下载，使用在线
    },
    "image_caption": {
        "name": "Salesforce/blip-image-captioning-base",
        "type": "blip",
        "use_local": False  # 未下载，使用在线
    },
    "llm": {
        "name": "Qwen/Qwen1.5-1.8B-Chat",
        "local_path": str(MODELS_DIR / "Qwen1.5-1.8B-Chat"),
        "type": "qwen",
        "context_length": 32768,
        "use_local": True
    }
}

# 系统配置
SYSTEM_CONFIG = {
    "host": "localhost",
    "port": 6001,
    "debug": False,
    "max_file_size": 100 * 1024 * 1024,  # 100MB
    "allowed_extensions": ['.pdf']
}

# Milvus配置
MILVUS_CONFIG = {
    "host": "localhost",
    "port": "19530",
    "collection_name": "multimodal_docs"
}

# 内存优化配置
MEMORY_CONFIG = {
    "max_workers": 2,
    "batch_size": {
        "text_embedding": 4,
        "image_processing": 1,
    },
    "max_content_length": 1000000
}

# 环境变量配置
os.environ['TRANSFORMERS_CACHE'] = str(MODEL_CACHE_DIR)
os.environ['HF_HOME'] = str(MODEL_CACHE_DIR)

# 文本处理配置
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50