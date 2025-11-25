# Created by erainm on 2025/11/25 11:26.
# IDE：PyCharm 
# @Project: image_text_rag
# @File：vector_manager
# @Description:向量化管理

import numpy as np
from PIL import Image
import torch
from typing import List
from model_manager import model_manager
from config import MEMORY_CONFIG
import logging

logger = logging.getLogger(__name__)


class VectorManager:
    def __init__(self):
        self.batch_size = MEMORY_CONFIG["batch_size"]

    def get_text_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """批量获取文本嵌入"""
        model = model_manager.load_text_embedding_model()

        all_embeddings = []
        for i in range(0, len(texts), self.batch_size["text_embedding"]):
            batch_texts = texts[i:i + self.batch_size["text_embedding"]]
            logger.debug(f"处理文本批次 {i // self.batch_size['text_embedding'] + 1}")

            with torch.no_grad():
                batch_embeddings = model.encode(
                    batch_texts,
                    batch_size=2,
                    show_progress_bar=False,
                    convert_to_tensor=False,
                    normalize_embeddings=True
                )

            all_embeddings.append(batch_embeddings)

            # 批次间清理内存
            if i % (self.batch_size["text_embedding"] * 4) == 0:
                model_manager.cleanup_memory()

        embeddings = np.vstack(all_embeddings)
        logger.info(f"文本向量化完成: {len(embeddings)} 个向量, 维度: {embeddings.shape}")
        return embeddings

    def get_image_embedding(self, image_path: str) -> np.ndarray:
        """获取图片向量"""
        model, processor = model_manager.load_multimodal_model()

        try:
            image = Image.open(image_path).convert('RGB')

            # 调整图片尺寸以减少内存占用
            max_size = 224
            if max(image.size) > max_size:
                image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

            inputs = processor(images=image, return_tensors="pt")

            with torch.no_grad():
                image_features = model.get_image_features(**inputs)

            embedding = image_features.cpu().numpy().flatten()
            embedding = embedding / np.linalg.norm(embedding)

            logger.debug(f"图片向量维度: {embedding.shape}")
            return embedding

        except Exception as e:
            logger.error(f"图片向量化失败 {image_path}: {e}")
            # 返回零向量作为fallback，但确保维度正确
            return np.zeros(512)

    def generate_image_caption(self, image_path: str) -> str:
        """生成图片描述"""
        model, processor = model_manager.load_image_caption_model()

        try:
            image = Image.open(image_path).convert('RGB')

            # 调整图片尺寸
            max_size = 384
            if max(image.size) > max_size:
                image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

            inputs = processor(images=image, return_tensors="pt")

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=30,
                    num_beams=3,
                    early_stopping=True
                )

            caption = processor.decode(outputs[0], skip_special_tokens=True)
            return caption

        except Exception as e:
            logger.error(f"图片描述生成失败 {image_path}: {e}")
            return "图片描述生成失败"

# 全局Milvus管理器实例
vector_manager = VectorManager()