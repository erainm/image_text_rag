# Created by erainm on 2025/11/25 11:24.
# IDE：PyCharm 
# @Project: image_text_rag
# @File：models
# @Description:模型管理

import torch
import psutil
import gc
from transformers import (
    AutoModel, AutoTokenizer,
    CLIPModel, CLIPProcessor,
    BlipForConditionalGeneration, BlipProcessor
)
from sentence_transformers import SentenceTransformer
from config import MODEL_CONFIG
import logging
# In model_manager.py
from transformers import Qwen2ForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


class ModelManager:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.loaded_models = {}
        logger.info(f"使用设备: {self.device}")

    def cleanup_memory(self):
        """清理内存"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.debug("内存清理完成")

    def load_text_embedding_model(self):
        """加载文本嵌入模型"""
        model_info = MODEL_CONFIG["text_embedding"]
        model_key = "text_embedding"

        if model_key in self.loaded_models:
            return self.loaded_models[model_key]

        try:
            if model_info.get("use_local", False):
                local_path = model_info["local_path"]
                logger.info(f"从本地加载文本嵌入模型: {local_path}")
                model = SentenceTransformer(local_path, device=self.device)
            else:
                logger.info(f"从网络加载文本嵌入模型: {model_info['name']}")
                model = SentenceTransformer(model_info["name"], device=self.device)

            self.loaded_models[model_key] = model
            logger.info("文本嵌入模型加载成功")
            return model

        except Exception as e:
            logger.error(f"文本嵌入模型加载失败: {e}")
            raise

    def load_multimodal_model(self):
        """加载多模态模型"""
        model_info = MODEL_CONFIG["multimodal"]
        model_key = "multimodal"

        if model_key in self.loaded_models:
            return self.loaded_models[model_key]

        try:
            logger.info(f"加载多模态模型: {model_info['name']}")
            model = CLIPModel.from_pretrained(
                model_info["name"],
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
            processor = CLIPProcessor.from_pretrained(model_info["name"])
            model = model.to(self.device)

            self.loaded_models[model_key] = (model, processor)
            logger.info("多模态模型加载成功")
            return model, processor

        except Exception as e:
            logger.error(f"多模态模型加载失败: {e}")
            raise

    def load_image_caption_model(self):
        """加载图片描述模型"""
        model_info = MODEL_CONFIG["image_caption"]
        model_key = "image_caption"

        if model_key in self.loaded_models:
            return self.loaded_models[model_key]

        try:
            logger.info(f"加载图片描述模型: {model_info['name']}")
            model = BlipForConditionalGeneration.from_pretrained(
                model_info["name"],
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
            processor = BlipProcessor.from_pretrained(model_info["name"])
            model = model.to(self.device)

            self.loaded_models[model_key] = (model, processor)
            logger.info("图片描述模型加载成功")
            return model, processor

        except Exception as e:
            logger.error(f"图片描述模型加载失败: {e}")
            raise

    def load_llm_model(self):
        """加载LLM模型"""
        model_info = MODEL_CONFIG["llm"]
        model_key = "llm"

        if model_key in self.loaded_models:
            return self.loaded_models[model_key]

        try:
            if model_info.get("use_local", False):
                local_path = model_info["local_path"]
                logger.info(f"从本地加载LLM模型: {local_path}")

                # 检查模型文件是否存在
                import os
                if not os.path.exists(local_path):
                    raise FileNotFoundError(f"模型路径不存在: {local_path}")

                # 首先尝试使用正确的Auto类
                from transformers import AutoModelForCausalLM, AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(
                    local_path,
                    trust_remote_code=True,
                    local_files_only=True
                )

                # 使用正确的模型类
                model = AutoModelForCausalLM.from_pretrained(
                    local_path,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                    local_files_only=True,
                    device_map="auto" if torch.cuda.is_available() else None
                )

                # 如果上面的方法失败，尝试使用Qwen特定的类
                if not hasattr(model, 'generate'):
                    try:
                        from transformers import Qwen2ForCausalLM
                        model = Qwen2ForCausalLM.from_pretrained(
                            local_path,
                            torch_dtype=torch.float32,
                            low_cpu_mem_usage=True,
                            trust_remote_code=True,
                            local_files_only=True,
                            device_map="auto" if torch.cuda.is_available() else None
                        )
                    except ImportError:
                        logger.warning("Qwen2ForCausalLM 不可用，使用基础模型")

            else:
                logger.info(f"从网络加载LLM模型: {model_info['name']}")
                from transformers import AutoModelForCausalLM, AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(
                    model_info["name"],
                    trust_remote_code=True
                )

                model = AutoModelForCausalLM.from_pretrained(
                    model_info["name"],
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                    device_map="auto" if torch.cuda.is_available() else None
                )

            # 确保模型有generate方法
            if not hasattr(model, 'generate'):
                raise AttributeError("加载的模型不支持生成功能，请检查模型类型")

            self.loaded_models[model_key] = (model, tokenizer)
            logger.info("LLM模型加载成功")
            return model, tokenizer

        except Exception as e:
            logger.error(f"LLM模型加载失败: {e}")
            raise


# 全局模型管理器实例
model_manager = ModelManager()