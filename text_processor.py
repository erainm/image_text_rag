# Created by erainm on 2025/11/25 11:25.
# IDE：PyCharm 
# @Project: image_text_rag
# @File：text_processor
# @Description:文本处理

from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP


class TextProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", "、", " "]
        )

    def split_text(self, text_chunks):
        """分割文本"""
        all_splits = []

        for chunk in text_chunks:
            splits = self.text_splitter.split_text(chunk['content'])
            for split in splits:
                if len(split.strip()) > 10:  # 过滤太短的文本
                    new_chunk = chunk.copy()
                    new_chunk['content'] = split.strip()
                    all_splits.append(new_chunk)

        return all_splits