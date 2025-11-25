# Created by erainm on 2025/11/25 11:25.
# IDE：PyCharm 
# @Project: image_text_rag
# @File：document_parser
# @Description:文档解析

import fitz  # PyMuPDF
import uuid
import os
from PIL import Image
import io
from config import IMAGES_DIR
import logging

logger = logging.getLogger(__name__)


class DocumentParser:
    def __init__(self):
        self.images_dir = IMAGES_DIR
        # 确保图片目录存在
        self.images_dir.mkdir(exist_ok=True)

    def parse_pdf(self, file_path):
        """解析PDF文档 - 确保图片URL正确"""
        logger.info(f"开始解析PDF: {file_path}")
        doc = fitz.open(file_path)
        results = {'text_chunks': [], 'images': []}

        try:
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)

                # 提取文本
                text = page.get_text()
                if text.strip():
                    results['text_chunks'].append({
                        'content': text,
                        'page': page_num + 1,
                        'type': 'text',
                        'source': os.path.basename(file_path)
                    })

                # 提取图片
                image_list = page.get_images()
                for img_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]

                        # 检查图片大小
                        if len(image_bytes) < 1024:
                            continue

                        # 保存图片
                        image_ext = base_image["ext"]
                        image_id = f"{uuid.uuid4()}.{image_ext}"
                        image_path = os.path.join(self.images_dir, image_id)

                        with open(image_path, "wb") as f:
                            f.write(image_bytes)

                        # 生成图片描述
                        try:
                            from vector_manager import vector_manager
                            caption = vector_manager.generate_image_caption(image_path)
                        except Exception as e:
                            logger.warning(f"生成图片描述失败: {e}")
                            caption = f"文档第{page_num + 1}页的图片"

                        # 确保URL格式正确 - 使用绝对路径
                        image_url = f"/api/images/{image_id}"

                        results['images'].append({
                            'image_path': image_path,
                            'image_url': image_url,  # 确保这个字段存在且正确
                            'url': image_url,  # 同时设置url字段用于检索
                            'page': page_num + 1,
                            'type': 'image',
                            'source': os.path.basename(file_path),
                            'filename': image_id,
                            'description': caption,
                            'size': len(image_bytes)
                        })

                        logger.info(f"提取图片: {image_url} - {caption}")

                    except Exception as e:
                        logger.warning(f"提取图片失败: {e}")
                        continue

            logger.info(f"PDF解析完成: {len(results['text_chunks'])} 文本块, {len(results['images'])} 图片")
            return results

        finally:
            doc.close()