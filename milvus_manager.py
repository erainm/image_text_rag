# Created by erainm on 2025/11/26 11:36.
# IDE：PyCharm 
# @Project: image_text_rag
# @File：milvus_manager
# @Description:

import uuid
from pymilvus import (
    connections, FieldSchema, CollectionSchema,
    DataType, Collection, utility
)
from config import MILVUS_CONFIG
import logging

logger = logging.getLogger(__name__)


class MilvusManager:
    def __init__(self):
        self.host = MILVUS_CONFIG["host"]
        self.port = MILVUS_CONFIG["port"]
        self.collection_name = MILVUS_CONFIG["collection_name"]
        self.dim = 768  # BGE模型的向量维度

    def connect(self):
        """连接Milvus"""
        try:
            connections.connect("default", host=self.host, port=self.port)
            logger.info("成功连接到Milvus")
        except Exception as e:
            logger.error(f"连接Milvus失败: {e}")
            raise

    def create_collection(self):
        """创建集合"""
        # 检查集合是否已存在
        if utility.has_collection(self.collection_name):
            logger.info(f"集合 {self.collection_name} 已存在")
            collection = Collection(self.collection_name)
            collection.load()
            return collection

        # 定义字段
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=4000),
            FieldSchema(name="content_type", dtype=DataType.VARCHAR, max_length=20),
            FieldSchema(name="image_url", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="page", dtype=DataType.INT64),
            FieldSchema(name="metadata", dtype=DataType.JSON)
        ]

        schema = CollectionSchema(fields, "Multimodal document collection")
        collection = Collection(self.collection_name, schema)

        # 创建索引
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 128}
        }
        collection.create_index("embedding", index_params)
        collection.load()

        logger.info(f"集合 {self.collection_name} 创建成功")
        return collection

    def get_entity_count(self):
        """Get the number of entities in the collection"""
        try:
            if utility.has_collection(self.collection_name):
                collection = Collection(self.collection_name)
                return collection.num_entities
            return 0
        except Exception as e:
            logger.error(f"Failed to get entity count: {e}")
            return 0

    def validate_data_consistency(self, data_list):
        """验证数据一致性"""
        if not data_list:
            return True, "数据为空"

        expected_count = len(data_list)

        # 检查每个字段的数据长度
        embeddings = [item['embedding'] for item in data_list]
        contents = [item['content'] for item in data_list]
        content_types = [item['content_type'] for item in data_list]
        image_urls = [item.get('image_url', '') for item in data_list]
        sources = [item['source'] for item in data_list]
        pages = [item.get('page', 0) for item in data_list]
        metadatas = [item.get('metadata', {}) for item in data_list]

        # 检查长度一致性
        fields = {
            'embeddings': len(embeddings),
            'contents': len(contents),
            'content_types': len(content_types),
            'image_urls': len(image_urls),
            'sources': len(sources),
            'pages': len(pages),
            'metadatas': len(metadatas)
        }

        # 检查所有字段长度是否一致
        for field_name, length in fields.items():
            if length != expected_count:
                return False, f"字段 {field_name} 长度不一致: 期望 {expected_count}, 实际 {length}"

        # 检查embedding维度
        for i, embedding in enumerate(embeddings):
            if len(embedding) != self.dim:
                return False, f"第 {i} 个embedding维度错误: 期望 {self.dim}, 实际 {len(embedding)}"

        return True, "数据验证通过"

    def insert_data(self, data_list):
        """插入数据到Milvus"""
        if not data_list:
            logger.warning("没有数据可插入")
            return

        # 验证数据一致性
        is_valid, message = self.validate_data_consistency(data_list)
        if not is_valid:
            logger.error(f"数据验证失败: {message}")
            raise ValueError(f"数据不一致: {message}")

        logger.info(f"开始插入 {len(data_list)} 条数据")

        collection = self.create_collection()

        try:
            # 准备数据 - 确保所有字段长度一致
            ids = [str(uuid.uuid4()) for _ in data_list]
            embeddings = [item['embedding'] for item in data_list]
            contents = [item['content'] for item in data_list]
            content_types = [item['content_type'] for item in data_list]
            image_urls = [item.get('image_url', '') for item in data_list]
            sources = [item['source'] for item in data_list]
            pages = [item.get('page', 0) for item in data_list]
            metadatas = [item.get('metadata', {}) for item in data_list]

            # 再次验证长度
            field_lengths = {
                'ids': len(ids),
                'embeddings': len(embeddings),
                'contents': len(contents),
                'content_types': len(content_types),
                'image_urls': len(image_urls),
                'sources': len(sources),
                'pages': len(pages),
                'metadatas': len(metadatas)
            }

            logger.info(f"各字段数据长度: {field_lengths}")

            # 插入数据
            entities = [
                ids,
                embeddings,
                contents,
                content_types,
                image_urls,
                sources,
                pages,
                metadatas
            ]

            collection.upsert(entities)
            collection.flush()

            logger.info(f"成功插入 {len(data_list)} 条数据到Milvus")

        except Exception as e:
            logger.error(f"插入数据失败: {e}")
            raise

    # 在 MilvusManager 类中添加以下方法
    def query_images(self, limit=50):
        """Query images from Milvus database"""
        try:
            if not utility.has_collection(self.collection_name):
                return []

            collection = Collection(self.collection_name)
            collection.load()

            # Query for images only
            expr = "content_type == 'image'"
            results = collection.query(
                expr=expr,
                output_fields=["content", "image_url", "source", "page"],
                limit=limit
            )

            # Fix image URLs to match Flask routes
            for result in results:
                if result.get('image_url'):
                    # Ensure image URLs use the correct API endpoint
                    image_url = result['image_url']
                    if image_url.startswith('/images/') or image_url.startswith('/static/images/'):
                        # Convert to API endpoint for reliability
                        filename = image_url.split('/')[-1]
                        result['image_url'] = f"/api/images/{filename}"

            return results
        except Exception as e:
            logger.error(f"Query images failed: {e}")
            return []

    def query_by_source(self, source: str):
        """根据来源查询数据"""
        try:
            collection = self.create_collection()
            results = collection.query(
                expr=f'source == "{source}"',
                output_fields=["content", "content_type", "image_url", "source", "page", "metadata"],
                limit=1000
            )
            return results if results else []
        except Exception as e:
            logger.error(f"按来源查询失败: {e}")
            return []

    def search(self, query_embedding, top_k=5):
        """向量搜索"""
        collection = Collection(self.collection_name)
        collection.load()

        search_params = {
            "metric_type": "L2",
            "params": {"nlist": 128}
        }

        results = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["content", "content_type", "image_url", "source", "page", "metadata"]
        )

        retrieved_items = []
        for hits in results:
            for hit in hits:
                item = {
                    'content': hit.entity.get('content'),
                    'content_type': hit.entity.get('content_type'),
                    'image_url': hit.entity.get('image_url'),
                    'source': hit.entity.get('source'),
                    'page': hit.entity.get('page'),
                    'score': hit.score,
                    'metadata': hit.entity.get('metadata')
                }
                # 确保图片URL正确
                if item['content_type'] == 'image' and item['image_url']:
                    image_url = item['image_url']
                    if image_url.startswith('/images/') or image_url.startswith('/static/images/'):
                        filename = image_url.split('/')[-1]
                        item['image_url'] = f"/api/images/{filename}"
                
                retrieved_items.append(item)

        logger.info(f"向量搜索完成，返回 {len(retrieved_items)} 条结果")
        return retrieved_items

    def get_entity_count(self):
        """获取实体数量"""
        try:
            collection = Collection(self.collection_name)
            if hasattr(collection, 'num_entities'):
                return collection.num_entities
            else:
                # 通过查询获取数量
                result = collection.query(expr="id != ''", limit=1000)
                return len(result) if result else 0
        except Exception as e:
            logger.error(f"获取实体数量失败: {e}")
            return 0

    def clear_collection(self):
        """清空集合"""
        if utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)
            logger.info(f"集合 {self.collection_name} 已清空")


# 全局Milvus管理器实例
milvus_manager = MilvusManager()