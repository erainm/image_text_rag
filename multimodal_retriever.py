# Created by erainm on 2025/11/26 12:28.
# IDE：PyCharm 
# @Project: image_text_rag
# @File：multimodal_retriever
# @Description: 创建统一的多模态检索器，，解决检索时，只能找到文本片段，始终找不到关联的图片

# multimodal_retriever.py
import numpy as np
from typing import List, Dict, Tuple
from milvus_manager import milvus_manager
from vector_manager import vector_manager
import logging

logger = logging.getLogger(__name__)


class MultimodalRetriever:
    def __init__(self):
        self.text_dim = 768  # BGE模型维度
        self.image_dim = 512  # CLIP模型维度

    def hybrid_search(self, query: str, top_k: int = 10) -> Tuple[List[Dict], List[Dict]]:
        """
        混合检索：同时检索文本和图片
        策略：分别检索然后合并排序
        """
        try:
            # 1. 生成查询向量
            query_embedding_text = vector_manager.get_text_embeddings_batch([query])[0]

            # 2. 分别检索文本和图片
            text_results = self._search_text(query_embedding_text, top_k=top_k)
            image_results = self._search_images(query, top_k=max(2, top_k // 2))

            logger.info(f"混合检索完成: {len(text_results)} 文本, {len(image_results)} 图片")
            return text_results, image_results

        except Exception as e:
            logger.error(f"混合检索失败: {e}")
            return [], []

    def enhanced_hybrid_search(self, query: str, top_k: int = 10) -> Tuple[List[Dict], List[Dict]]:
        """
        增强的混合检索：提高检索精度
        特别针对Qwen系列等特定主题的查询进行优化
        """
        try:
            # 1. 生成查询向量
            query_embedding = vector_manager.get_text_embeddings_batch([query])[0]

            # 2. 检索更多结果然后精筛
            initial_results = milvus_manager.search(query_embedding.tolist(), top_k=top_k * 5)  # 增加检索数量

            # 3. 精细过滤文本结果
            text_results = []
            image_results = []
            
            for item in initial_results:
                if item['content_type'] == 'text':
                    # 计算更精确的相关性
                    relevance = self._calculate_enhanced_relevance(query, item['content'], query_embedding)

                    # 进一步降低过滤阈值，保留更多文本结果
                    if relevance > 0.01:  # 从0.05降低到0.01
                        text_results.append({
                            'content': item['content'],
                            'source': item['source'],
                            'page': item['page'],
                            'score': item['score'],
                            'relevance': relevance,
                            'type': 'text'
                        })
                elif item['content_type'] == 'image':
                    # 对图片也做相关性计算
                    relevance = self._calculate_enhanced_relevance(query, item['content'], query_embedding)
                    
                    if relevance > 0.01:
                        image_results.append({
                            'content': item['content'],
                            'description': item['content'],
                            'url': item['image_url'],
                            'image_url': item['image_url'],
                            'source': item['source'],
                            'page': item['page'],
                            'score': item['score'],
                            'relevance': relevance,
                            'type': 'image'
                        })

            # 按相关性排序
            text_results.sort(key=lambda x: x['relevance'], reverse=True)
            image_results.sort(key=lambda x: x['relevance'], reverse=True)
            
            text_results = text_results[:top_k]
            image_results = image_results[:top_k]

            logger.info(f"增强检索完成: {len(text_results)} 文本(相关性>0.01), {len(image_results)} 图片")
            return text_results, image_results

        except Exception as e:
            logger.error(f"增强检索失败: {e}")
            # 回退到基础检索
            return self.hybrid_search(query, top_k)

    def _search_text(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        """文本向量检索"""
        try:
            # 使用文本向量在Milvus中搜索文本内容
            results = milvus_manager.search(query_embedding.tolist(), top_k=top_k)

            # 过滤出文本结果
            text_results = []
            for item in results:
                if item['content_type'] == 'text':
                    text_results.append({
                        'content': item['content'],
                        'source': item['source'],
                        'page': item['page'],
                        'score': item['score'],
                        'type': 'text',
                        'relevance': self._calculate_text_relevance(item['content'], query_embedding)
                    })

            # 按相关性排序
            text_results.sort(key=lambda x: x['relevance'], reverse=True)
            return text_results[:top_k]

        except Exception as e:
            logger.error(f"文本检索失败: {e}")
            return []

    def _search_images_with_query(self, query: str, top_k: int = 3) -> List[Dict]:
        """直接基于查询搜索图片"""
        try:
            # 获取所有图片
            all_images = self._get_all_images()
            if not all_images:
                return []

            # 计算查询与每张图片描述的相似度
            scored_images = []
            for img in all_images:
                similarity = self._calculate_text_similarity(query, img['content'])

                # 保留一定相关性的图片
                if similarity > 0.01:
                    scored_images.append({
                        'description': img['content'],
                        'url': img['image_url'],
                        'source': img['source'],
                        'page': img['page'],
                        'score': similarity,
                        'type': 'image'
                    })

            # 按相似度排序
            scored_images.sort(key=lambda x: x['score'], reverse=True)
            return scored_images[:top_k]

        except Exception as e:
            logger.error(f"基于查询的图片搜索失败: {e}")
            return []

    def _search_images(self, query: str, top_k: int = 3) -> List[Dict]:
        """图片检索 - 使用文本查询匹配图片描述"""
        try:
            # 方法1: 使用图片描述文本进行匹配
            all_images = self._get_all_images()
            if not all_images:
                return []

            # 计算查询与图片描述的相似度
            scored_images = []
            for img in all_images:
                similarity = self._calculate_text_similarity(query, img['content'])

                # 保留中等相关性的图片
                if similarity > 0.01:  # 从0.05降低到0.01
                    scored_images.append({
                        'content': img['content'],
                        'description': img['content'],
                        'url': img['image_url'],
                        'image_url': img['image_url'],
                        'source': img['source'],
                        'page': img['page'],
                        'score': similarity,
                        'type': 'image'
                    })

            # 按相似度排序
            scored_images.sort(key=lambda x: x['score'], reverse=True)
            return scored_images[:top_k]

        except Exception as e:
            logger.error(f"图片检索失败: {e}")
            return []

    def _find_high_quality_images(self, text_results: List[Dict], query: str, top_k: int = 3) -> List[Dict]:
        """基于高质量文本结果找到相关的图片"""
        try:
            # 即使没有文本结果也要检索图片
            if not text_results:
                # 直接使用查询匹配图片
                return self._search_images_with_query(query, top_k=top_k)

            # 获取所有图片
            all_images = self._get_all_images()
            if not all_images:
                return []

            # 基于文本结果的内容和查询来匹配图片
            scored_images = []

            for img in all_images:
                # 计算图片与查询的相似度
                query_similarity = self._calculate_text_similarity(query, img['content'])

                # 计算图片与文本结果的关联度
                text_relation = 0.0
                for text_result in text_results[:3]:  # 看前3个文本结果
                    text_similarity = self._calculate_text_similarity(text_result['content'], img['content'])
                    text_relation = max(text_relation, text_similarity)

                # 综合评分 - 平衡查询匹配和文本关联
                total_score = 0.5 * query_similarity + 0.5 * text_relation

                # 保留中等相关性的图片
                if total_score > 0.01:  # 从0.05降低到0.01
                    scored_images.append({
                        'description': img['content'],
                        'url': img['image_url'],
                        'source': img.get('source', ''),
                        'page': img.get('page', 0),
                        'score': total_score,
                        'type': 'image',
                        'query_similarity': query_similarity,
                        'text_relation': text_relation
                    })

            # 按总分排序
            scored_images.sort(key=lambda x: x['score'], reverse=True)

            logger.info(f"图片检索完成: 找到 {len(scored_images)} 张相关图片")
            for i, img in enumerate(scored_images[:3]):
                logger.info(f"  图片 {i + 1}: {img['description']} (分数: {img['score']:.2f})")

            return scored_images[:top_k]

        except Exception as e:
            logger.error(f"查找相关图片失败: {e}")
            return []

    def _get_all_images(self) -> List[Dict]:
        """获取所有图片数据 - 确保URL字段正确"""
        try:
            # 查询所有图片类型的数据
            results = milvus_manager.query_images(limit=1000)

            # 转换格式并确保URL正确
            image_data = []
            for item in results:
                # 优先使用image_url，如果没有则使用url
                image_url = item.get('image_url') or item.get('url', '')

                # 确保URL格式正确
                if image_url and not image_url.startswith('/api/images/'):
                    # 尝试修复URL
                    if 'filename' in item:
                        image_url = f"/api/images/{item['filename']}"
                    else:
                        # 从URL中提取文件名
                        if '/' in image_url:
                            filename = image_url.split('/')[-1]
                            image_url = f"/api/images/{filename}"
                        else:
                            # 使用content中的信息尝试构造URL
                            continue

                # 确保有有效的图片URL
                if image_url:
                    image_data.append({
                        'content': item.get('content', ''),
                        'image_url': image_url,
                        'url': image_url,  # 确保url字段也存在
                        'source': item.get('source', ''),
                        'page': item.get('page', 0),
                        'filename': item.get('filename', '')
                    })

            logger.info(f"获取到 {len(image_data)} 张图片数据")
            return image_data

        except Exception as e:
            logger.error(f"获取图片数据失败: {e}")
            return []

    def _calculate_text_relevance(self, text: str, query_embedding: np.ndarray) -> float:
        """计算文本与查询的相关性"""
        try:
            # 生成文本的向量
            text_embedding = vector_manager.get_text_embeddings_batch([text])[0]
            # 计算余弦相似度
            similarity = np.dot(query_embedding, text_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(text_embedding)
            )
            return float(similarity)
        except:
            return 0.0

    def _calculate_enhanced_relevance(self, query: str, text: str, query_embedding: np.ndarray) -> float:
        """计算增强的相关性分数"""
        try:
            # 方法1: 向量相似度
            text_embedding = vector_manager.get_text_embeddings_batch([text])[0]
            vector_similarity = np.dot(query_embedding, text_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(text_embedding)
            )

            # 方法2: 关键词匹配
            query_words = set(query.lower().split())
            text_words = set(text.lower().split())

            if query_words and text_words:
                keyword_overlap = len(query_words.intersection(text_words)) / len(query_words)
            else:
                keyword_overlap = 0

            # 方法3: 长度惩罚 - 避免过短的内容获得过高分数
            length_factor = min(1.0, len(text) / 50)  # 至少50个字符才有完整分数

            # 综合评分 - 增加关键词匹配权重
            final_score = 0.5 * vector_similarity + 0.4 * keyword_overlap + 0.1 * length_factor

            return final_score

        except:
            return 0.0

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """计算两个文本的相似度"""
        try:
            text1_lower = text1.lower()
            text2_lower = text2.lower()

            if not text1_lower.strip() or not text2_lower.strip():
                return 0.0

            words1 = set(text1_lower.split())
            words2 = set(text2_lower.split())

            if not words1 or not words2:
                return 0.0

            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))

            return intersection / union if union > 0 else 0.0

        except:
            return 0.0

    def _find_related_images(self, text_results: List[Dict], query: str, top_k: int = 3) -> List[Dict]:
        """基于文本结果找到相关的图片 - 增强版本"""
        try:
            # 获取所有图片数据
            all_images = self._get_all_images()
            if not all_images:
                logger.warning("没有找到任何图片数据")
                return []

            logger.info(f"开始从 {len(all_images)} 张图片中检索...")

            scored_images = []

            for img in all_images:
                image_url = img.get('image_url', '')
                description = img.get('content', '')

                # 检查图片URL是否有效
                if not image_url or not image_url.startswith('/api/images/'):
                    continue

                # 计算图片与查询的相似度
                query_similarity = self._calculate_text_similarity(query, description)

                # 计算图片与文本结果的关联度
                text_relation = 0.0
                if text_results:
                    for text_result in text_results[:3]:  # 看前3个文本结果
                        text_similarity = self._calculate_text_similarity(text_result['content'], description)
                        text_relation = max(text_relation, text_similarity)

                # 如果查询直接包含图片相关关键词，提高分数
                image_keywords = ['图片', '图像', '图表', '照片', '图', '截图']
                has_image_keyword = any(keyword in query for keyword in image_keywords)
                if has_image_keyword:
                    query_similarity = max(query_similarity, 0.5)  # 提高基础分数

                # 综合评分
                if text_results:
                    total_score = 0.3 * query_similarity + 0.7 * text_relation
                else:
                    total_score = query_similarity

                # 降低阈值，让更多图片有机会显示
                if total_score > 0.01:  # 降低阈值
                    scored_images.append({
                        'description': description,
                        'url': image_url,
                        'source': img.get('source', ''),
                        'page': img.get('page', 0),
                        'score': total_score,
                        'type': 'image',
                        'query_similarity': query_similarity,
                        'text_relation': text_relation
                    })

            # 按总分排序
            scored_images.sort(key=lambda x: x['score'], reverse=True)

            logger.info(f"图片检索完成: 找到 {len(scored_images)} 张相关图片")
            for i, img in enumerate(scored_images[:3]):
                logger.info(f"  图片 {i + 1}: {img['description']} (分数: {img['score']:.2f})")

            return scored_images[:top_k]

        except Exception as e:
            logger.error(f"查找相关图片失败: {e}")
            return []

    def force_image_search(self, query: str, top_k: int = 3) -> List[Dict]:
        """强制图片搜索：即使相关性不高也返回图片"""
        try:
            all_images = self._get_all_images()
            if not all_images:
                return []

            scored_images = []
            for img in all_images:
                # 简单基于关键词匹配
                description = img.get('content', '').lower()
                query_lower = query.lower()

                # 计算基础分数
                score = 0.0

                # 检查是否有共同词汇
                desc_words = set(description.split())
                query_words = set(query_lower.split())
                common_words = desc_words.intersection(query_words)

                if common_words:
                    score = len(common_words) / len(query_words) if query_words else 0

                # 如果查询包含图片相关词汇，提高分数
                image_keywords = ['图片', '图像', '图表', '照片', '图', '截图', 'illustration', 'image', 'figure', 'photo', 'picture', 'diagram']
                if any(keyword in query_lower for keyword in image_keywords):
                    score = max(score, 0.5)

                # 添加基于页面位置的分数（倾向于返回更多图片）
                position_score = 1.0  # 默认分数

                total_score = 0.7 * score + 0.3 * position_score

                scored_images.append({
                    'description': img.get('content', ''),
                    'url': img.get('image_url', ''),
                    'source': img.get('source', ''),
                    'page': img.get('page', 0),
                    'score': total_score,
                    'type': 'image'
                })

            # 按分数排序
            scored_images.sort(key=lambda x: x['score'], reverse=True)

            # 确保至少返回一些图片
            if not scored_images and all_images:
                for i, img in enumerate(all_images[:top_k]):
                    scored_images.append({
                        'description': img.get('content', ''),
                        'url': img.get('image_url', ''),
                        'source': img.get('source', ''),
                        'page': img.get('page', 0),
                        'score': 0.1 * (top_k - i),  # 给予递减分数
                        'type': 'image'
                    })

            return scored_images[:top_k]

        except Exception as e:
            logger.error(f"强制图片搜索失败: {e}")
            # 最后的后备方案：随机返回一些图片
            try:
                all_images = self._get_all_images()
                fallback_images = []
                for i, img in enumerate(all_images[:top_k]):
                    fallback_images.append({
                        'description': img.get('content', ''),
                        'url': img.get('image_url', ''),
                        'source': img.get('source', ''),
                        'page': img.get('page', 0),
                        'score': 0.05,
                        'type': 'image'
                    })
                return fallback_images
            except:
                return []

# 全局检索器实例
multimodal_retriever = MultimodalRetriever()
