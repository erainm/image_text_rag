# Created by erainm on 2025/11/26 10:42.
# IDE：PyCharm 
# @Project: image_text_rag
# @File：llm_generator
# @Description: LLM生成器

from multimodal_retriever import multimodal_retriever
import logging
from vector_manager import vector_manager
from milvus_manager import milvus_manager

logger = logging.getLogger(__name__)


class LLMGenerator:
    def __init__(self):
        from config import MODEL_CONFIG
        model_path = MODEL_CONFIG["llm"]["local_path"]
        self.model_path = model_path
        self.answer_mode = "strict"  # strict, extractive, balanced

    def set_answer_mode(self, mode: str):
        """设置答案生成模式"""
        valid_modes = ["strict", "extractive", "balanced"]
        if mode in valid_modes:
            self.answer_mode = mode
            logger.info(f"答案生成模式设置为: {mode}")
        else:
            logger.warning(f"无效的模式: {mode}, 使用默认模式")

    def generate_rag_answer(self, query: str) -> str:
        """生成基于RAG的答案 - 确保图片正确显示"""
        try:
            # 1. 检索相关文档内容
            text_results, image_results = self.search_relevant_content(query, top_k=8)

            logger.info(f"检索结果: {len(text_results)} 文本, {len(image_results)} 图片")

            # 2. 如果没有找到文本内容，则尝试更宽松的检索
            if not text_results:
                logger.info("未找到高相关性文本，尝试宽松检索")
                text_results = self._loose_text_search(query, top_k=5)

            # 3. 如果有图片，优先确保图片URL正确
            for i, img in enumerate(image_results):
                # 确保图片URL格式正确
                if not img['url'].startswith('/api/images/'):
                    # 尝试修复URL格式
                    if 'image_url' in img and img['image_url']:
                        img['url'] = img['image_url']
                    else:
                        # 从描述或其他字段推断
                        logger.warning(f"图片 {i + 1} URL格式异常: {img['url']}")

                logger.info(f"图片 {i + 1} URL: {img['url']}")

            # 4. 构建提示词
            prompt = self._build_strict_rag_prompt(query, text_results, image_results)

            # 5. 生成答案
            response = self.generate_response(prompt, max_length=1024, temperature=0.1)

            # 6. 添加引用信息（包括文本和图片）
            logger.info(f"添加引用信息: {len(text_results)} 文本, {len(image_results)} 图片")
            response = self._add_references(response, text_results, image_results)
            
            # 7. 如果有相关图片，直接显示图片
            if image_results:
                logger.info(f"直接显示 {len(image_results)} 张图片")
                response = self._add_image_references(response, image_results)

            return response

        except Exception as e:
            logger.error(f"RAG答案生成失败: {e}")
            return f"抱歉，处理您的问题时出现错误: {str(e)}"

    def _loose_text_search(self, query: str, top_k: int = 5):
        """宽松的文本搜索"""
        try:
            # 生成查询向量
            query_embedding = vector_manager.get_text_embeddings_batch([query])[0]
            
            # 搜索更多结果
            results = milvus_manager.search(query_embedding.tolist(), top_k=top_k*2)
            
            # 过滤文本结果，使用更低的阈值
            text_results = []
            for item in results:
                if item['content_type'] == 'text':
                    # 计算相关性但使用更低的阈值
                    relevance = self._calculate_loose_relevance(query, item['content'])
                    
                    if relevance > 0.01:  # 非常低的阈值
                        text_results.append({
                            'content': item['content'],
                            'source': item['source'],
                            'page': item['page'],
                            'score': item['score'],
                            'relevance': relevance,
                            'type': 'text'
                        })

            # 按相关性排序
            text_results.sort(key=lambda x: x['relevance'], reverse=True)
            return text_results[:top_k]

        except Exception as e:
            logger.error(f"宽松文本搜索失败: {e}")
            return []

    def _calculate_loose_relevance(self, query: str, text: str) -> float:
        """计算宽松的相关性分数"""
        try:
            # 简单的关键词匹配
            query_words = set(query.lower().split())
            text_words = set(text.lower().split())
            
            if not query_words:
                return 0.0
                
            # 计算重叠比例
            overlap = len(query_words.intersection(text_words))
            return overlap / len(query_words)
            
        except:
            return 0.0

    def _generate_fallback_answer(self, query: str, text_results: list, image_results: list) -> str:
        """生成备选答案"""
        if not text_results and not image_results:
            return "在文档中没有找到与您问题相关的内容。"

        answer = "根据文档内容：\n\n"

        if text_results:
            answer += "**相关文本内容：**\n"
            for i, text in enumerate(text_results[:3]):
                answer += f"{i + 1}. {text['content']} (来自: {text['source']} 第{text['page']}页)\n\n"

        if image_results:
            answer += "**相关图片：**\n"
            for i, img in enumerate(image_results[:2]):
                answer += f"- {img['description']} (来自: {img['source']} 第{img['page']}页)\n"

        return answer

    def _generate_error_answer(self, query: str, error: str) -> str:
        """生成错误答案"""
        return f"抱歉，处理您的问题时遇到技术问题：{error}\n\n问题：{query}"

    def _post_process_response(self, response: str, text_results: list, query: str) -> str:
        """后处理：确保回答基于原文"""
        # 检查回答是否偏离原文
        if self._is_response_deviated(response, text_results):
            logger.warning("检测到回答偏离原文，进行修正")
            return self.generate_extractive_answer(query, text_results, [])

        return response

    def _is_response_deviated(self, response: str, text_results: list) -> bool:
        """检查回答是否偏离原文"""
        if not text_results:
            return False

        # 简单的检查：回答中是否包含原文的关键词
        original_keywords = set()
        for text in text_results[:3]:
            words = text['content'].lower().split()[:10]  # 取前10个词作为关键词
            original_keywords.update(words)

        response_words = set(response.lower().split())
        overlap = len(original_keywords.intersection(response_words))

        # 如果重叠度太低，可能偏离原文
        deviation_ratio = overlap / len(original_keywords) if original_keywords else 0
        return deviation_ratio < 0.2  # 如果重叠度低于20%，认为偏离

    def generate_response(self, prompt: str, max_length: int = 512, temperature: float = 0.7) -> str:
        """生成LLM响应"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            # 加载模型和tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                local_files_only=True
            )

            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                local_files_only=True,
                device_map="auto" if torch.cuda.is_available() else None
            )

            # 编码输入
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
                padding=True
            )

            # 移动到模型设备
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )

            # 解码响应
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # 移除输入部分
            if response.startswith(prompt):
                response = response[len(prompt):].strip()

            return response

        except Exception as e:
            logger.error(f"LLM生成失败: {e}")
            return f"抱歉，生成回答时出现错误: {str(e)}"

    def search_relevant_content(self, query: str, top_k: int = 8):
        """搜索相关的文档内容 - 增强图片检索"""
        try:
            # 使用增强的混合检索
            text_results, image_results = multimodal_retriever.enhanced_hybrid_search(query, top_k=top_k)

            # 如果文本结果太少，尝试宽松检索
            if len(text_results) < 2:
                logger.info("文本结果较少，使用宽松检索")
                loose_text_results = self._loose_text_search(query, top_k=5)
                # 合并结果，去重
                existing_contents = {text['content'][:50] for text in text_results}  # 使用前50个字符去重
                for text in loose_text_results:
                    if text['content'][:50] not in existing_contents:
                        text_results.append(text)
                        existing_contents.add(text['content'][:50])

            # 如果图片结果太少，使用强制图片搜索
            if len(image_results) < 2:
                logger.info("图片结果较少，使用强制图片搜索")
                force_images = multimodal_retriever.force_image_search(query, top_k=3)
                # 合并结果，去重
                existing_urls = {img['url'] for img in image_results}
                for img in force_images:
                    if img['url'] not in existing_urls and img['url']:  # 确保URL不为空
                        image_results.append(img)
                        existing_urls.add(img['url'])

            logger.info(f"最终检索结果: {len(text_results)} 文本, {len(image_results)} 图片")

            return text_results, image_results

        except Exception as e:
            logger.error(f"内容检索失败: {e}")
            # 最后的后备方案
            try:
                loose_text_results = self._loose_text_search(query, top_k=3)
                force_images = multimodal_retriever.force_image_search(query, top_k=3)
                logger.info(f"使用后备方案检索结果: {len(loose_text_results)} 文本, {len(force_images)} 图片")
                return loose_text_results, force_images
            except:
                return [], []

    def _build_strict_rag_prompt(self, query: str, text_results: list, image_results: list) -> str:
        """构建严格的RAG提示词 - 修复版本"""

        # 构建系统提示
        system_prompt = """你是一个专业的文档问答助手。请严格按照提供的文档内容回答问题。

重要要求：
1. **必须基于原文回答**：只使用提供的文档内容
2. **必须给出具体答案**：不能只说"文档中有相关信息"，要给出具体内容
3. **直接引用原文**：尽量使用原文的词句
4. **注明来源**：每个信息点都要注明来自哪个文档第几页
5. **如果信息不足**：明确说"根据提供的文档内容，无法完整回答这个问题，但相关内容包括：..."
6. **必须回答**：无论如何都要给出有意义的回答

禁止事项：
- 不要添加外部知识
- 不要进行创造性发挥
- 不要只说"文档中有相关信息"而不给出具体内容
"""

        prompt = f"{system_prompt}\n\n用户问题: {query}\n\n"

        # 添加文本内容
        if text_results:
            prompt += "**相关文档原文:**\n\n"
            for i, text in enumerate(text_results[:5]):
                source_info = f"{text['source']} 第{text['page']}页"
                relevance = f"(相关性: {text['relevance']:.3f})"

                prompt += f"【原文片段 {i + 1}】{relevance} - {source_info}\n"
                # 包含完整的内容，不只是截断的
                prompt += f"{text['content']}\n\n"
        else:
            prompt += "**文档原文:** 未找到相关文本内容。\n\n"
            # 即使没有找到文本，也添加一个通用说明
            prompt += "注意：系统未找到与您的问题高度相关的文本内容。\n\n"

        # 添加图片描述
        if image_results:
            prompt += "**相关图片信息:**\n\n"
            for i, img in enumerate(image_results[:5]):  # 增加到5张图片
                score = f"(匹配度: {img['score']:.3f})"
                prompt += f"【图片 {i + 1}】{img['description']} {score} 来自: {img['source']} 第{img['page']}页\n\n"

        prompt += f"""
**请基于以上原文内容直接回答这个问题: "{query}"**

你的回答必须包含：
1. 基于原文的具体答案
2. 引用具体的信息来源
3. 如果有图片，提及图片内容

如果没有找到相关文本内容，请明确说明这一点，并尝试基于图片信息回答问题。

现在请直接回答问题:
"""
        return prompt

    def generate_extractive_answer(self, query: str, text_results: list, image_results: list) -> str:
        """生成基于原文抽取的答案"""
        try:
            if not text_results:
                return "文档中没有找到与问题相关的信息。"

            # 直接组合最相关的文本片段
            relevant_contents = []
            for i, text in enumerate(text_results[:3]):
                if text['relevance'] > 0.4:  # 只使用相关性高的片段
                    source_info = f"（来自: {text['source']} 第{text['page']}页）"
                    relevant_contents.append(f"{text['content']} {source_info}")

            if not relevant_contents:
                return "文档中虽然有相关内容，但相关性较低，无法准确回答。"

            # 直接组合原文片段
            answer = "根据文档内容：\n\n"
            for i, content in enumerate(relevant_contents):
                answer += f"{i + 1}. {content}\n\n"

            # 添加图片信息
            if image_results:
                answer += "相关图片信息：\n"
                for img in image_results[:2]:
                    answer += f"- {img['description']}（{img['source']} 第{img['page']}页）\n"

            return answer

        except Exception as e:
            logger.error(f"抽取式答案生成失败: {e}")
            return self.generate_rag_answer(query)  # 回退到生成式方法

    def _add_references(self, response: str, text_results: list, image_results: list) -> str:
        """在回答中添加引用信息，包括文本来源和图片"""
        reference_section = "\n\n---\n**引用日期和图片**\n"
        
        # 添加文本引用
        if text_results:
            reference_section += "\n**相关文本内容:**\n"
            for i, text in enumerate(text_results[:5]):  # 显示最多5个文本引用
                source_info = f"{text['source']} 第{text['page']}页"
                relevance = f"(相关性: {text['relevance']:.2f})"
                reference_section += f"\n[{i+1}] {relevance} {source_info}\n"
                # 截取部分内容作为引用
                content_preview = text['content'][:200] + "..." if len(text['content']) > 200 else text['content']
                reference_section += f"    {content_preview}\n"
        
        # 添加图片引用
        if image_results:
            reference_section += "\n**相关图片:**\n"
            for i, img in enumerate(image_results[:5]):  # 显示最多5张图片
                source_info = f"{img['source']} 第{img['page']}页"
                score = f"(匹配度: {img['score']:.2f})"
                reference_section += f"\n[{i+1}] {score} {source_info}\n"
                reference_section += f"    描述: {img.get('description', '无描述')}\n"
        
        return response + reference_section

    def _add_image_references(self, response: str, image_results: list) -> str:
        """在回答中添加真实的图片显示 - 直接显示图片而不是描述"""
        if not image_results:
            return response

        logger.info(f"准备直接显示 {len(image_results)} 张图片")

        # 导入配置获取正确的端口号
        from config import SYSTEM_CONFIG
        port = SYSTEM_CONFIG.get('port', 6001)

        image_section = "\n\n---\n**🖼️ 相关图片**\n\n"

        displayed_count = 0
        for i, img in enumerate(image_results):
            if displayed_count >= 5:  # 增加到最多显示5张图片
                break

            image_url = img['url']
            description = img.get('description', '文档图片')
            source = img['source']
            page = img['page']

            # 构建完整的图片URL，使用正确的端口号
            if image_url.startswith('/api/images/'):
                full_url = f"http://localhost:{port}{image_url}"
            else:
                # 尝试从其他格式提取文件名
                filename = image_url.split('/')[-1] if '/' in image_url else image_url
                full_url = f"http://localhost:{port}/api/images/{filename}"

            logger.info(f"显示图片 {i + 1}: {full_url}")

            # 直接显示图片，使用更大的尺寸和更好的布局
            image_section += f"""
    <div style="
        border: 2px solid #3498db; 
        border-radius: 12px; 
        padding: 20px; 
        margin: 20px 0; 
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    ">
        <div style="text-align: center; margin-bottom: 15px;">
            <img 
                src="{full_url}" 
                alt="{description}"
                style="
                    max-width: 90%; 
                    max-height: 400px; 
                    height: auto; 
                    border-radius: 8px; 
                    border: 1px solid #bdc3c7;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.15);
                    transition: transform 0.3s ease;
                "
                onmouseover="this.style.transform='scale(1.02)'"
                onmouseout="this.style.transform='scale(1)'"
                onerror="
                    this.onerror=null; 
                    this.src='https://via.placeholder.com/600x400/95a5a6/ffffff?text=图片加载失败'; 
                    this.alt='图片加载失败，请检查图片URL';
                    console.error('图片加载失败:', this.src);
                "
                onload="console.log('图片加载成功:', this.src)"
            >
        </div>
        <div style="
            text-align: center; 
            font-size: 14px; 
            color: #2c3e50; 
            background: rgba(255,255,255,0.8); 
            padding: 10px; 
            border-radius: 6px;
            border-left: 4px solid #3498db;
        ">
            <div style="font-weight: bold; margin-bottom: 5px;">
                📸 图片 {i + 1} - {description}
            </div>
            <div style="font-size: 12px; color: #7f8c8d;">
                📁 来源: {source} | 📄 页码: {page}
            </div>
        </div>
    </div>
    """
            displayed_count += 1

        if displayed_count > 0:
            # 添加一些CSS样式确保图片显示正常
            image_section += """
    <style>
    @media (max-width: 768px) {
        .image-container img {
            max-width: 100% !important;
        }
    }
    </style>
    """

        return response + image_section
