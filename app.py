# Created by erainm on 2025/11/25 11:29.
# IDE：PyCharm 
# @Project: image_text_rag
# @File：app
# @Description:主应用
from PIL.ImagePath import Path
from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import logging
from logging.handlers import RotatingFileHandler
from model_manager import model_manager
from document_parser import DocumentParser
from text_processor import TextProcessor
from vector_manager import VectorManager
from llm_generator import LLMGenerator
from memory_monitor import MemoryMonitor, memory_protected
from config import SYSTEM_CONFIG, IMAGES_DIR, UPLOADS_DIR
from werkzeug.utils import secure_filename


# 配置日志
def setup_logging():
    """配置日志系统"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            RotatingFileHandler('app.log', maxBytes=10*1024*1024, backupCount=5),
            logging.StreamHandler()
        ]
    )

setup_logging()
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = SYSTEM_CONFIG['max_file_size']

# 初始化组件
vector_manager = VectorManager()
llm_generator = LLMGenerator()
document_parser = DocumentParser()
text_processor = TextProcessor()
memory_monitor = MemoryMonitor()

# 启动内存监控
memory_monitor.start_monitoring(interval=30)


@app.route('/')
def index():
    """主页"""
    return render_template('index.html')


@app.route('/static/images/<path:filename>')
def serve_images(filename):
    """Serve static image files"""
    try:
        # Security check
        if '..' in filename or filename.startswith('/'):
            return "Invalid filename", 400

        image_path = IMAGES_DIR / filename
        if not image_path.exists():
            logger.error(f"Image file not found: {image_path}")
            return "Image not found", 404

        # Set correct MIME type
        mime_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.bmp': 'image/bmp',
            '.webp': 'image/webp'
        }

        ext = os.path.splitext(filename)[1].lower()
        mimetype = mime_types.get(ext, 'image/jpeg')

        response = send_from_directory(str(IMAGES_DIR), filename, mimetype=mimetype)
        response.headers['Cache-Control'] = 'public, max-age=3600'
        return response

    except Exception as e:
        logger.error(f"Failed to serve image {filename}: {e}")
        return "Image service error", 500


@app.route('/api/images/<path:filename>')
def serve_images_api(filename):
    """API endpoint to serve images"""
    try:
        # 更严格的安全检查
        safe_filename = secure_filename(filename)
        if safe_filename != filename:
            logger.warning(f"文件名不安全: {filename}")
            return jsonify({'error': 'Invalid filename'}), 400

        if '..' in filename or filename.startswith('/'):
            return jsonify({'error': 'Invalid filename'}), 400

        image_path = IMAGES_DIR / safe_filename
        if not image_path.exists():
            logger.error(f"API image not found: {image_path}")
            return jsonify({'error': 'Image not found'}), 404

        # 设置正确的MIME类型
        mime_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.bmp': 'image/bmp',
            '.webp': 'image/webp'
        }

        ext = os.path.splitext(safe_filename)[1].lower()
        mimetype = mime_types.get(ext, 'image/jpeg')

        response = send_from_directory(str(IMAGES_DIR), safe_filename, mimetype=mimetype)
        response.headers['Cache-Control'] = 'public, max-age=3600'
        return response
    except Exception as e:
        logger.error(f"API failed to serve image {filename}: {e}")
        return jsonify({'error': 'Image not found'}), 404


@app.route('/api/debug/retrieval/<query>')
def debug_retrieval(query):
    """调试检索结果"""
    try:
        # 使用增强的混合检索
        from multimodal_retriever import multimodal_retriever
        text_results, image_results = multimodal_retriever.enhanced_hybrid_search(query, top_k=10)

        formatted_results = []
        for result in text_results:
            formatted_results.append({
                'id': id(result),
                'content': result.get('content', '')[:200],  # 只显示前200字符
                'content_type': 'text',
                'source': result.get('source'),
                'page': result.get('page'),
                'score': result.get('score'),
                'relevance': result.get('relevance'),
                'has_image': False
            })
            
        for result in image_results:
            formatted_results.append({
                'id': id(result),
                'content': result.get('description', '')[:200],  # 只显示前200字符
                'content_type': 'image',
                'source': result.get('source'),
                'page': result.get('page'),
                'score': result.get('score'),
                'relevance': result.get('score'),  # 图片使用score作为relevance
                'image_url': result.get('url', ''),
                'has_image': True
            })

        return jsonify({
            'query': query,
            'results': formatted_results,
            'text_count': len(text_results),
            'image_count': len(image_results),
            'total_results': len(formatted_results)
        })
    except Exception as e:
        logger.error(f"调试检索失败: {e}")
        return jsonify({'error': str(e)}), 500

# 添加图片检查接口
@app.route('/api/verify_image/<filename>')
def verify_image(filename):
    """验证图片是否存在并可访问"""
    try:
        image_path = IMAGES_DIR / filename
        exists = image_path.exists() and image_path.is_file()

        if exists:
            size = image_path.stat().st_size
            url = f"/api/images/{filename}"
            return jsonify({
                'exists': True,
                'filename': filename,
                'size': size,
                'url': url,
                'full_url': f"http://localhost:5002{url}"
            })
        else:
            return jsonify({
                'exists': False,
                'filename': filename,
                'error': '文件不存在'
            }), 404

    except Exception as e:
        return jsonify({
            'exists': False,
            'filename': filename,
            'error': str(e)
        }), 500


@app.route('/api/verify_stored_images')
def verify_stored_images():
    """Verify that all stored images can be accessed"""
    try:
        images = milvus_manager.query_images(limit=100)
        verified_images = []

        for img in images:
            image_url = img.get('image_url', '')
            if image_url:
                # Extract filename from URL
                filename = image_url.split('/')[-1]
                image_path = IMAGES_DIR / filename

                verified_images.append({
                    'content': img.get('content', ''),
                    'image_url': image_url,
                    'filename': filename,
                    'file_exists': image_path.exists(),
                    'accessible': check_image_accessibility(image_url)
                })

        return jsonify({
            'total': len(verified_images),
            'accessible': len([img for img in verified_images if img['accessible']]),
            'images': verified_images
        })
    except Exception as e:
        logger.error(f"Image verification failed: {e}")
        return jsonify({'error': str(e)}), 500


def check_image_accessibility(image_url):
    """Check if an image URL is accessible"""
    try:
        from urllib.request import urlopen
        from urllib.error import URLError

        full_url = f"http://localhost:{SYSTEM_CONFIG['port']}{image_url}"
        response = urlopen(full_url, timeout=5)
        return response.getcode() == 200
    except:
        return False

@app.route('/api/check_image/<path:filename>')
def check_image(filename):
    """检查图片是否存在"""
    image_path = IMAGES_DIR / filename
    exists = image_path.exists()
    logger.info(f"检查图片 {filename}: {'存在' if exists else '不存在'}")
    return jsonify({
        'filename': filename,
        'exists': exists,
        'path': str(image_path)
    })


@app.route('/api/system_status')
def system_status():
    """系统状态检查"""
    mem_info = memory_monitor.get_memory_info()
    models_loaded = list(model_manager.loaded_models.keys())

    return jsonify({
        'status': 'running',
        'memory': {
            'total_gb': round(mem_info['total'] / 1024 ** 3, 1),
            'used_gb': round(mem_info['used'] / 1024 ** 3, 1),
            'available_gb': round(mem_info['available'] / 1024 ** 3, 1),
            'percent': mem_info['percent']
        },
        'models_loaded': models_loaded,
        'using_local_models': True
    })


@app.route('/api/debug/image_display')
def debug_image_display():
    """调试图片显示"""
    try:
        from milvus_manager import milvus_manager

        images = milvus_manager.query_images(limit=10)
        image_info = []

        for i, img in enumerate(images):
            image_url = img.get('image_url') or img.get('url', '')
            full_url = f"http://localhost:5002{image_url}" if image_url.startswith('/') else image_url

            image_info.append({
                'index': i + 1,
                'description': img.get('content', '')[:50],
                'url': image_url,
                'full_url': full_url,
                'source': img.get('source', ''),
                'page': img.get('page', 0),
                'file_exists': check_image_file_exists(image_url)
            })

        return jsonify({
            'total_images': len(images),
            'images': image_info
        })

    except Exception as e:
        logger.error(f"调试图片显示失败: {e}")
        return jsonify({'error': str(e)}), 500


def check_image_file_exists(image_url):
    """检查图片文件是否存在"""
    if not image_url or not image_url.startswith('/api/images/'):
        return False

    filename = image_url.split('/')[-1]
    image_path = IMAGES_DIR / filename
    return image_path.exists()


@app.route('/api/upload', methods=['POST'])
@memory_protected
def upload_document():
    """上传文档API"""
    try:
        # 检查请求中是否有文件
        if 'file' not in request.files:
            logger.error("请求中没有文件")
            return jsonify({'error': '没有文件'}), 400

        file = request.files['file']

        # 检查文件名
        if file.filename == '':
            logger.error("文件名为空")
            return jsonify({'error': '未选择文件'}), 400

        # 检查文件扩展名
        if not file.filename.lower().endswith('.pdf'):
            logger.error(f"不支持的文件类型: {file.filename}")
            return jsonify({'error': '仅支持PDF文件'}), 400

        # 确保上传目录存在
        UPLOADS_DIR.mkdir(exist_ok=True)

        # 生成安全的文件名
        from werkzeug.utils import secure_filename
        filename = secure_filename(file.filename)
        file_path = UPLOADS_DIR / filename

        logger.info(f"开始处理文件上传: {filename}")

        # 保存文件
        file.save(file_path)
        logger.info(f"文件保存成功: {file_path}")

        # 处理文档
        process_document(str(file_path))

        # 清理上传的文件
        try:
            os.remove(file_path)
            logger.info(f"临时文件已清理: {file_path}")
        except Exception as e:
            logger.warning(f"清理临时文件失败: {e}")

        return jsonify({'success': '文档处理成功，系统已学习文档内容'})

    except Exception as e:
        logger.error(f"文档上传处理失败: {e}")
        # 确保清理临时文件
        try:
            if 'file_path' in locals():
                os.remove(file_path)
        except:
            pass
        return jsonify({'error': f'处理失败: {str(e)}'}), 500

from milvus_manager import milvus_manager


def process_document(file_path):
    """处理文档内容并存储到向量数据库"""
    try:
        logger.info(f"开始处理文档: {file_path}")

        # 解析文档
        parsed_data = document_parser.parse_pdf(file_path)
        logger.info(f"文档解析完成: {len(parsed_data['text_chunks'])} 文本块, {len(parsed_data['images'])} 图片")

        # 处理文本
        text_chunks = text_processor.split_text(parsed_data['text_chunks'])
        logger.info(f"文本处理完成: {len(text_chunks)} 个文本片段")

        # 连接Milvus
        milvus_manager.connect()

        # 处理文本数据并生成向量
        text_data_list = []
        if text_chunks:
            # 批量生成文本向量
            text_contents = [chunk['content'] for chunk in text_chunks]
            logger.info(f"开始生成 {len(text_contents)} 个文本向量")

            text_embeddings = vector_manager.get_text_embeddings_batch(text_contents)
            logger.info(f"文本向量生成完成: {len(text_embeddings)} 个向量")

            for i, chunk in enumerate(text_chunks):
                # 确保embedding是列表格式
                embedding_list = text_embeddings[i].tolist() if hasattr(text_embeddings[i], 'tolist') else \
                text_embeddings[i]

                text_data_list.append({
                    'embedding': embedding_list,
                    'content': chunk['content'][:3900],  # 限制长度
                    'content_type': 'text',
                    'image_url': '',
                    'source': chunk['source'],
                    'page': chunk.get('page', 0),
                    'metadata': {
                        'type': 'text',
                        'chunk_id': i,
                        'length': len(chunk['content'])
                    }
                })

        # 处理图片数据
        image_data_list = []
        if parsed_data['images']:
            logger.info(f"开始处理 {len(parsed_data['images'])} 张图片")
            for i, image_info in enumerate(parsed_data['images']):
                try:
                    # 生成图片描述
                    caption = vector_manager.generate_image_caption(image_info['image_path'])
                    logger.info(f"图片 {i + 1}/{len(parsed_data['images'])} 描述: {caption}")

                    # 生成图片向量
                    image_embedding = vector_manager.get_image_embedding(image_info['image_path'])

                    # 确保embedding是列表格式且维度正确
                    embedding_list = image_embedding.tolist() if hasattr(image_embedding, 'tolist') else image_embedding

                    # 如果维度不匹配，进行调整
                    if len(embedding_list) != milvus_manager.dim:
                        logger.warning(f"图片向量维度不匹配: {len(embedding_list)}，进行调整")
                        if len(embedding_list) > milvus_manager.dim:
                            embedding_list = embedding_list[:milvus_manager.dim]
                        else:
                            embedding_list.extend([0.0] * (milvus_manager.dim - len(embedding_list)))

                    image_data_list.append({
                        'embedding': embedding_list,
                        'content': caption[:3900],
                        'content_type': 'image',
                        'image_url': f"/api/images/{os.path.basename(image_info['image_path'])}",  # Use API endpoint
                        'source': image_info['source'],
                        'page': image_info.get('page', 0),
                        'metadata': {
                            'type': 'image',
                            'original_path': image_info['image_path'],
                            'embedding_dim': len(embedding_list)
                        }
                    })

                except Exception as e:
                    logger.warning(f"处理图片失败 {image_info['image_path']}: {e}")
                    continue

        # 合并所有数据
        all_data = text_data_list + image_data_list
        logger.info(
            f"数据准备完成: {len(text_data_list)} 文本, {len(image_data_list)} 图片, 总计 {len(all_data)} 条数据")

        if all_data:
            # 插入数据前进行最终验证
            logger.info("开始插入数据到Milvus...")
            milvus_manager.insert_data(all_data)
            logger.info(f"数据存储完成: {len(all_data)} 条数据")
        else:
            logger.warning("没有有效数据可存储")

        logger.info("文档处理流程完成")

    except Exception as e:
        logger.error(f"文档处理失败: {e}")
        raise


@app.route('/api/ask', methods=['POST'])
@memory_protected
def ask_question():
    """提问API - 基于RAG的智能问答"""
    data = request.get_json()
    if not data:
        return jsonify({'error': '无效的JSON数据'}), 400

    question = data.get('question', '').strip()
    if not question:
        return jsonify({'error': '问题不能为空'}), 400

    if len(question) > 500:
        return jsonify({'error': '问题过长，请简化问题'}), 400

    try:
        # 使用RAG生成答案
        answer = llm_generator.generate_rag_answer(question)

        logger.info(f"问题回答完成: {question[:50]}...")
        return jsonify({'answer': answer})

    except Exception as e:
        logger.error(f"回答问题失败: {e}")
        return jsonify({'error': f'回答失败: {str(e)}'}), 500


@app.route('/api/test_llm', methods=['POST'])
def test_llm():
    """测试LLM API"""
    try:
        test_prompt = "请用一句话介绍你自己。"
        response = llm_generator.generate_response(test_prompt)

        return jsonify({
            'status': 'success',
            'response': response,
            'model': 'Qwen1.5-1.8B-Chat (本地)'
        })

    except Exception as e:
        logger.error(f"LLM测试失败: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': '接口不存在'}), 404


@app.errorhandler(500)
def internal_error(error):
    logger.error(f"服务器内部错误: {error}")
    return jsonify({'error': '服务器内部错误'}), 500


@app.route('/api/database_status')
def database_status():
    """数据库状态检查"""
    try:
        milvus_manager.connect()

        # 检查集合是否存在
        from pymilvus import utility
        collection_exists = utility.has_collection(milvus_manager.collection_name)

        if not collection_exists:
            return jsonify({
                'status': 'empty',
                'collection_name': milvus_manager.collection_name,
                'message': '集合不存在，请先上传文档'
            })

        # 获取实体数量
        entity_count = milvus_manager.get_entity_count()

        return jsonify({
            'status': 'connected',
            'collection_name': milvus_manager.collection_name,
            'entity_count': entity_count,
            'message': f'集合中有 {entity_count} 条数据'
        })

    except Exception as e:
        logger.error(f"数据库状态检查失败: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500


@app.route('/api/reset_database', methods=['POST'])
def reset_database():
    """重置数据库（清空并重新创建）"""
    try:
        # 清空现有集合
        milvus_manager.clear_collection()

        # 重新创建集合
        milvus_manager.create_collection()

        return jsonify({
            'success': '数据库重置成功',
            'message': '集合已清空并重新创建，请重新上传文档'
        })

    except Exception as e:
        logger.error(f"重置数据库失败: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/check_milvus')
def check_milvus():
    """检查Milvus连接和版本"""
    try:
        milvus_manager.connect()

        # 获取Milvus版本信息
        from pymilvus import __version__ as milvus_version
        from pymilvus import utility

        # 检查所有集合
        collections = utility.list_collections()

        return jsonify({
            'status': 'connected',
            'milvus_version': milvus_version,
            'collections': collections,
            'current_collection': milvus_manager.collection_name,
            'collection_exists': utility.has_collection(milvus_manager.collection_name)
        })

    except Exception as e:
        logger.error(f"Milvus检查失败: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500


def initialize_system():
    """初始化系统"""
    logger.info("=" * 50)
    logger.info("多模态RAG系统启动")
    logger.info("=" * 50)

    # 预加载模型
    try:
        logger.info("预加载模型中...")
        model_manager.load_llm_model()
        model_manager.load_text_embedding_model()
        logger.info("模型预加载完成")

    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        raise

    # 打印内存状态
    memory_monitor.print_memory_status()


@app.route('/api/clear_database', methods=['POST'])
def clear_database():
    """清空数据库"""
    try:
        # 连接Milvus
        milvus_manager.connect()

        # 清空集合
        milvus_manager.clear_collection()

        # 重新创建空集合
        milvus_manager.create_collection()

        logger.info("数据库清空完成")
        return jsonify({
            'success': True,
            'message': '数据库已清空，所有文档数据已被清除'
        })

    except Exception as e:
        logger.error(f"清空数据库失败: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/database_info')
def database_info():
    """获取数据库信息"""
    try:
        from milvus_manager import milvus_manager
        from pymilvus import utility

        milvus_manager.connect()

        # 检查集合是否存在
        collection_exists = utility.has_collection(milvus_manager.collection_name)

        info = {
            'collection_name': milvus_manager.collection_name,
            'collection_exists': collection_exists,
            'entity_count': 0
        }

        if collection_exists:
            info['entity_count'] = milvus_manager.get_entity_count()

        return jsonify(info)

    except Exception as e:
        logger.error(f"获取数据库信息失败: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/debug/upload', methods=['POST'])
def debug_upload():
    """调试上传功能"""
    try:
        logger.info("调试上传请求接收")
        logger.info(f"请求方法: {request.method}")
        logger.info(f"请求头: {dict(request.headers)}")
        logger.info(f"请求文件: {request.files}")

        if 'file' not in request.files:
            return jsonify({'error': '没有文件', 'files': list(request.files.keys())}), 400

        file = request.files['file']
        logger.info(f"文件信息: 文件名={file.filename}, 大小={len(file.read()) if file else 0}")

        # 重置文件指针
        file.seek(0)

        return jsonify({
            'success': '调试信息',
            'filename': file.filename,
            'content_type': file.content_type,
            'content_length': len(file.read())
        })

    except Exception as e:
        logger.error(f"调试上传失败: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/decheck')
def debug_check():
    """系统调试检查"""
    return jsonify({
        'upload_dir_exists': UPLOADS_DIR.exists(),
        'upload_dir': str(UPLOADS_DIR),
        'flask_env': os.environ.get('FLASK_ENV', 'production'),
        'max_content_length': app.config.get('MAX_CONTENT_LENGTH')
    })


@app.route('/api/debug/images')
def debug_images():
    """调试图片数据"""
    try:
        images = milvus_manager.query_images(limit=50)

        # 统计信息
        stats = {
            'total_images': len(images),
            'sources': {},
            'sample_descriptions': []
        }

        for img in images[:5]:  # 只显示前5个作为样本
            stats['sample_descriptions'].append({
                'description': img.get('content', ''),
                'source': img.get('source', ''),
                'page': img.get('page', 0),
                'url': img.get('image_url', '')
            })

            # 统计来源
            source = img.get('source', 'unknown')
            if source in stats['sources']:
                stats['sources'][source] += 1
            else:
                stats['sources'][source] = 1

        return jsonify({
            'status': 'success',
            'stats': stats,
            'all_images': images  # 返回所有图片数据用于调试
        })

    except Exception as e:
        logger.error(f"调试图片数据失败: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/debug/simple_search')
def debug_simple_search():
    """简单的搜索调试接口"""
    try:
        from multimodal_retriever import multimodal_retriever
        query = request.args.get('query', '测试')
        
        # 使用不同的搜索方法
        text_results1, image_results1 = multimodal_retriever.hybrid_search(query, top_k=5)
        text_results2, image_results2 = multimodal_retriever.enhanced_hybrid_search(query, top_k=5)
        
        return jsonify({
            'query': query,
            'hybrid_search': {
                'texts': text_results1,
                'images': image_results1,
                'text_count': len(text_results1),
                'image_count': len(image_results1)
            },
            'enhanced_search': {
                'texts': text_results2,
                'images': image_results2,
                'text_count': len(text_results2),
                'image_count': len(image_results2)
            }
        })
    except Exception as e:
        logger.error(f"简单搜索调试失败: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/debug/search_test')
def debug_search_test():
    """调试搜索功能"""
    query = request.args.get('query', '')
    if not query:
        return jsonify({'error': '需要提供查询参数'}), 400

    try:
        from multimodal_retriever import multimodal_retriever

        # 测试不同检索方法
        text_results1, image_results1 = multimodal_retriever.hybrid_search(query, top_k=5)
        text_results2, image_results2 = multimodal_retriever.enhanced_hybrid_search(query, top_k=5)

        return jsonify({
            'query': query,
            'hybrid_search': {
                'text_results': text_results1,
                'image_results': image_results1
            },
            'enhanced_search': {
                'text_results': text_results2,
                'image_results': image_results2
            }
        })

    except Exception as e:
        logger.error(f"搜索测试失败: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/debug_data')
def debug_data():
    """调试数据格式"""
    sample_data = {
        "text_example": {
            "embedding": [0.1] * 768,  # 768维向量
            "content": "这是一个示例文本内容",
            "content_type": "text",
            "image_url": "",
            "source": "example.pdf",
            "page": 1,
            "metadata": {"type": "text"}
        },
        "image_example": {
            "embedding": [0.1] * 512,  # 512维向量（CLIP）
            "content": "这是一张示例图片的描述",
            "content_type": "image",
            "image_url": "/static/images/example.jpg",
            "source": "example.pdf",
            "page": 1,
            "metadata": {"type": "image"}
        }
    }

    return jsonify(sample_data)


@app.route('/api/test_insert', methods=['POST'])
def test_insert():
    """测试插入少量数据"""
    try:
        # 创建测试数据
        test_data = [
            {
                'embedding': [0.1] * 768,
                'content': '这是一个测试文本片段',
                'content_type': 'text',
                'image_url': '',
                'source': 'test.pdf',
                'page': 1,
                'metadata': {'type': 'test'}
            }
        ]

        milvus_manager.insert_data(test_data)
        return jsonify({'success': '测试数据插入成功'})

    except Exception as e:
        logger.error(f"测试插入失败: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/test_image_display')
def test_image_display():
    """测试图片显示"""
    # 检查images目录
    image_files = list(IMAGES_DIR.glob('*'))
    image_info = []

    for img_path in image_files[:5]:  # 只显示前5个
        if img_path.is_file():
            image_info.append({
                'filename': img_path.name,
                'size': img_path.stat().st_size,
                'url': f"/api/images/{img_path.name}",
                'exists': True
            })

    return jsonify({
        'images_dir': str(IMAGES_DIR),
        'image_count': len(image_files),
        'sample_images': image_info
    })


@app.route('/api/check_image_urls')
def check_image_urls():
    """检查所有图片URL"""
    try:
        images = milvus_manager.query_images(limit=20)
        checked_images = []

        for img in images:
            image_url = img.get('image_url', '')
            filename = image_url.split('/')[-1] if image_url else 'unknown'
            image_path = IMAGES_DIR / filename

            checked_images.append({
                'description': img.get('content', ''),
                'url': image_url,
                'filename': filename,
                'file_exists': image_path.exists(),
                'source': img.get('source', ''),
                'page': img.get('page', 0)
            })

        return jsonify({
            'total_checked': len(checked_images),
            'images': checked_images
        })

    except Exception as e:
        logger.error(f"检查图片URL失败: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/debug/test_image_serving')
def test_image_serving():
    """测试图片服务功能"""
    try:
        # 获取images目录中的图片文件
        from pathlib import Path
        from config import IMAGES_DIR
        
        image_files = list(Path(IMAGES_DIR).glob('*'))
        if not image_files:
            return jsonify({'error': 'images目录中没有图片文件'}), 404
            
        # 选择第一个图片文件进行测试
        test_image = image_files[0]
        filename = test_image.name
        
        # 检查文件是否存在
        image_path = IMAGES_DIR / filename
        exists = image_path.exists()
        
        # 构造测试URL
        test_url = f"/api/images/{filename}"
        full_url = f"http://localhost:{SYSTEM_CONFIG['port']}{test_url}"
        
        return jsonify({
            'filename': filename,
            'exists': exists,
            'file_path': str(image_path),
            'test_url': test_url,
            'full_url': full_url,
            'file_size': test_image.stat().st_size if exists else 0
        })
        
    except Exception as e:
        logger.error(f"测试图片服务失败: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/debug/text_items')
def debug_text_items():
    """调试文本项显示"""
    try:
        from milvus_manager import milvus_manager

        # 查询文本项
        collection = milvus_manager.create_collection()
        expr = "content_type == 'text'"
        results = collection.query(
            expr=expr,
            output_fields=["content", "source", "page"],
            limit=10
        )

        return jsonify({
            'total_text_items': len(results),
            'sample_texts': results[:5] if results else []
        })
    except Exception as e:
        logger.error(f"调试文本项显示失败: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/debug/all_content')
def debug_all_content():
    """调试显示所有内容"""
    try:
        from milvus_manager import milvus_manager
        
        # 查询所有内容
        collection = milvus_manager.create_collection()
        
        # 分别查询文本和图片
        text_expr = "content_type == 'text'"
        image_expr = "content_type == 'image'"
        
        text_results = collection.query(
            expr=text_expr,
            output_fields=["content", "source", "page", "content_type"],
            limit=20
        )
        
        image_results = collection.query(
            expr=image_expr,
            output_fields=["content", "image_url", "source", "page", "content_type"],
            limit=20
        )
        
        return jsonify({
            'text_count': len(text_results),
            'image_count': len(image_results),
            'texts': text_results[:10],  # 只返回前10个避免过大
            'images': image_results[:10]  # 只返回前10个避免过大
        })
    except Exception as e:
        logger.error(f"调试所有内容失败: {e}")
        return jsonify({'error': str(e)}), 500

    @app.route('/api/set_answer_mode', methods=['POST'])
    def set_answer_mode():
        """设置答案生成模式"""
        data = request.get_json()
        mode = data.get('mode', 'strict')

        valid_modes = ['strict', 'extractive', 'balanced']
        if mode not in valid_modes:
            return jsonify({'error': f'无效模式，可选: {valid_modes}'}), 400

        llm_generator.set_answer_mode(mode)

        return jsonify({
            'success': f'答案生成模式已设置为: {mode}',
            'current_mode': mode,
            'description': {
                'strict': '严格模式：强制基于原文，禁止推理',
                'extractive': '抽取模式：直接组合原文片段',
                'balanced': '平衡模式：基于原文但允许适当解释'
            }[mode]
        })

    @app.route('/api/get_answer_mode')
    def get_answer_mode():
        """获取当前答案生成模式"""
        return jsonify({
            'current_mode': llm_generator.answer_mode,
            'available_modes': ['strict', 'extractive', 'balanced']
        })


if __name__ == '__main__':
    initialize_system()

    # 使用固定端口启动
    app.run(
        host=SYSTEM_CONFIG['host'],
        port=SYSTEM_CONFIG['port'],
        debug=SYSTEM_CONFIG['debug'],
        threaded=False
    )
