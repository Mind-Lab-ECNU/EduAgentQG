#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于FAISS的向量检索RAG系统
"""

import re
import json
import logging
import pickle
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logger = logging.getLogger(__name__)

class FAISSRetriever:
    """基于FAISS的向量检索器"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", 
                 index_path: str = "data/faiss_index", 
                 chunk_size: int = 512, 
                 overlap: int = 50):
        """
        初始化FAISS检索器
        
        Args:
            model_name: 句子嵌入模型名称
            index_path: FAISS索引保存路径
            chunk_size: 文本块大小
            overlap: 文本块重叠大小
        """
        self.model_name = model_name
        self.index_path = Path(index_path)
        self.chunk_size = chunk_size
        self.overlap = overlap
        
        # 初始化组件
        self.model = None
        self.index = None
        self.chunks = []
        self.chunk_metadata = []
        self.dimension = 384  # all-MiniLM-L6-v2的维度
        
        # 创建索引目录
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        # 初始化
        self._initialize_model()
        self._load_or_build_index()
    
    def _initialize_model(self):
        """初始化句子嵌入模型"""
        try:
            self.model = SentenceTransformer(self.model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"成功初始化语义模型: {self.model_name}, 维度: {self.dimension}")
        except Exception as e:
            logger.error(f"初始化语义模型失败: {e}")
            raise
    
    def _load_or_build_index(self):
        """加载或构建FAISS索引"""
        index_file = self.index_path / "faiss_index.bin"
        chunks_file = self.index_path / "chunks.json"
        metadata_file = self.index_path / "metadata.json"
        
        if index_file.exists() and chunks_file.exists() and metadata_file.exists():
            try:
                # 加载现有索引
                self.index = faiss.read_index(str(index_file))
                with open(chunks_file, 'r', encoding='utf-8') as f:
                    self.chunks = json.load(f)
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    self.chunk_metadata = json.load(f)
                logger.info(f"成功加载现有FAISS索引，包含 {len(self.chunks)} 个文档块")
            except Exception as e:
                logger.warning(f"加载现有索引失败: {e}，将重新构建")
                self._build_index()
        else:
            logger.info("未找到现有索引，开始构建新索引")
            self._build_index()
    
    def _build_index(self):
        """构建FAISS索引"""
        try:
            # 加载和预处理文档
            documents = self._load_documents()
            logger.info(f"加载了 {len(documents)} 个文档")
            
            # 分割文档为块
            self.chunks, self.chunk_metadata = self._split_documents(documents)
            logger.info(f"分割为 {len(self.chunks)} 个文档块")
            
            # 生成嵌入向量
            logger.info("开始生成文档嵌入向量...")
            embeddings = []
            batch_size = 32
            
            for i in tqdm(range(0, len(self.chunks), batch_size), desc="生成嵌入"):
                batch_chunks = self.chunks[i:i + batch_size]
                batch_embeddings = self.model.encode(batch_chunks, convert_to_numpy=True)
                embeddings.append(batch_embeddings)
            
            # 合并所有嵌入
            all_embeddings = np.vstack(embeddings)
            logger.info(f"生成了 {all_embeddings.shape[0]} 个嵌入向量")
            
            # 创建FAISS索引
            self.index = faiss.IndexFlatIP(self.dimension)  # 使用内积相似度
            self.index.add(all_embeddings.astype('float32'))
            
            # 保存索引和元数据
            faiss.write_index(self.index, str(self.index_path / "faiss_index.bin"))
            with open(self.index_path / "chunks.json", 'w', encoding='utf-8') as f:
                json.dump(self.chunks, f, ensure_ascii=False, indent=2)
            with open(self.index_path / "metadata.json", 'w', encoding='utf-8') as f:
                json.dump(self.chunk_metadata, f, ensure_ascii=False, indent=2)
            
            logger.info("FAISS索引构建完成并已保存")
            
        except Exception as e:
            logger.error(f"构建FAISS索引失败: {e}")
            raise
    
    def _load_documents(self) -> List[Dict[str, Any]]:
        """加载所有文档"""
        documents = []
        
        # 加载MD文件
        md_file = Path("data/课标miner.md")
        if md_file.exists():
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
                documents.append({
                    'id': 'curriculum_standards',
                    'title': '义务教育数学课程标准（2022年版）',
                    'content': content,
                    'type': 'curriculum',
                    'source': str(md_file)
                })
        
        # 加载知识图谱
        kg_file = Path("data/kowledge_graph.jsonl")
        if kg_file.exists():
            with open(kg_file, 'r', encoding='utf-8') as f:
                kg_data = []
                for line in f:
                    if line.strip():
                        try:
                            kg_data.append(json.loads(line.strip()))
                        except:
                            continue
                
                # 将知识图谱转换为文档
                for item in kg_data:
                    if 'name' in item and 'path' in item:
                        content = f"知识点: {item['name']}\n"
                        if 'path' in item:
                            content += f"知识路径: {item['path']}\n"
                        if 'description' in item:
                            content += f"描述: {item['description']}\n"
                        
                        documents.append({
                            'id': f"kg_{item.get('id', 'unknown')}",
                            'title': item['name'],
                            'content': content,
                            'type': 'knowledge_graph',
                            'source': 'knowledge_graph'
                        })
        
        # 加载题目样例
        questions_file = Path("data/all_questions_filtered.jsonl")
        if questions_file.exists():
            with open(questions_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            q_data = json.loads(line.strip())
                            if 'content' in q_data and 'knowledge' in q_data:
                                content = f"题目: {q_data['content']}\n"
                                content += f"知识点: {q_data['knowledge']}\n"
                                if 'difficulty' in q_data:
                                    content += f"难度: {q_data['difficulty']}\n"
                                if 'grade' in q_data:
                                    content += f"年级: {q_data['grade']}\n"
                                if 'competence' in q_data:
                                    content += f"素养: {q_data['competence']}\n"
                                
                                documents.append({
                                    'id': f"q_{q_data.get('id', 'unknown')}",
                                    'title': f"题目样例: {q_data['content'][:50]}...",
                                    'content': content,
                                    'type': 'question_sample',
                                    'source': 'question_samples'
                                })
                        except:
                            continue
        
        return documents
    
    def _split_documents(self, documents: List[Dict[str, Any]]) -> Tuple[List[str], List[Dict[str, Any]]]:
        """将文档分割为块"""
        chunks = []
        metadata = []
        
        for doc in documents:
            content = doc['content']
            
            # 按段落分割
            paragraphs = re.split(r'\n\s*\n', content)
            
            for para in paragraphs:
                para = para.strip()
                if len(para) < 50:  # 过滤太短的段落
                    continue
                
                # 如果段落太长，进一步分割
                if len(para) > self.chunk_size:
                    sub_chunks = self._split_long_text(para)
                    for i, sub_chunk in enumerate(sub_chunks):
                        chunks.append(sub_chunk)
                        metadata.append({
                            'doc_id': doc['id'],
                            'doc_title': doc['title'],
                            'doc_type': doc['type'],
                            'source': doc['source'],
                            'chunk_index': i,
                            'total_chunks': len(sub_chunks)
                        })
                else:
                    chunks.append(para)
                    metadata.append({
                        'doc_id': doc['id'],
                        'doc_title': doc['title'],
                        'doc_type': doc['type'],
                        'source': doc['source'],
                        'chunk_index': 0,
                        'total_chunks': 1
                    })
        
        return chunks, metadata
    
    def _split_long_text(self, text: str) -> List[str]:
        """分割长文本"""
        sentences = re.split(r'[。！？]', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            if len(current_chunk + sentence) > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += sentence + "。"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def search(self, query: str, top_k: int = 5, score_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """
        搜索相关文档
        
        Args:
            query: 查询文本
            top_k: 返回前k个结果
            score_threshold: 相似度阈值
            
        Returns:
            搜索结果列表
        """
        if self.index is None:
            logger.error("FAISS索引未初始化")
            return []
        
        try:
            # 生成查询向量
            query_embedding = self.model.encode([query], convert_to_numpy=True)
            
            # 搜索
            scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
            
            # 构建结果
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if score >= score_threshold:
                    results.append({
                        'content': self.chunks[idx],
                        'score': float(score),
                        'metadata': self.chunk_metadata[idx]
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"FAISS搜索失败: {e}")
            return []
    
    def search_curriculum_standards(self, knowledge_point: str, difficulty: str, 
                                  core_competency: str, grade: str, grade_level: str) -> Dict[str, Any]:
        """
        基于FAISS检索课程标准
        
        Args:
            knowledge_point: 知识点
            difficulty: 难度
            core_competency: 核心素养
            grade: 年级
            grade_level: 年级级别
            
        Returns:
            课程标准检索结果
        """
        try:
            results = {
                "knowledge_point": knowledge_point,
                "difficulty": difficulty,
                "core_competency": core_competency,
                "grade": grade,
                "grade_level": grade_level,
                "curriculum_requirements": [],
                "teaching_suggestions": [],
                "learning_objectives": [],
                "assessment_criteria": []
            }
            
            # 构建多个查询
            queries = [
                f"{knowledge_point} {core_competency} 教学要求 课程标准",
                f"{knowledge_point} {core_competency} 学习目标 培养",
                f"{grade} {knowledge_point} 数学 教育 标准",
                f"{core_competency} 素养 培养 教学建议",
                f"{difficulty} {knowledge_point} 评价 考核 标准",
                f"数学核心素养 {core_competency} 发展",
                f"{knowledge_point} 数学思维 能力 培养"
            ]
            
            # 搜索并收集结果
            all_results = []
            for query in queries:
                search_results = self.search(query, top_k=3, score_threshold=0.2)
                all_results.extend(search_results)
            
            # 去重
            seen_contents = set()
            unique_results = []
            for result in all_results:
                if result['content'] not in seen_contents:
                    seen_contents.add(result['content'])
                    unique_results.append(result)
            
            # 按分数排序
            unique_results.sort(key=lambda x: x['score'], reverse=True)
            
            # 分类结果
            for result in unique_results[:15]:  # 取前15个最相关的结果
                content = result['content']
                score = result['score']
                metadata = result['metadata']
                
                # 根据内容特征和元数据分类
                if metadata.get('doc_type') == 'curriculum':
                    if any(word in content for word in ["要求", "标准", "目标", "掌握", "理解"]):
                        results["curriculum_requirements"].append(f"{content} (相似度: {score:.3f})")
                    elif any(word in content for word in ["建议", "方法", "策略", "培养", "发展"]):
                        results["teaching_suggestions"].append(f"{content} (相似度: {score:.3f})")
                    elif any(word in content for word in ["学习", "掌握", "理解", "能力", "素养"]):
                        results["learning_objectives"].append(f"{content} (相似度: {score:.3f})")
                    elif any(word in content for word in ["评价", "评估", "考核", "测试", "检测"]):
                        results["assessment_criteria"].append(f"{content} (相似度: {score:.3f})")
                    else:
                        results["curriculum_requirements"].append(f"{content} (相似度: {score:.3f})")
                elif metadata.get('doc_type') == 'question_sample':
                    # 题目样例可以作为教学建议的参考
                    results["teaching_suggestions"].append(f"题目样例: {content} (相似度: {score:.3f})")
                elif metadata.get('doc_type') == 'knowledge_graph':
                    # 知识图谱信息可以作为课程要求
                    results["curriculum_requirements"].append(f"知识结构: {content} (相似度: {score:.3f})")
            
            # 限制每个类别的数量
            for key in results:
                if isinstance(results[key], list):
                    results[key] = results[key][:5]  # 最多5条
            
            logger.info(f"FAISS检索完成，找到 {sum(len(v) for v in results.values() if isinstance(v, list))} 条相关内容")
            return results
            
        except Exception as e:
            logger.error(f"FAISS课程标准检索失败: {e}")
            return {
                "knowledge_point": knowledge_point,
                "error": str(e)
            }
    
    def search_knowledge_graph(self, knowledge_point: str, grade: str = None, grade_level: str = None) -> List[Dict[str, Any]]:
        """基于FAISS检索知识图谱"""
        query = f"{knowledge_point} 知识图谱 概念 关系"
        if grade:
            query += f" {grade}"
        if grade_level:
            query += f" {grade_level}"
        
        results = self.search(query, top_k=10, score_threshold=0.2)
        
        # 过滤知识图谱相关结果
        kg_results = []
        for result in results:
            if result['metadata'].get('doc_type') == 'knowledge_graph':
                kg_results.append({
                    'name': result['metadata'].get('doc_title', ''),
                    'content': result['content'],
                    'score': result['score']
                })
        
        return kg_results
    
    def get_question_samples(self, knowledge_point: str, difficulty: str, grade: str, core_competency: str) -> List[Dict[str, Any]]:
        """基于FAISS获取题目样例"""
        query = f"{knowledge_point} {difficulty} {grade} {core_competency} 题目"
        
        results = self.search(query, top_k=5, score_threshold=0.2)
        
        # 过滤题目样例
        question_results = []
        for result in results:
            if result['metadata'].get('doc_type') == 'question_sample':
                question_results.append({
                    'content': result['content'],
                    'score': result['score']
                })
        
        return question_results

def test_faiss_retriever():
    """测试FAISS检索器"""
    print("🔍 测试FAISS检索器...")
    
    try:
        retriever = FAISSRetriever()
        
        # 测试基本搜索
        print("\n📝 测试基本搜索:")
        results = retriever.search("运算能力 培养 教学", top_k=3)
        print(f"找到 {len(results)} 个结果:")
        for i, result in enumerate(results, 1):
            print(f"  {i}. 相似度: {result['score']:.3f}")
            print(f"     类型: {result['metadata']['doc_type']}")
            print(f"     内容: {result['content'][:100]}...")
        
        # 测试课程标准检索
        print(f"\n📚 测试课程标准检索:")
        curriculum_results = retriever.search_curriculum_standards(
            knowledge_point="乘法分配律",
            difficulty="易",
            core_competency="运算能力",
            grade="四年级下",
            grade_level="下"
        )
        
        for key, values in curriculum_results.items():
            if isinstance(values, list) and values:
                print(f"   {key}: {len(values)} 条")
                for value in values[:2]:
                    print(f"     - {value[:80]}...")
        
        # 测试知识图谱检索
        print(f"\n🗺️ 测试知识图谱检索:")
        kg_results = retriever.search_knowledge_graph("乘法分配律", "四年级下", "下")
        print(f"找到 {len(kg_results)} 个知识图谱结果")
        for result in kg_results[:2]:
            print(f"  - {result['name']}: {result['content'][:80]}...")
        
        print(f"\n✅ FAISS检索器测试完成")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_faiss_retriever()
