#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于语义相似度的知识检索器
"""

import re
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class SemanticRetriever:
    """基于语义相似度的知识检索器"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        初始化语义检索器
        
        Args:
            model_name: 句子嵌入模型名称
        """
        self.model_name = model_name
        self.model = None
        self.math_knowledge_data = ""
        self.knowledge_chunks = []
        self.chunk_embeddings = None
        
        # 初始化数据
        self._load_data()
        self._initialize_model()
        self._prepare_chunks()
    
    def _load_data(self):
        """加载MD文件数据"""
        try:
            current_dir = Path(__file__).parent
            data_dir = current_dir / "data"
            math_knowledge_file = data_dir / "课标miner.md"
            
            if math_knowledge_file.exists():
                with open(math_knowledge_file, 'r', encoding='utf-8') as f:
                    self.math_knowledge_data = f.read()
                logger.info(f"成功加载MD文件: {math_knowledge_file}")
            else:
                logger.warning(f"MD文件不存在: {math_knowledge_file}")
                
        except Exception as e:
            logger.error(f"加载MD文件失败: {e}")
    
    def _initialize_model(self):
        """初始化句子嵌入模型"""
        try:
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"成功初始化语义模型: {self.model_name}")
        except Exception as e:
            logger.error(f"初始化语义模型失败: {e}")
            # 如果模型加载失败，使用简单的文本匹配作为备选
            self.model = None
    
    def _prepare_chunks(self):
        """将MD内容分割成语义块并生成嵌入"""
        if not self.math_knowledge_data or not self.model:
            return
        
        try:
            # 按段落分割内容
            paragraphs = self._split_into_paragraphs(self.math_knowledge_data)
            
            # 过滤和清理段落
            self.knowledge_chunks = []
            for para in paragraphs:
                if self._is_relevant_paragraph(para):
                    cleaned_para = self._clean_paragraph(para)
                    if cleaned_para and len(cleaned_para) > 20:  # 过滤太短的段落
                        self.knowledge_chunks.append(cleaned_para)
            
            # 生成嵌入向量
            if self.knowledge_chunks:
                self.chunk_embeddings = self.model.encode(self.knowledge_chunks)
                logger.info(f"成功生成 {len(self.knowledge_chunks)} 个知识块的嵌入向量")
            else:
                logger.warning("没有找到有效的知识块")
                
        except Exception as e:
            logger.error(f"准备知识块失败: {e}")
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """将文本分割成段落"""
        # 按双换行符分割
        paragraphs = re.split(r'\n\s*\n', text)
        
        # 进一步按标题分割
        result = []
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            # 如果段落很长，按句子进一步分割
            if len(para) > 500:
                sentences = re.split(r'[。！？]', para)
                current_chunk = ""
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                    if len(current_chunk + sentence) > 300:
                        if current_chunk:
                            result.append(current_chunk.strip())
                        current_chunk = sentence
                    else:
                        current_chunk += sentence + "。"
                if current_chunk:
                    result.append(current_chunk.strip())
            else:
                result.append(para)
        
        return result
    
    def _is_relevant_paragraph(self, para: str) -> bool:
        """判断段落是否与数学教育相关"""
        # 排除明显不相关的内容
        exclude_keywords = [
            "齐鲁校园", "关注", "微信公众号", "二维码", "图片", "表格",
            "续表", "图", "表", "![](images/", "http", "www."
        ]
        
        for keyword in exclude_keywords:
            if keyword in para:
                return False
        
        # 包含数学教育相关关键词
        math_keywords = [
            "数学", "运算", "几何", "推理", "数据", "素养", "能力", "意识",
            "学习", "教学", "课程", "标准", "目标", "要求", "建议", "评价",
            "小学", "初中", "年级", "学生", "教师", "教育"
        ]
        
        return any(keyword in para for keyword in math_keywords)
    
    def _clean_paragraph(self, para: str) -> str:
        """清理段落内容"""
        # 移除多余的空白字符
        para = re.sub(r'\s+', ' ', para)
        
        # 移除表格标记
        para = re.sub(r'<table>.*?</table>', '', para, flags=re.DOTALL)
        para = re.sub(r'<tr>.*?</tr>', '', para, flags=re.DOTALL)
        para = re.sub(r'<td>.*?</td>', '', para, flags=re.DOTALL)
        
        # 移除图片标记
        para = re.sub(r'!\[.*?\]\(.*?\)', '', para)
        
        return para.strip()
    
    def search_semantic(self, query: str, top_k: int = 5, threshold: float = 0.3) -> List[Dict[str, Any]]:
        """
        基于语义相似度搜索相关内容
        
        Args:
            query: 查询文本
            top_k: 返回前k个最相关的结果
            threshold: 相似度阈值
            
        Returns:
            相关内容的列表
        """
        if not self.model or not self.chunk_embeddings is not None:
            logger.warning("语义模型未初始化，使用关键词匹配")
            return self._fallback_keyword_search(query)
        
        try:
            # 生成查询向量
            query_embedding = self.model.encode([query])
            
            # 计算相似度
            similarities = cosine_similarity(query_embedding, self.chunk_embeddings)[0]
            
            # 获取最相关的结果
            results = []
            for i, similarity in enumerate(similarities):
                if similarity >= threshold:
                    results.append({
                        'content': self.knowledge_chunks[i],
                        'similarity': float(similarity),
                        'index': i
                    })
            
            # 按相似度排序并返回前k个
            results.sort(key=lambda x: x['similarity'], reverse=True)
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"语义搜索失败: {e}")
            return self._fallback_keyword_search(query)
    
    def _fallback_keyword_search(self, query: str) -> List[Dict[str, Any]]:
        """备选的关键词搜索方法"""
        results = []
        query_lower = query.lower()
        
        for i, chunk in enumerate(self.knowledge_chunks):
            chunk_lower = chunk.lower()
            # 简单的关键词匹配
            if any(word in chunk_lower for word in query_lower.split()):
                results.append({
                    'content': chunk,
                    'similarity': 0.5,  # 给一个默认相似度
                    'index': i
                })
        
        return results[:5]
    
    def search_curriculum_standards_semantic(self, knowledge_point: str, difficulty: str, 
                                           core_competency: str, grade: str, grade_level: str) -> Dict[str, Any]:
        """
        基于语义相似度检索课程标准
        
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
            
            # 构建查询文本
            queries = [
                f"{knowledge_point} {core_competency} 教学要求",
                f"{knowledge_point} {core_competency} 学习目标",
                f"{grade} {knowledge_point} 课程标准",
                f"{core_competency} 培养 教学建议",
                f"{difficulty} {knowledge_point} 评价标准"
            ]
            
            # 搜索相关内容
            all_results = []
            for query in queries:
                semantic_results = self.search_semantic(query, top_k=3, threshold=0.2)
                all_results.extend(semantic_results)
            
            # 去重并按相似度排序
            seen_contents = set()
            unique_results = []
            for result in all_results:
                if result['content'] not in seen_contents:
                    seen_contents.add(result['content'])
                    unique_results.append(result)
            
            unique_results.sort(key=lambda x: x['similarity'], reverse=True)
            
            # 分类内容
            for result in unique_results[:10]:  # 取前10个最相关的结果
                content = result['content']
                similarity = result['similarity']
                
                # 根据内容特征分类
                if any(word in content for word in ["要求", "标准", "目标", "掌握", "理解"]):
                    results["curriculum_requirements"].append(f"{content} (相似度: {similarity:.3f})")
                elif any(word in content for word in ["建议", "方法", "策略", "培养", "发展"]):
                    results["teaching_suggestions"].append(f"{content} (相似度: {similarity:.3f})")
                elif any(word in content for word in ["学习", "掌握", "理解", "能力", "素养"]):
                    results["learning_objectives"].append(f"{content} (相似度: {similarity:.3f})")
                elif any(word in content for word in ["评价", "评估", "考核", "测试", "检测"]):
                    results["assessment_criteria"].append(f"{content} (相似度: {similarity:.3f})")
                else:
                    # 默认归类为要求
                    results["curriculum_requirements"].append(f"{content} (相似度: {similarity:.3f})")
            
            # 限制每个类别的数量
            for key in results:
                if isinstance(results[key], list):
                    results[key] = results[key][:3]
            
            logger.info(f"语义检索完成，找到 {sum(len(v) for v in results.values() if isinstance(v, list))} 条相关内容")
            return results
            
        except Exception as e:
            logger.error(f"语义课程标准检索失败: {e}")
            return {
                "knowledge_point": knowledge_point,
                "error": str(e)
            }

def test_semantic_retrieval():
    """测试语义检索功能"""
    print("🔍 测试语义检索功能...")
    
    try:
        retriever = SemanticRetriever()
        
        # 测试查询
        test_queries = [
            "运算能力 培养 教学建议",
            "几何直观 空间观念 学习目标",
            "推理能力 逻辑思维 评价标准",
            "数据观念 统计 课程要求"
        ]
        
        for query in test_queries:
            print(f"\n📝 查询: {query}")
            results = retriever.search_semantic(query, top_k=3)
            
            print(f"   找到 {len(results)} 个相关结果:")
            for i, result in enumerate(results, 1):
                print(f"     {i}. 相似度: {result['similarity']:.3f}")
                print(f"        内容: {result['content'][:100]}...")
        
        # 测试完整的课程标准检索
        print(f"\n📚 测试课程标准语义检索:")
        curriculum_results = retriever.search_curriculum_standards_semantic(
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
        
        print(f"\n✅ 语义检索测试完成")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_semantic_retrieval()
