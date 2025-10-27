#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版知识检索器 - 只做检索，不做分析
"""

import json
import os
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
import random

logger = logging.getLogger(__name__)

class KnowledgeRetriever:
    """简化版知识检索器 - 只做检索，不做分析"""
    
    def __init__(self, data_dir: str = None):
        # 使用相对路径
        if data_dir is None:
            current_dir = Path(__file__).parent
            self.data_dir = current_dir / "data"
            self.knowledge_graph_file = self.data_dir / "kowledge_graph.jsonl"
            self.math_knowledge_file = self.data_dir / "课标miner.md"
            self.questions_file = self.data_dir / "all_questions_choice_blank_with_type.jsonl"
        else:
            current_dir = Path.cwd()
            self.data_dir = current_dir / data_dir
            self.knowledge_graph_file = self.data_dir / "kowledge_graph.jsonl"
            self.math_knowledge_file = self.data_dir / "课标miner.md"
            self.questions_file = self.data_dir / "all_questions_choice_blank_with_type.jsonl"
        
        self.knowledge_graph_data = []
        self.math_knowledge_data = ""
        self.questions_data = []
        self._load_knowledge_data()
    
    # 通用安全转换工具
    def _to_text(self, value: Any) -> str:
        """将任意值安全转换为字符串（list会被拼接）。"""
        try:
            if value is None:
                return ""
            if isinstance(value, str):
                return value
            if isinstance(value, list):
                return " ".join([self._to_text(v) for v in value])
            return str(value)
        except Exception:
            return ""
    
    def _load_knowledge_data(self):
        """加载知识数据"""
        try:
            # 加载知识图谱数据
            if self.knowledge_graph_file.exists():
                with open(self.knowledge_graph_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            data = json.loads(line.strip())
                            self.knowledge_graph_data.append(data)
                        except json.JSONDecodeError:
                            continue
                logger.info(f"成功加载知识图谱数据: {len(self.knowledge_graph_data)} 条")
            else:
                logger.warning(f"知识图谱文件不存在: {self.knowledge_graph_file}")
            
            # 加载数学课程标准数据
            if self.math_knowledge_file.exists():
                with open(self.math_knowledge_file, 'r', encoding='utf-8') as f:
                    self.math_knowledge_data = f.read()
                logger.info(f"成功加载数学课程标准数据: {len(self.math_knowledge_data)} 字符")
            else:
                logger.warning(f"数学课程标准文件不存在: {self.math_knowledge_file}")
            
            # 加载题目信息数据
            if self.questions_file.exists():
                with open(self.questions_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            data = json.loads(line.strip())
                            self.questions_data.append(data)
                        except json.JSONDecodeError:
                            continue
                logger.info(f"成功加载题目信息数据: {len(self.questions_data)} 条")
            else:
                logger.warning(f"题目信息文件不存在: {self.questions_file}")
                
        except Exception as e:
            logger.error(f"加载知识数据失败: {e}")
    
    def search_knowledge_graph(self, knowledge_point: str, grade: str = None, grade_level: str = None) -> List[Dict[str, Any]]:
        """从知识图谱中搜索知识点"""
        try:
            # 统一参数为字符串
            knowledge_point = self._to_text(knowledge_point)
            grade = self._to_text(grade)
            grade_level = self._to_text(grade_level)
            results = []
            
            # 构建年级匹配模式
            grade_patterns = []
            if grade and grade_level:
                if "小学" in grade:
                    if "四" in grade_level:
                        grade_patterns = [r"四[上下]知识图谱"]
                    elif "五" in grade_level:
                        grade_patterns = [r"五[上下]知识图谱"]
                    elif "六" in grade_level:
                        grade_patterns = [r"六[上下]知识图谱"]
                elif "初中" in grade:
                    if "七" in grade_level:
                        grade_patterns = [r"七[上下]知识图谱"]
                    elif "八" in grade_level:
                        grade_patterns = [r"八[上下]知识图谱"]
                    elif "九" in grade_level:
                        grade_patterns = [r"九[上下]知识图谱"]
            
            # 优先匹配name字段
            for item in self.knowledge_graph_data:
                item_name = self._to_text(item.get("name", ""))
                if knowledge_point and item_name and knowledge_point == item_name:
                    if grade_patterns:
                        item_grade = self._to_text(item.get("grade", ""))
                        for pattern in grade_patterns:
                            if re.search(pattern, item_grade):
                                results.append(item)
                                break
                    else:
                        results.append(item)
            
            # 如果name匹配没结果，再用text_for_embedding匹配
            if not results:
                for item in self.knowledge_graph_data:
                    text_emb = self._to_text(item.get("text_for_embedding", ""))
                    if knowledge_point and text_emb and knowledge_point in text_emb:
                        if grade_patterns:
                            item_grade = self._to_text(item.get("grade", ""))
                            for pattern in grade_patterns:
                                if re.search(pattern, item_grade):
                                    results.append(item)
                                    break
                        else:
                            results.append(item)
            
            return results
            
        except Exception as e:
            logger.error(f"搜索知识图谱失败: {e}")
            return []
    
    def search_curriculum_standards(self, knowledge_point: str, grade: str = None, difficulty: str = None, core_competency: str = None) -> str:
        """搜索课程标准相关内容"""
        try:
            knowledge_point = self._to_text(knowledge_point)
            grade = self._to_text(grade)
            if not self.math_knowledge_data:
                return ""
            
            # 简单的关键词匹配
            lines = self.math_knowledge_data.split('\n')
            relevant_lines = []
            
            for line in lines:
                line_text = self._to_text(line)
                if knowledge_point and knowledge_point in line_text:
                    relevant_lines.append(line.strip())
            
            return '\n'.join(relevant_lines[:10])  # 只返回前10行相关内容
            
        except Exception as e:
            logger.error(f"搜索课程标准失败: {e}")
            return ""
    
    def search_question_samples(self, knowledge_point: str, difficulty: str, grade: str, competence: str, question_type: str = None, max_samples: int = 3) -> List[Dict[str, Any]]:
        """搜索题目样例"""
        try:
            # 设置随机种子确保每次调用都有不同的随机性
            random.seed()
            # 统一参数为字符串
            knowledge_point = self._to_text(knowledge_point)
            grade = self._to_text(grade)
            competence_str = self._to_text(competence)
            def normalize_difficulty(diff: str) -> str:
                if "易" in diff or "简单" in diff:
                    return "易"
                elif "中" in diff:
                    return "中"
                elif "难" in diff or "困难" in diff:
                    return "难"
                return diff

            def grade_matches(cfg_grade: str, q_grade: str) -> bool:
                if not cfg_grade or not q_grade:
                    return False
                def base(g: str) -> str:
                    return g.replace("上", "").replace("下", "").replace("（", "").replace(")", "")
                return base(cfg_grade) in base(q_grade) or base(q_grade) in base(cfg_grade)

            def competency_matches(cfg_comp: str, q_comp: list) -> bool:
                if not cfg_comp or not isinstance(q_comp, list):
                    return False
                tokens = [t.strip() for t in re.split(r"[，、,\s]+", cfg_comp) if t.strip()]
                for token in tokens:
                    for comp in q_comp:
                        if token == comp or token in comp or comp in token:
                            return True
                return False

            # 筛选相关题目
            relevant_questions = []
            for question in self.questions_data:
                q_grade = self._to_text(question.get('grade', ''))
                q_kn = self._to_text(question.get('knowledge', ''))
                q_comp = question.get('competence', [])

                if not (knowledge_point and q_kn and (knowledge_point in q_kn or q_kn in knowledge_point)):
                    continue
                if not grade_matches(grade, q_grade):
                    continue
                if not competency_matches(competence_str, q_comp):
                    continue

                d = normalize_difficulty(self._to_text(question.get('difficulty', '')))
                if d in ['易', '中', '难']:
                    relevant_questions.append(question)

            # 按难度分组选择
            difficulty_groups = {'易': [], '中': [], '难': []}
            for q in relevant_questions:
                d = normalize_difficulty(q.get('difficulty', ''))
                if d in difficulty_groups:
                    difficulty_groups[d].append(q)
            
            # 对每个难度组的题目进行随机打乱
            for diff_key in difficulty_groups:
                random.shuffle(difficulty_groups[diff_key])

            # 从每个难度组中随机选择1个
            selected_questions = []
            for diff_key in ['易', '中', '难']:
                candidates = difficulty_groups.get(diff_key, [])
                if candidates:
                    if question_type:
                        same_type = [q for q in candidates if self._get_question_type(q) == question_type]
                        pool = same_type if same_type else candidates
                    else:
                        pool = candidates
                    
                    if pool:
                        # 使用random.choice确保更好的随机性
                        selected_questions.append(random.choice(pool))
                        if len(selected_questions) >= max_samples:
                            break

            return selected_questions[:max_samples]
            
        except Exception as e:
            logger.error(f"搜索题目样例失败: {e}")
            return []
    
    def _get_question_type(self, question: Dict[str, Any]) -> str:
        """检测题目类型"""
        try:
            qtype = question.get('type')
            if isinstance(qtype, str) and qtype in {"选择题", "填空题", "判断题"}:
                return qtype
            
            content = question.get('content', '')
            answer = question.get('answer', '')
            
            # 填空题检测
            if re.search(r"_{3,}|＿{3,}|——+|—{3,}", content):
                return "填空题"
            # 判断题检测
            if re.search(r"(^|\n)\s*[AB]\s*[\.、\)]\s*(对|错)(\s|$)", content, flags=re.MULTILINE):
                return "判断题"
            # 选择题检测
            clean_ans = re.sub(r"[^A-Za-z]", "", answer).upper()[:1]
            if clean_ans in {"A", "B", "C", "D"}:
                return "选择题"
            return "其他"
        except Exception:
            return "其他"

    def retrieve_knowledge_for_planner(self, exercise_config: Dict) -> str:
        """为Planner检索知识（只检索，不分析）"""
        try:
            knowledge_point = exercise_config.get('knowledge_point', '')
            grade = exercise_config.get('grade', '')
            grade_level = exercise_config.get('grade_level', '')
            core_competency = exercise_config.get('core_competency', '')
            
            results = {
                "knowledge_graph": [],
                "curriculum_standards": "",
                "question_samples": []
            }
            
            # 检索知识图谱
            if knowledge_point:
                kg_results = self.search_knowledge_graph(knowledge_point, grade, grade_level)
                results["knowledge_graph"] = kg_results
            
            # 检索课程标准
            if knowledge_point:
                curriculum_content = self.search_curriculum_standards(knowledge_point, grade)
                results["curriculum_standards"] = curriculum_content
            
            # 检索题目样例 - 检索同年级同素养同知识点，不同难度（易中难）
            if knowledge_point and grade and core_competency:
                question_samples = self.search_question_samples(
                    knowledge_point, 
                    '',  # 不限制难度，检索所有难度
                    grade, 
                    core_competency,
                    exercise_config.get('exercise_type', ''),
                    max_samples=3
                )
                results["question_samples"] = question_samples
            
            return results
            
        except Exception as e:
            logger.error(f"为Planner检索知识失败: {e}")
            return {"knowledge_graph": [], "curriculum_standards": "", "question_samples": []}

    def retrieve_knowledge_for_writer(self, exercise_config: Dict) -> str:
        """为Writer检索知识（只检索，不分析）"""
        try:
            knowledge_point = exercise_config.get('knowledge_point', '')
            grade = exercise_config.get('grade', '')
            core_competency = exercise_config.get('core_competency', '')
            question_type = exercise_config.get('exercise_type', '')
            
            # 只检索题目样例 - 检索同年级同素养同知识点，不同难度（易中难）
            question_samples = self.search_question_samples(
                knowledge_point, 
                '',  # 不限制难度，检索所有难度
                grade, 
                core_competency,
                question_type,
                max_samples=3
            )
            
            # 简单格式化输出
            if not question_samples:
                return ""
            
            parts = []
            for i, q in enumerate(question_samples, 1):
                content = q.get('content', '')
                answer = q.get('answer', '')
                difficulty = q.get('difficulty', '未知')
                parts.append(f"参考样例{i} (难度: {difficulty}):")
                parts.append(f"{content}")
                parts.append(f"答案: {answer}")
            
            return "\n".join(parts)
            
        except Exception as e:
            logger.error(f"为Writer检索知识失败: {e}")
            return ""
    
    def retrieve_knowledge_for_evaluators(self, exercise_config: Dict) -> str:
        """为评分者检索知识（只检索，不分析）"""
        try:
            knowledge_point = exercise_config.get('knowledge_point', '')
            grade = exercise_config.get('grade', '')
            core_competency = exercise_config.get('core_competency', '')
            question_type = exercise_config.get('exercise_type', '')
            
            # 只检索题目样例 - 检索同年级同素养同知识点，不同难度（易中难）
            question_samples = self.search_question_samples(
                knowledge_point, 
                '',  # 不限制难度，检索所有难度
                grade, 
                core_competency,
                question_type,
                max_samples=3
            )
            
            # 简单格式化输出
            if not question_samples:
                return ""
            
            parts = []
            for i, q in enumerate(question_samples, 1):
                content = q.get('content', '')
                answer = q.get('answer', '')
                difficulty = q.get('difficulty', '未知')
                parts.append(f"参考样例{i} (难度: {difficulty}):")
                parts.append(f"{content}")
                parts.append(f"答案: {answer}")
            
            return "\n".join(parts)
            
        except Exception as e:
            logger.error(f"为评分者检索知识失败: {e}")
            return ""
    
    def format_knowledge_for_prompt(self, knowledge_graph_results: List[Dict], curriculum_results: str) -> str:
        """格式化知识为提示词"""
        try:
            parts = []
            
            # 添加知识图谱信息
            if knowledge_graph_results:
                parts.append("=== 知识图谱信息 ===")
                for i, kg in enumerate(knowledge_graph_results, 1):
                    name = kg.get('name', '')
                    text = kg.get('text_for_embedding', '')
                    if name and text:
                        parts.append(f"知识点{i}: {name}")
                        parts.append(f"内容: {text}")
            
            # 添加课程标准信息
            if curriculum_results:
                parts.append("\n=== 课程标准信息 ===")
                parts.append(curriculum_results)
            
            return "\n".join(parts)
            
        except Exception as e:
            logger.error(f"格式化知识失败: {e}")
            return ""
    
    def get_question_samples_for_planner(self, knowledge_point: str, difficulty: str, grade: str, competence: str) -> str:
        """为Planner提供题目样例（只检索，不分析）"""
        try:
            if not self.questions_data:
                return "题目数据未加载"
            
            # 只检索题目样例，不做分析
            question_samples = self.search_question_samples(
                knowledge_point, 
                difficulty, 
                grade, 
                competence,
                max_samples=3
            )
            
            if not question_samples:
                return f"未找到匹配的题目样例：知识点={knowledge_point}, 难度={difficulty}, 年级={grade}, 素养={competence}"
            
            # 简单格式化输出
            parts = []
            parts.append(f"=== 题目样例参考 ===")
            parts.append(f"以下为{difficulty}等题的真实样例，供规划参考：")
            
            for i, q in enumerate(question_samples, 1):
                content = q.get('content', '')
                answer = q.get('answer', '')
                difficulty = q.get('difficulty', '未知')
                parts.append(f"\n参考样例{i} (难度: {difficulty}):")
                parts.append(f"题目: {content}")
                parts.append(f"答案: {answer}")
            
            return "\n".join(parts)
            
        except Exception as e:
            logger.error(f"为Planner提供题目样例失败: {e}")
            return f"获取题目样例失败: {str(e)}"