#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
标准化输出解析器 - 用于解析各智能体的标准化输出格式
"""

import json
import re
import logging
from typing import Dict, Any, Optional, List, Union

logger = logging.getLogger(__name__)

class OutputParser:
    """标准化输出解析器"""
    
    def __init__(self):
        self.parsers = {
            'planner': self._parse_planner_output,
            'writer': self._parse_writer_output,
            'solver': self._parse_solver_output,
            'educator': self._parse_educator_output,
            'teacher': self._parse_teacher_output
        }
    
    def parse_agent_output(self, agent_type: str, raw_output: str) -> Dict[str, Any]:
        """解析智能体输出"""
        if agent_type not in self.parsers:
            raise ValueError(f"不支持的智能体类型: {agent_type}")
        
        try:
            # 首先尝试JSON解析
            if raw_output.strip().startswith('{'):
                return json.loads(raw_output)
            
            # 如果不是JSON，使用正则表达式解析
            return self.parsers[agent_type](raw_output)
        except Exception as e:
            logger.error(f"解析{agent_type}输出失败: {e}")
            return {"status": "error", "raw_output": raw_output}
    
    def _parse_planner_output(self, output: str) -> Dict[str, Any]:
        """解析Planner输出"""
        result = {
            "planning_id": self._extract_planning_id(output),
            "knowledge_planning": self._extract_section(output, "知识点规划"),
            "difficulty_planning": self._extract_section(output, "学情与难度规划"),
            "competency_planning": self._extract_section(output, "素养规划"),
            "question_directions": self._extract_question_directions(output),
            "status": "success"
        }
        return result
    
    def _parse_writer_output(self, output: str) -> Dict[str, Any]:
        """解析Writer输出"""
        result = {
            "question_id": self._extract_question_id(output),
            "selected_direction": self._extract_selected_direction(output),
            "question_content": self._extract_question_content(output),
            "status": "success"
        }
        return result
    
    def _parse_solver_output(self, output: str) -> Dict[str, Any]:
        """解析Solver输出"""
        result = {
            "evaluation_id": self._extract_evaluation_id(output),
            "scores": self._extract_solver_scores(output),
            "overall_score": self._extract_overall_score(output),
            "pass_status": self._extract_pass_status(output),
            "feedback": self._extract_solver_feedback(output),
            "corrected_answer": self._extract_corrected_answer(output),
            "status": "success"
        }
        return result
    
    def _parse_educator_output(self, output: str) -> Dict[str, Any]:
        """解析Educator输出"""
        result = {
            "evaluation_id": self._extract_evaluation_id(output),
            "scores": self._extract_educator_scores(output),
            "overall_score": self._extract_overall_score(output),
            "pass_status": self._extract_pass_status(output),
            "feedback": self._extract_educator_feedback(output),
            "status": "success"
        }
        return result
    
    def _parse_teacher_output(self, output: str) -> Dict[str, Any]:
        """解析Teacher输出"""
        result = {
            "check_id": self._extract_check_id(output),
            "check_result": self._extract_check_result(output),
            "feedback": self._extract_teacher_feedback(output),
            "status": "success"
        }
        return result
    
    # ==================== 辅助解析方法 ====================
    
    def _extract_planning_id(self, output: str) -> str:
        """提取规划ID"""
        match = re.search(r'规划ID[：:]\s*(\w+)', output)
        return match.group(1) if match else f"plan_{hash(output) % 10000:04d}"
    
    def _extract_section(self, output: str, section_name: str) -> str:
        """提取指定章节内容"""
        pattern = rf'### {section_name}\n(.*?)(?=### |$)'
        match = re.search(pattern, output, re.DOTALL)
        return match.group(1).strip() if match else ""
    
    def _extract_question_directions(self, output: str) -> List[Dict[str, str]]:
        """提取出题方向"""
        directions = []
        direction_pattern = r'方向([一二三])[：:]\s*(.*?)(?=方向[一二三]|$)'
        matches = re.findall(direction_pattern, output, re.DOTALL)
        
        for i, (num, content) in enumerate(matches, 1):
            directions.append({
                "direction_id": f"dir_{i}",
                "description": content.strip(),
                "scenario_type": self._classify_scenario_type(content)
            })
        
        return directions
    
    def _classify_scenario_type(self, content: str) -> str:
        """分类情境类型"""
        if any(keyword in content for keyword in ["生活", "日常", "购物", "运动", "游戏"]):
            return "生活情境类"
        elif any(keyword in content for keyword in ["图形", "几何", "形状", "视觉"]):
            return "图形几何类"
        elif any(keyword in content for keyword in ["逻辑", "推理", "分析", "思维"]):
            return "逻辑推理类"
        else:
            return "通用类"
    
    def _extract_question_id(self, output: str) -> str:
        """提取题目ID"""
        match = re.search(r'题目ID[：:]\s*(\w+)', output)
        return match.group(1) if match else f"q_{hash(output) % 10000:04d}"
    
    def _extract_selected_direction(self, output: str) -> Dict[str, str]:
        """提取选择的方向"""
        direction_match = re.search(r'我选择了规划中的方向([一二三])', output)
        reason_match = re.search(r'理由[：:]\s*(.*?)(?=\n|$)', output)
        
        return {
            "direction_id": f"dir_{'一二三'.index(direction_match.group(1)) + 1}" if direction_match else "dir_1",
            "reason": reason_match.group(1).strip() if reason_match else "未提供理由"
        }
    
    def _extract_question_content(self, output: str) -> Dict[str, Any]:
        """提取题目内容"""
        # 提取题干
        stem_match = re.search(r'### 题目：\n(.*?)(?=\n\n|$)', output, re.DOTALL)
        stem = stem_match.group(1).strip() if stem_match else ""
        
        # 提取选项
        options = []
        for option in ['A', 'B', 'C', 'D']:
            option_match = re.search(rf'{option}\.\s*(.*?)(?=\n[A-D]\.|$)', output)
            if option_match:
                options.append(f"{option}. {option_match.group(1).strip()}")
        
        # 提取答案
        answer_match = re.search(r'答案为[：:]?\s*([A-D])', output)
        answer = answer_match.group(1) if answer_match else ""
        
        # 判断题型
        question_type = "填空题" if "______" in stem else "选择题"
        
        return {
            "stem": stem,
            "options": options,
            "answer": answer,
            "question_type": question_type
        }
    
    def _extract_evaluation_id(self, output: str) -> str:
        """提取评估ID"""
        match = re.search(r'评估ID[：:]\s*(\w+)', output)
        return match.group(1) if match else f"eval_{hash(output) % 10000:04d}"
    
    def _extract_solver_scores(self, output: str) -> Dict[str, int]:
        """提取Solver评分"""
        scores = {}
        score_patterns = {
            "logic_completeness": r'逻辑完备性[：:]\s*(\d)',
            "clarity": r'表述清晰度[：:]\s*(\d)',
            "misconception_guidance": r'误区引导性[：:]\s*(\d)',
            "solution_appropriateness": r'解法适切性[：:]\s*(\d)',
            "question_type_compliance": r'题型符合性[：:]\s*(\d)'
        }
        
        for key, pattern in score_patterns.items():
            match = re.search(pattern, output)
            scores[key] = int(match.group(1)) if match else 0
        
        return scores
    
    def _extract_educator_scores(self, output: str) -> Dict[str, int]:
        """提取Educator评分"""
        # Educator使用相同的评分维度
        return self._extract_solver_scores(output)
    
    def _extract_overall_score(self, output: str) -> float:
        """提取总体评分"""
        # 对于binary模式，计算平均分
        scores = self._extract_solver_scores(output)
        if scores:
            return sum(scores.values()) / len(scores)
        return 0.0
    
    def _extract_pass_status(self, output: str) -> str:
        """提取通过状态"""
        match = re.search(r'结论[：:]\s*(PASS=\d)', output)
        return match.group(1) if match else "PASS=0"
    
    def _extract_solver_feedback(self, output: str) -> Dict[str, Any]:
        """提取Solver反馈"""
        conclusion_match = re.search(r'结论[：:]\s*(.*?)(?=\n|$)', output)
        conclusion = conclusion_match.group(1).strip() if conclusion_match else ""
        
        suggestions = []
        suggestion_pattern = r'修改建议[：:]\s*(.*?)(?=\n\n|$)'
        suggestion_match = re.search(suggestion_pattern, output, re.DOTALL)
        if suggestion_match:
            suggestion_text = suggestion_match.group(1).strip()
            # 按行分割建议
            for line in suggestion_text.split('\n'):
                line = line.strip()
                if line and (line.startswith('-') or line.startswith('•') or 
                           any(line.startswith(f'{i}.') for i in range(1, 10))):
                    suggestions.append(line)
        
        return {
            "conclusion": conclusion,
            "suggestions": suggestions
        }
    
    def _extract_educator_feedback(self, output: str) -> Dict[str, Any]:
        """提取Educator反馈"""
        # 使用相同的逻辑
        return self._extract_solver_feedback(output)
    
    def _extract_corrected_answer(self, output: str) -> str:
        """提取修正答案"""
        patterns = [
            r'将答案更正为[：:]\s*([A-D])',
            r'将答案更正为[：:]\s*([^，,；;。\n]+)',
            r'答案[为是]?[：:]\s*([A-D])',
            r'答案[为是]?[：:]\s*([^，,；;。\n]+)',
            r'正确答案[为是]?[：:]\s*([A-D])',
            r'正确答案[为是]?[：:]\s*([^，,；;。\n]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, output)
            if match:
                return match.group(1).strip()
        
        return ""
    
    def _extract_check_id(self, output: str) -> str:
        """提取检查ID"""
        match = re.search(r'检查ID[：:]\s*(\w+)', output)
        return match.group(1) if match else f"check_{hash(output) % 10000:04d}"
    
    def _extract_check_result(self, output: str) -> str:
        """提取检查结果"""
        match = re.search(r'CHECK\s*=\s*(\d)', output)
        return f"CHECK={match.group(1)}" if match else "CHECK=0"
    
    def _extract_teacher_feedback(self, output: str) -> str:
        """提取教师反馈"""
        # 提取除CHECK之外的内容作为反馈
        check_match = re.search(r'(.*?)CHECK\s*=\s*\d', output, re.DOTALL)
        if check_match:
            return check_match.group(1).strip()
        return output.strip()

# ==================== 使用示例 ====================

def test_parser():
    """测试解析器"""
    parser = OutputParser()
    
    # 测试Planner输出
    planner_output = """
    ### 知识点规划
    这是一个关于二次函数的题目...
    
    ### 学情与难度规划
    适合九年级学生...
    
    ### 素养规划
    培养学生的数学运算能力...
    
    ### 出题方向（三种不同方向）
    方向一：生活情境类 - 以购物场景为背景
    方向二：图形几何类 - 结合抛物线图像
    方向三：逻辑推理类 - 注重解题过程
    """
    
    result = parser.parse_agent_output('planner', planner_output)
    print("Planner解析结果:", json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    test_parser()
