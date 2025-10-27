#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
轻量级解析器 - 解析轻量级标准化输出格式
"""

import json
import re
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

class LightweightParser:
    """轻量级解析器 - 解析轻量级标准化输出格式"""
    
    def __init__(self):
        self.parsers = {
            'writer': self._parse_writer_output,
            'solver': self._parse_solver_output,
            'educator': self._parse_educator_output,
            'rewrite': self._parse_rewrite_output
        }
    
    def parse_agent_output(self, agent_type: str, raw_output: str) -> Dict[str, Any]:
        """解析智能体输出"""
        if agent_type not in self.parsers:
            logger.warning(f"不支持的智能体类型: {agent_type}")
            return {"raw_output": raw_output}
        
        try:
            return self.parsers[agent_type](raw_output)
        except Exception as e:
            logger.error(f"解析{agent_type}输出失败: {e}")
            return {"raw_output": raw_output, "parse_error": str(e)}
    
    def _parse_writer_output(self, output: str) -> Dict[str, Any]:
        """解析Writer输出"""
        try:
            # 首先尝试JSON解析
            if output.strip().startswith('{'):
                return json.loads(output)
            
            # 检查是否被包装在代码块中
            if '```json' in output:
                # 提取JSON部分
                json_match = re.search(r'```json\s*\n(.*?)\n```', output, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                    # 处理JSON中的中文引号问题
                    json_str = json_str.replace('"', '"').replace('"', '"')
                    try:
                        result = json.loads(json_str)
                        logger.info(f"Writer JSON解析成功，question: {result.get('question', '')[:50]}...")
                        return result
                    except json.JSONDecodeError as e:
                        logger.error(f"Writer JSON解析失败: {e}")
                        logger.error(f"JSON字符串: {json_str[:200]}...")
                        # 继续使用正则表达式解析
            
            # 如果不是JSON，使用正则表达式解析
            result = {}
            
            # 提取方向
            dir_match = re.search(r'"dir":\s*"([^"]+)"', output)
            result['dir'] = dir_match.group(1) if dir_match else ""
            
            # 提取理由
            reason_match = re.search(r'"reason":\s*"([^"]+)"', output)
            result['reason'] = reason_match.group(1) if reason_match else ""
            
            # 提取题目
            question_match = re.search(r'"question":\s*"([^"]+)"', output)
            result['question'] = question_match.group(1) if question_match else ""
            
            # 提取答案
            answer_match = re.search(r'"answer":\s*"([^"]+)"', output)
            result['answer'] = answer_match.group(1) if answer_match else ""
            
            # 提取解题过程
            solution_match = re.search(r'"solution":\s*"([^"]+)"', output)
            result['solution'] = solution_match.group(1) if solution_match else ""
            
            # 提取选项
            options = []
            for option in ['A', 'B', 'C', 'D']:
                option_match = re.search(rf'"{option}\.\s*([^"]+)"', output)
                if option_match:
                    options.append(f"{option}. {option_match.group(1)}")
            result['options'] = options
            
            return result
            
        except Exception as e:
            logger.error(f"解析Writer输出失败: {e}")
            return {"raw_output": output, "parse_error": str(e)}
    
    def _parse_solver_output(self, output: str) -> Dict[str, Any]:
        """解析Solver输出"""
        try:
            # 首先尝试JSON解析
            if output.strip().startswith('{'):
                return json.loads(output)
            
            # 检查是否被包装在代码块中
            if '```json' in output:
                # 提取JSON部分
                json_match = re.search(r'```json\s*\n(.*?)\n```', output, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                    # 处理JSON中的中文引号问题
                    json_str = json_str.replace('"', '"').replace('"', '"')
                    try:
                        result = json.loads(json_str)
                        logger.info(f"Solver JSON解析成功，scores: {result.get('scores', {})}")
                        return result
                    except json.JSONDecodeError as e:
                        logger.error(f"Solver JSON解析失败: {e}")
                        logger.error(f"JSON字符串: {json_str[:200]}...")
                        # 继续使用正则表达式解析
            
            
            # 如果不是JSON，使用正则表达式解析
            result = {}
            
            # 提取评分
            scores = {}
            score_keys = ["logic", "clarity", "guidance", "solution", "type"]
            for key in score_keys:
                match = re.search(rf'"{key}":\s*(\d)', output)
                scores[key] = int(match.group(1)) if match else 0
            result['scores'] = scores
            
            # 提取详细理由
            reasons = {}
            for key in score_keys:
                reason_match = re.search(rf'"{key}":\s*"([^"]+)"', output)
                if reason_match:
                    reasons[key] = reason_match.group(1).replace('\\n', '\n')
            result['reasons'] = reasons
            
            # 提取PASS状态
            pass_match = re.search(r'"pass":\s*"([^"]+)"', output)
            result['pass'] = pass_match.group(1) if pass_match else "PASS=0"
            
            # 提取建议
            suggestions = []
            # 查找suggestions数组
            suggestions_match = re.search(r'"suggestions":\s*\[(.*?)\]', output, re.DOTALL)
            if suggestions_match:
                suggestions_text = suggestions_match.group(1)
                # 提取数组中的字符串
                suggestion_items = re.findall(r'"([^"]+)"', suggestions_text)
                suggestions.extend(suggestion_items)
            result['suggestions'] = suggestions
            
            # 提取修正答案
            corrected_match = re.search(r'"corrected_answer":\s*"([^"]+)"', output)
            result['corrected_answer'] = corrected_match.group(1) if corrected_match else ""
            
            return result
            
        except Exception as e:
            logger.error(f"解析Solver输出失败: {e}")
            return {"raw_output": output, "parse_error": str(e)}
    
    def _parse_educator_output(self, output: str) -> Dict[str, Any]:
        """解析Educator输出"""
        try:
            # 首先尝试JSON解析
            if output.strip().startswith('{'):
                return json.loads(output)
            
            # 检查是否被包装在代码块中
            if '```json' in output:
                # 提取JSON部分
                json_match = re.search(r'```json\s*\n(.*?)\n```', output, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                    # 处理JSON中的中文引号问题
                    json_str = json_str.replace('"', '"').replace('"', '"')
                    try:
                        result = json.loads(json_str)
                        logger.info(f"Educator JSON解析成功，suggestions: {result.get('suggestions', [])}")
                        return result
                    except json.JSONDecodeError as e:
                        logger.error(f"Educator JSON解析失败: {e}")
                        logger.error(f"JSON字符串: {json_str[:200]}...")
                        # 继续使用正则表达式解析
            
            # 如果不是JSON，使用正则表达式解析
            result = {}
            
            # 提取评分
            scores = {}
            score_keys = ["knowledge", "difficulty", "competency", "organization"]
            for key in score_keys:
                match = re.search(rf'"{key}":\s*(\d)', output)
                scores[key] = int(match.group(1)) if match else 0
            result['scores'] = scores
            
            # 提取详细理由
            reasons = {}
            for key in score_keys:
                reason_match = re.search(rf'"{key}":\s*"([^"]+)"', output)
                if reason_match:
                    reasons[key] = reason_match.group(1).replace('\\n', '\n')
            result['reasons'] = reasons
            
            # 提取PASS状态
            pass_match = re.search(r'"pass":\s*"([^"]+)"', output)
            result['pass'] = pass_match.group(1) if pass_match else "PASS=0"
            
            # 提取建议
            suggestions = []
            suggestions_match = re.search(r'"suggestions":\s*\[(.*?)\]', output, re.DOTALL)
            if suggestions_match:
                suggestions_text = suggestions_match.group(1)
                # 使用更智能的方式提取字符串，处理包含中文引号的情况
                # 匹配 "..." 格式的字符串，但排除内部的中文引号
                suggestion_items = re.findall(r'"([^"]*(?:[^"]*"[^"]*)*)"', suggestions_text)
                suggestions.extend(suggestion_items)
            result['suggestions'] = suggestions
            
            return result
            
        except Exception as e:
            logger.error(f"解析Educator输出失败: {e}")
            return {"raw_output": output, "parse_error": str(e)}
    
    def _parse_rewrite_output(self, output: str) -> Dict[str, Any]:
        """解析重写输出"""
        try:
            # 首先尝试JSON解析
            if output.strip().startswith('{'):
                return json.loads(output)
            
            # 如果不是JSON，使用正则表达式解析
            result = {}
            
            # 提取题目
            question_match = re.search(r'"question":\s*"([^"]+)"', output)
            result['question'] = question_match.group(1) if question_match else ""
            
            # 提取答案
            answer_match = re.search(r'"answer":\s*"([^"]+)"', output)
            result['answer'] = answer_match.group(1) if answer_match else ""
            
            # 提取解题过程
            solution_match = re.search(r'"solution":\s*"([^"]+)"', output)
            result['solution'] = solution_match.group(1) if solution_match else ""
            
            # 提取选项（选择题）
            options = []
            for option in ['A', 'B', 'C', 'D']:
                option_match = re.search(rf'"{option}\.\s*([^"]+)"', output)
                if option_match:
                    options.append(f"{option}. {option_match.group(1)}")
            result['options'] = options
            
            return result
            
        except Exception as e:
            logger.error(f"解析重写输出失败: {e}")
            return {"raw_output": output, "parse_error": str(e)}
    
    def extract_question_content(self, writer_output: Dict[str, Any]) -> str:
        """从Writer输出中提取题目内容"""
        if 'raw_output' in writer_output:
            return writer_output['raw_output']
        
        question = writer_output.get('question', '')
        options = writer_output.get('options', [])
        answer = writer_output.get('answer', '')
        
        if not question:
            return ""
        
        # 判断题型
        if '______' in question or not options:
            # 填空题
            return f"{question}\n\n答案为：{answer}"
        else:
            # 选择题
            options_text = "\n".join(options) if options else ""
            return f"{question}\n\n{options_text}\n\n答案为{answer}"
    
    def extract_rewrite_question_content(self, rewrite_output: Dict[str, Any]) -> str:
        """从重写输出中提取题目内容"""
        if 'raw_output' in rewrite_output:
            return rewrite_output['raw_output']
        
        question = rewrite_output.get('question', '')
        options = rewrite_output.get('options', [])
        answer = rewrite_output.get('answer', '')
        
        if not question:
            return ""
        
        # 判断题型
        if '______' in question or not options:
            # 填空题
            return f"{question}\n\n答案为：{answer}"
        else:
            # 选择题
            options_text = "\n".join(options) if options else ""
            return f"{question}\n\n{options_text}\n\n答案为{answer}"
    
    def extract_solver_feedback(self, solver_output: Dict[str, Any]) -> str:
        """从Solver输出中提取反馈信息"""
        if 'raw_output' in solver_output:
            return solver_output['raw_output']
        
        pass_status = solver_output.get('pass', 'PASS=0')
        suggestions = solver_output.get('suggestions', [])
        reasons = solver_output.get('reasons', {})
        
        result = ""
        
        # 添加详细评分理由
        if reasons:
            for key, reason in reasons.items():
                result += f"{reason}\n"
        
        result += f"结论：{pass_status}\n"
        if suggestions:
            result += "修改建议：\n" + "\n".join(suggestions)
        
        return result
    
    def extract_educator_feedback(self, educator_output: Dict[str, Any]) -> str:
        """从Educator输出中提取反馈信息"""
        if 'raw_output' in educator_output:
            return educator_output['raw_output']
        
        pass_status = educator_output.get('pass', 'PASS=0')
        suggestions = educator_output.get('suggestions', [])
        reasons = educator_output.get('reasons', {})
        
        result = ""
        
        # 添加详细评分理由
        if reasons:
            for key, reason in reasons.items():
                result += f"{reason}\n"
        
        result += f"结论：{pass_status}\n"
        if suggestions:
            result += "修改建议：\n" + "\n".join(suggestions)
        
        return result
    
    def extract_corrected_answer(self, solver_output: Dict[str, Any]) -> str:
        """从Solver输出中提取修正答案"""
        if 'raw_output' in solver_output:
            return ""
        
        return solver_output.get('corrected_answer', '')
    
    def extract_scores(self, agent_output: Dict[str, Any]) -> tuple:
        """提取评分信息"""
        if 'raw_output' in agent_output:
            return 0.0, False
        
        scores = agent_output.get('scores', {})
        if not scores:
            return 0.0, False
        
        # 计算平均分
        overall_score = sum(scores.values()) / len(scores) if scores else 0.0
        
        # 检查是否通过
        pass_status = agent_output.get('pass', 'PASS=0')
        is_pass = "PASS=1" in pass_status
        
        return overall_score, is_pass
    
    def extract_writer_direction_selection(self, writer_output: Dict[str, Any]) -> Dict[str, str]:
        """从Writer输出中提取方向选择信息"""
        if 'raw_output' in writer_output:
            return {"direction": "unknown", "reason": "未使用标准化格式"}
        
        return {
            "direction": writer_output.get('dir', 'unknown'),
            "reason": writer_output.get('reason', '未提供理由')
        }

# ==================== 使用示例 ====================

def test_lightweight_parser():
    """测试轻量级解析器"""
    parser = LightweightParser()
    
    # 测试Writer输出
    writer_output = '''
    {
      "dir": "方向二",
      "reason": "图形几何类能更好地培养学生的数形结合能力",
      "question": "已知二次函数y = x² - 4x + 3的图像，下列说法正确的是：",
      "options": ["A. 顶点坐标为(2, -1)", "B. 对称轴为x = 2", "C. 与x轴的交点为(1, 0)和(3, 0)", "D. 开口向上"],
      "answer": "A"
    }
    '''
    
    result = parser.parse_agent_output('writer', writer_output)
    print("Writer解析结果:", json.dumps(result, ensure_ascii=False, indent=2))
    
    # 测试题目内容提取
    question_content = parser.extract_question_content(result)
    print("提取的题目内容:", question_content)
    
    # 测试Solver输出
    solver_output = '''
    {
      "scores": {"logic": 0, "clarity": 1, "guidance": 1, "solution": 1, "type": 1},
      "pass": "PASS=0",
      "suggestions": ["逻辑完备性：将答案更正为：B"],
      "corrected_answer": "B"
    }
    '''
    
    result = parser.parse_agent_output('solver', solver_output)
    print("Solver解析结果:", json.dumps(result, ensure_ascii=False, indent=2))
    
    # 测试反馈提取
    feedback = parser.extract_solver_feedback(result)
    print("提取的反馈:", feedback)

if __name__ == "__main__":
    test_lightweight_parser()
