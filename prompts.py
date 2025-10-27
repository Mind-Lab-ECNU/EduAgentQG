#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
集中管理各个智能体的 system prompt 与 user prompt 构造函数
"""

from typing import Dict
import logging
import re
import random

logger = logging.getLogger(__name__)


# 通用：清洗与选择参考样例（仅保留填空题/选择题，随机抽取≤3条）
def _sanitize_reference_text(text: str) -> str:
    try:
        if not isinstance(text, str):
            return str(text)
        lines = [ln.rstrip("\n") for ln in text.splitlines()]
        cleaned = []
        skip_prefixes = (
            "=== 题目参考样例 ===",
            "以下为",
            "题目参考样例",
            "参考题目样例",
        )
        for ln in lines:
            s = ln.strip()
            if not s:
                continue
            if any(s.startswith(p) for p in skip_prefixes):
                continue
            cleaned.append(ln)
        return "\n".join(cleaned).strip()
    except Exception:
        return text if isinstance(text, str) else str(text)


def _select_reference_examples(text: str, max_count: int = 3, rag_mode: str = "writer_only") -> str:
    try:
        if not isinstance(text, str):
            return str(text)
        cleaned = _sanitize_reference_text(text)
        blocks = re.split(r"\n{2,}", cleaned)
        def is_fill_in(block: str) -> bool:
            return "____" in block or "______" in block or "用______" in block
        def is_choice(block: str) -> bool:
            return bool(re.search(r"(^|\n)\s*[ABCD]\s*[\.、\)]\s*", block))
        eligible = [b.strip() for b in blocks if b.strip() and (is_fill_in(b) or is_choice(b))]
        if not eligible:
            return cleaned
        # 根据RAG模式调整参考示例数量
        actual_max_count = 1 if rag_mode == "writer" else max_count
        k = min(actual_max_count, len(eligible))
        sampled = random.sample(eligible, k)
        return "\n\n".join(sampled)
    except Exception:
        return text if isinstance(text, str) else str(text)


def build_planner_user_prompt(exercise_config: Dict, knowledge_context: str = "", use_rag: bool = True, replanning_feedback: str = "", previous_planning: str = "", rewrite_history: list = None) -> str:
    """构建Planner的提示词"""
    
    # 判断是否为重新规划模式
    is_replanning = bool(previous_planning and rewrite_history)
    
    # 基础prompt（重新规划和首次规划完全一样）
    base_prompt = (
        "你是一到九年级数学教育方面的专家。\n\n"
        "通用硬性规则：\n"
        "- 教育目标（年级/难度/知识点/核心素养）为固定约束，任何阶段不得建议或擅自修改\n"
        "- 严格遵守各函数指定的输出格式与字数要求;只输出要求的内容，不要额外解释\n"
        "- 中文回答，术语准确，逻辑清晰\n\n"
        "规划角色规则：\n"
        "- RAG仅作参考，若与教育目标冲突，必须以教育目标为准\n"
        "- 仅输出规划内容，不给出具体题目文本\n"
        "- 严禁给出任何具体题目或示例（包含具体数字/表达式/选项/填空样式等）\n\n"
        "你的任务是根据给定的教育目标制定题目设计规划方案。\n\n"
        "分析要点：\n"
        "教育目标分析：\n"
        "- 知识点覆盖要求\n"
        "- 难度适配标准\n"
        "- 素养体现方式\n\n"
        "设计策略：\n"
        "- 题型设计框架\n"
        "- 选项设计思路\n"
        "- 误区引导策略\n\n"
    )
    
    # 如果有知识检索信息，添加到prompt中（RAG参考）
    # 重新规划时不使用RAG
    if use_rag and knowledge_context and not is_replanning:
        base_prompt += f"参考信息（RAG）：\n{knowledge_context}\n\n"
    
    qtype = exercise_config.get('exercise_type', '选择题')
    # 重要提示放在输入信息前
    base_prompt += (
        "⚠️ 重要提示：如检索参考信息（RAG）与教育目标不一致，请以教育目标为准，禁止修改教育目标（包含年级、难度、知识点与素养）。\n\n"
    )

    # 输入信息（重新规划和首次规划完全一样）
    base_prompt += (
        "请从以下角度给出题目规划（控制在200字以内）：\n\n"
        "### 知识点规划\n"
        "概述关键知识点及其前后依赖关系，并指出学生可能出现的典型误区。\n\n"
        "### 学情与难度规划\n"
        "结合该年级学生的认知特点，说明本题难度的设计思路及适配策略。\n\n"
        "### 素养规划\n"
        "阐述该素养在本题中的体现方式，给出达成路径。\n\n"
        "### 出题方向（三种不同方向）\n"
        "请提供三种不同的出题方向，每种方向都要体现不同的情境、表达方式或解题思路：\n"
        "注意：只提供出题思路和方向指导，不要包含具体的数字、例子或题目内容\n"
        "方向一：[描述第一种出题思路和情境类型]\n"
        "方向二：[描述第二种出题思路和情境类型]\n"
        "方向三：[描述第三种出题思路和情境类型]\n\n"
    )
    
    # 教育目标
    base_prompt += (
        f"教育目标：知识点={exercise_config['knowledge_point']}，"
        f"年级={exercise_config['grade']}{exercise_config['grade_level']}，"
        f"难度={exercise_config['difficulty']}，"
        f"素养={exercise_config['core_competency']}，"
        f"题型={qtype}\n\n"
    )
    
    return base_prompt


def build_planner_user_prompt_with_teacher_feedback(exercise_config: Dict, knowledge_context: str = "", teacher_analysis: str = "", use_rag: bool = True) -> str:
    """构建包含教师反馈的planner用户提示"""
    base_prompt = (
        "你是一到九年级数学教育方面的专家。\n"
        "通用硬性规则：\n"
        "- 教育目标（年级/难度/知识点/核心素养）为固定约束，任何阶段不得建议或擅自修改\n"
        "- 严格遵守各函数指定的输出格式与字数要求;只输出要求的内容，不要额外解释\n"
        "- 中文回答，术语准确，逻辑清晰\n"
        "规划角色规则：\n"
        "- RAG仅作参考，若与教育目标冲突，必须以教育目标为准\n"
        "- 仅输出规划内容，不给出具体题目文本\n\n"
        "请根据知识图谱和课程标准信息，结合你的专业经验，设计合适的题目规划。\n"
        "重点关注：\n"
        "• 知识点的前置要求和相关概念\n"
        "• 该年级学生的认知水平和学习目标\n"
        "• 难度等级对应的考查重点\n"
        "• 核心素养的具体体现方式\n\n"
        "⚠️ 重要：如检索参考信息（RAG）与教育目标不一致，请以教育目标为准。\n\n"
    )
    
    # 如果有知识检索信息，添加到prompt中
    if use_rag and knowledge_context:
        base_prompt += f"{knowledge_context}\n"
    
    # 如果有教师分析，添加到prompt中
    if teacher_analysis:
        base_prompt += (
            "## 教师分析反馈\n"
            f"{teacher_analysis}\n\n"
            "请仔细阅读上述教师分析，特别关注：\n"
            "1. 问题诊断中提到的具体问题\n"
            "2. 改进建议中的关键要点\n"
            "3. 重新规划注意事项中的指导\n"
            "在本次规划中要避免之前的问题，并采纳教师的建议。\n\n"
        )
    
    qtype = exercise_config.get('exercise_type', '选择题')
    base_prompt += (
        "请用中文专业回答，逻辑清晰，控制在300字以内。\n\n"
        "请从以下角度给出题目规划：\n\n"
        "### 知识点规划\n"
        "概述关键知识点及其前后依赖关系，并指出学生可能出现的典型误区。\n\n"
        "### 学情与难度规划\n"
        "结合该年级学生的认知特点，说明本题难度的设计思路及适配策略。\n\n"
        "### 素养规划\n"
        "阐述该素养在本题中的体现方式，给出达成路径，并提示学生可能的偏差。\n\n"
    )
    # 去除与教师反馈相关的“改进措施”模块，避免无上下文的提示重复
    base_prompt += (
        f"教育目标：知识点={exercise_config['knowledge_point']}，"
        f"年级={exercise_config['grade']}{exercise_config['grade_level']}，"
        f"难度={exercise_config['difficulty']}，"
        f"素养={exercise_config['core_competency']}，"
        f"题型={qtype}\n\n"
    )
    
    return base_prompt



def build_writer_user_prompt_for_first(planner_content_clean: str, question_type: str = "选择题", use_writer_rag: bool = False, knowledge_retriever=None, exercise_config: Dict = None, rag_content: str = "", rag_mode: str = "writer_only") -> str:
    """根据题型生成不同的Writer prompt - 使用轻量级标准化格式"""
    
    # 轻量级标准化输出格式 - 生成三条题目
    if question_type == "填空题":
        format_instruction = (
            "## 输出格式（必须严格按此JSON格式）：\n"
            "[\n"
            "  {\n"
            '    "dir": "方向一",\n'
            '    "reason": "选择理由1",\n'
            '    "question": "题目内容1，用______表示填空位置",\n'
            '    "answer": "具体答案1"\n'
            "  },\n"
            "  {\n"
            '    "dir": "方向二",\n'
            '    "reason": "选择理由2",\n'
            '    "question": "题目内容2，用______表示填空位置",\n'
            '    "answer": "具体答案2"\n'
            "  },\n"
            "  {\n"
            '    "dir": "方向三",\n'
            '    "reason": "选择理由3",\n'
            '    "question": "题目内容3，用______表示填空位置",\n'
            '    "answer": "具体答案3"\n'
            "  }\n"
            "]\n\n"
        )
    
    else:  # 默认选择题
        format_instruction = (
            "## 输出格式（必须严格按此JSON格式）：\n"
            "[\n"
            "  {\n"
            '    "dir": "方向一",\n'
            '    "reason": "选择理由1",\n'
            '    "question": "题目内容1",\n'
            '    "options": ["A. 选项A1", "B. 选项B1", "C. 选项C1", "D. 选项D1"],\n'
            '    "answer": "A"\n'
            "  },\n"
            "  {\n"
            '    "dir": "方向二",\n'
            '    "reason": "选择理由2",\n'
            '    "question": "题目内容2",\n'
            '    "options": ["A. 选项A2", "B. 选项B2", "C. 选项C2", "D. 选项D2"],\n'
            '    "answer": "B"\n'
            "  },\n"
            "  {\n"
            '    "dir": "方向三",\n'
            '    "reason": "选择理由3",\n'
            '    "question": "题目内容3",\n'
            '    "options": ["A. 选项A3", "B. 选项B3", "C. 选项C3", "D. 选项D3"],\n'
            '    "answer": "C"\n'
            "  }\n"
            "]\n\n"
        )
    
    # 构建基础prompt（角色设定）
    base_prompt = (
        "你是一到九年级数学教育方面的专家。\n"
        "通用硬性规则：\n"
        "- 教育目标（年级/难度/知识点/核心素养）为固定约束，任何阶段不得建议或擅自修改\n"
        "- 严格遵守各函数指定的输出格式与字数要求;只输出要求的内容，不要额外解释\n"
        "- 中文回答，术语准确，逻辑清晰\n"
        "编写角色规则：\n"
        "- 严格按规划与题型要求编写;不得改变知识点/难度/年级/题型\n"
        "- 轻量修正模式下，只做最小必要修改以修正硬性问题（答案/提示/格式），不要大改结构\n\n"
        "你的任务是根据给定的题目规划方案编写数学题目。注意：必须生成三条题目\n\n"
        "编写要求：\n"
        "- 严格按照规划方案编写\n"
        "- 确保题目逻辑完整\n"
        "- 保持与教育目标一致\n"
        "- 从规划中提供的三种出题方向中分别选择一种进行编写，确保三条题目覆盖不同方向\n"
        "- 必须输出有效的JSON格式，不得包含markdown代码块标记\n"
        "- **重要**：answer字段的答案必须与solution字段中的最终计算结果完全一致\n\n"
    )
    
    # 清洗RAG参考文本，去除冗余前缀与标题
    def _sanitize_reference_text(text: str) -> str:
        try:
            if not isinstance(text, str):
                return str(text)
            lines = [ln.rstrip("\n") for ln in text.splitlines()]
            cleaned = []
            skip_prefixes = (
                "=== 题目参考样例 ===",
                "以下为",
                "题目参考样例",
                "参考题目样例",
            )
            for ln in lines:
                s = ln.strip()
                if not s:
                    continue
                if any(s.startswith(p) for p in skip_prefixes):
                    continue
                cleaned.append(ln)
            return "\n".join(cleaned).strip()
        except Exception:
            return text if isinstance(text, str) else str(text)

    # 选择不超过3条样例，且仅限"填空题/选择题"，随机抽取
    def _select_reference_examples_local(text: str, max_count: int = 3) -> str:
        try:
            if not isinstance(text, str):
                return str(text)
            cleaned = _sanitize_reference_text(text)
            # 按"参考样例"分割，保留分割标记
            pattern = r"(参考样例\d+.*?)(?=参考样例\d+|$)"
            matches = re.findall(pattern, cleaned, re.DOTALL)
            # 过滤掉空块和不符合条件的块
            def is_fill_in(block: str) -> bool:
                return "____" in block or "______" in block or "用______" in block
            def is_choice(block: str) -> bool:
                return bool(re.search(r"(^|\n)\s*[ABCD]\s*[\.、\)]\s*", block))
            eligible = [block.strip() for block in matches if block.strip() and (is_fill_in(block) or is_choice(block))]
            if not eligible:
                return cleaned
            # 根据RAG模式调整参考示例数量
            actual_max_count = 1 if rag_mode == "writer" else max_count
            k = min(actual_max_count, len(eligible))
            sampled = random.sample(eligible, k)
            return "\n\n".join(sampled)
        except Exception:
            return text if isinstance(text, str) else str(text)

    # 统一参考优先：若传入rag_content，则直接使用;否则按开关检索
    if rag_content:
        base_prompt += "参考题目样例（仅供结构和表达方式参考）：\n" + _select_reference_examples_local(rag_content) + "\n\n"
    elif use_writer_rag and knowledge_retriever and exercise_config:
        try:
            writer_reference = knowledge_retriever.retrieve_knowledge_for_writer(exercise_config)
            if writer_reference:
                base_prompt += "参考题目样例（仅供结构和表达方式参考）：\n" + _select_reference_examples_local(writer_reference) + "\n\n"
        except Exception as e:
            logger.warning(f"Writer RAG检索失败: {e}")
    
    base_prompt += f"{format_instruction}"

    # 将其他重要提示置于输入信息之前
    base_prompt += (
        "\n\n⚠️ 重要提示：\n"
        "- 教育目标（年级/难度/知识点/核心素养）为固定约束，禁止修改\n"
        "- 只参考样例的结构与表达方式，禁止照搬具体数字、场景或题干\n"
        "- 严格遵守输出格式与字数要求;只输出要求内容，不要额外解释\n"
        "- 从规划中提供的三种出题方向中随机选择一种进行编写\n"
        "- 输出必须是有效的JSON格式，不要包含任何其他文字\n"
    )

    # 输入信息（置于末尾）
    base_prompt += (
        f"\n输入信息：\n规划内容：{planner_content_clean}\n"
        f"题型要求：{question_type}\n\n"
        "💡 编写提示：请从规划中提供的三种出题方向中分别选择一种，按照不同方向编写三条题目。\n"
        "记住：只输出JSON格式，不要包含任何其他文字！\n"
    )
    
    return base_prompt


def _extract_key_feedback(feedback: str) -> str:
    """提取反馈中的关键信息：结论和修改建议部分"""
    if not feedback:
        return ""
    
    lines = feedback.split('\n')
    key_parts = []
    in_suggestion_section = False
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # 检查是否是结论行
        if line.startswith('结论：PASS=') or line.startswith('结论：'):
            key_parts.append(line)
        # 检查是否是修改建议行
        elif line.startswith('修改建议：'):
            key_parts.append(line)
            in_suggestion_section = True
        # 如果在修改建议部分，继续添加内容
        elif in_suggestion_section and (line.startswith('-') or line.startswith('•') or line.startswith('1.') or line.startswith('2.') or line.startswith('3.') or line.startswith('4.') or line.startswith('5.') or line.strip()):
            key_parts.append(line)
        # 如果遇到新的标题行，停止添加
        elif in_suggestion_section and (line.startswith('###') or line.startswith('##') or line.startswith('#')):
            break
    
    return '\n'.join(key_parts) if key_parts else feedback

def build_writer_user_prompt_for_rewrite(
    previous_exercise: str,
    solver_score: float,
    solver_feedback: str,
    educator_score: float,
    educator_feedback: str,
    student_feedback: str = "",
    question_type: str = "选择题",
    use_writer_rag: bool = False,
    knowledge_retriever=None,
    exercise_config: Dict = None,
    rag_content: str = "",
    rag_mode: str = "writer_only"
) -> str:
    # 根据题型选择不同的格式要求 - 生成三条题目
    if question_type == "填空题":
        format_instruction = (
            "3. 保持填空题格式，用______表示填空位置\n"
            "4. 答案为：[具体答案]\n"
            "5. 必须生成三条不同的填空题，覆盖不同的出题方向"
        )
    elif question_type == "判断题":
        format_instruction = (
            "3. 保持判断题格式，选项为A.对 B.错\n"
            "4. 答案为A或B\n"
            "5. 必须生成三条不同的判断题，覆盖不同的出题方向"
        )
    else:  # 选择题
        format_instruction = (
            "3. 保持选择题格式，包含A、B、C、D四个选项\n"
            "4. 答案为A、B、C或D\n"
            "5. 必须生成三条不同的选择题，覆盖不同的出题方向"
        )
    
    # 1. 角色设定
    prompt = (
        "你是一到九年级数学教育方面的专家。\n"
        "通用硬性规则：\n"
        "- 教育目标（年级/难度/知识点/核心素养）为固定约束，任何阶段不得建议或擅自修改\n"
        "- 严格遵守各函数指定的输出格式与字数要求;只输出要求的内容，不要额外解释\n"
        "- 中文回答，术语准确，逻辑清晰\n"
        "编写角色规则：\n"
        "- 严格按规划与题型要求编写;不得改变知识点/难度/年级/题型\n"
        "你的任务是根据反馈意见优化数学题目。\n\n"
        "重写原则：\n"
        "- 优先在原题目基础上进行修改和补充，而不是完全替换成新题目\n"
        "- 保持题目的连贯性和一致性\n\n"
        "⚠️ 重要：必须严格遵守修改意见：\n"
        "- 必须逐条执行所有修改建议，不得遗漏任何一条\n"
        "- 如果修改建议之间存在冲突，以合规与教育目标优先\n"
        "- 确保最终题目与各项建议逐条对应落实\n"
        "- 不得忽略或选择性执行修改建议\n\n"
        "优化策略：\n"
        "- 解决已确定的弱点\n"
        "- 保持现有优势\n"
        "- 增强与目标的一致性\n\n"
    )

    # 2. RAG参考
    if use_writer_rag and knowledge_retriever and exercise_config:
        try:
            writer_reference = knowledge_retriever.retrieve_knowledge_for_writer(exercise_config)
            if writer_reference:  # 只要有参考信息就添加，不再检查"未找到匹配的题目样例"
                # 复用与首次写题相同的选择逻辑
                prompt += "参考题目样例（仅供结构和表达方式参考）：\n" + _select_reference_examples(writer_reference, rag_mode=rag_mode) + "\n\n"
        except Exception as e:
            logger.warning(f"Writer RAG检索失败: {e}")
    elif rag_content:  # 如果有传入的RAG内容，直接使用
        prompt += "参考题目样例（仅供结构和表达方式参考）：\n" + _select_reference_examples(rag_content, rag_mode=rag_mode) + "\n\n"

    # 3. 重要提示
    prompt += (
        "⚠️ 重要提示：\n"
        "- 教育目标（年级/难度/知识点/核心素养）为固定约束，禁止修改\n"
        "- 仅做最小必要修改以修正硬性问题（答案/提示/格式），避免大改结构\n"
        "- 只输出要求内容，禁止加入学生提示或解题指导\n"
        "- 必须完全且完整地执行修改建议，不得遗漏任何一条;若建议之间存在冲突，以合规与教育目标优先，确保最终题目与各项建议逐条对应落实\n\n"
    )

    # 若为二值评估（通过反馈文本检测），强调0/1含义
    is_binary_eval = ("PASS=" in (solver_feedback or "")) or ("PASS=" in (educator_feedback or "")) or ("###二值评分" in (solver_feedback or ""))
    if is_binary_eval:
        prompt += (
            "[评估标注说明]本轮为二值评估：0=不通过，1=通过。若任一维度为0或结论PASS=0，则必须针对对应问题点进行最小必要修改。\n\n"
        )

    
    
    if student_feedback:
        prompt += f"学生建议：\n{student_feedback}\n\n"
        # 轻量级重写模式说明：当存在来自教师质检的问题点时，仅做最小必要修改
        prompt += (
            "[轻量修正模式]请在保留原有题干/结构/题型不变的前提下，仅针对上述问题做最小必要修改：\n"
            "- 若答案被判定不一致/错误：仅修正答案本身，必要时微调题干中的数字或措辞使之自洽\n"
            "- 若存在对学生提示/解题指导：删除提示性语言，不改变考查目标\n"
            "- 若题型格式不规范：补足下划线/选项标号/判断选项等格式性元素\n"
            "- 若知识点不匹配：必须按照修改建议调整题目内容以匹配正确的知识点\n"
            "- 若难度不符合要求：必须按照修改建议调整题目难度\n"
            "- 严禁：重写整题或更换题型，加入新的设问（除非修改建议明确要求）\n"
            "- 请最小化改动范围，除纠正硬性问题外不要改动无关内容\n"
            "- **答案修正规则**：如果建议中明确给出了需要修改的正确答案（如'将答案修改为[正确答案]'），且题目内容本身没有需要改变的地方，则直接使用建议中的正确答案\n"
            "- **强制要求**：必须完全且完整地执行修改建议，不得遗漏任何一条；若建议之间存在冲突，以合规与教育目标优先，确保最终题目与各项建议逐条对应落实\n\n"
        )
    else:
        # 当没有学生反馈时，也要强调遵守修改建议
        prompt += (
            "[重写模式]请根据修改建议优化题目：\n"
            "- 若知识点不匹配：必须按照修改建议调整题目内容以匹配正确的知识点\n"
            "- 若难度不符合要求：必须按照修改建议调整题目难度\n"
            "- 若需要增加运算复杂性：按照建议增加小数位数、整数因子或计算步骤\n"
            "- 若需要引入逆向思维：按照建议设计需要逆向分析的情境\n"
            "- 若需要多步关联计算：按照建议设计多步关联的计算过程\n"
            "- **强制要求**：必须完全且完整地执行修改建议，不得遗漏任何一条；若建议之间存在冲突，以合规与教育目标优先，确保最终题目与各项建议逐条对应落实\n\n"
        )
    
    # 根据题型选择不同的轻量级输出格式 - 生成三条题目
    if question_type == "填空题":
        rewrite_format = (
            "## 输出格式（必须严格按此JSON格式）：\n"
            "[\n"
            "  {\n"
            '    "question": "题目内容1，用______表示填空位置",\n'
            '    "answer": "具体答案1"\n'
            "  },\n"
            "  {\n"
            '    "question": "题目内容2，用______表示填空位置",\n'
            '    "answer": "具体答案2"\n'
            "  },\n"
            "  {\n"
            '    "question": "题目内容3，用______表示填空位置",\n'
            '    "answer": "具体答案3"\n'
            "  }\n"
            "]\n\n"
        )
    else:  # 默认选择题
        rewrite_format = (
            "## 输出格式（必须严格按此JSON格式）：\n"
            "[\n"
            "  {\n"
            '    "question": "题目内容1",\n'
            '    "options": ["A. 选项A1", "B. 选项B1", "C. 选项C1", "D. 选项D1"],\n'
            '    "answer": "A"\n'
            "  },\n"
            "  {\n"
            '    "question": "题目内容2",\n'
            '    "options": ["A. 选项A2", "B. 选项B2", "C. 选项C2", "D. 选项D2"],\n'
            '    "answer": "B"\n'
            "  },\n"
            "  {\n"
            '    "question": "题目内容3",\n'
            '    "options": ["A. 选项A3", "B. 选项B3", "C. 选项C3", "D. 选项D3"],\n'
            '    "answer": "C"\n'
            "  }\n"
            "]\n\n"
        )

    prompt += (
        f"重写要求：\n"
        f"1. 严格按照修改建议调整题目内容\n"
        f"2. 题干与选项清晰、准确\n"
        f"3. 优先在原题目基础上修改，除非明确要求完全重写\n"
        f"4. 保持原题目的基本结构和情境，只针对问题点进行精准修改\n"
        f"5. 题型要求：{question_type}\n"
        f"6. 请勿直接照搬参考样例，只参考结构和表达方式\n"
        f"7. 必须输出有效的JSON格式，不要包含任何其他文字\n"
        f"8. **严格遵守修改建议**：必须逐条执行所有修改建议，不得遗漏任何一条\n"
        f"9. **难度调整**：如果修改建议要求增加难度，必须按照建议增加运算复杂性、引入逆向思维或多步关联计算\n"
        f"10. **知识点匹配**：如果修改建议要求调整知识点，必须严格按照建议调整题目内容\n"
        f"11. **必须生成三条题目**：基于修改建议生成三条不同的题目，覆盖不同的出题方向\n\n"
        f"{rewrite_format}"
    )

    # 4. 输入信息（待重写的题目和修改意见）
    # 若来自教师质检（包含PARSED_CHECK或###检查结果），仅使用checker输出与待重写题目
    checker_only = bool(student_feedback) and ("PARSED_CHECK=" in student_feedback or "###检查结果" in student_feedback)
    if checker_only:
        prompt += (
            "[待重写的题目]\n"
            f"{previous_exercise}\n\n"
            f"教师质检输出（供最小必要修改参考）：\n{student_feedback}\n\n"
            "💡 重写提示：请根据反馈意见精准修改题目，解决具体问题。\n"
        )
    else:
        if is_binary_eval:
            # 提取关键反馈信息
            solver_key_feedback = _extract_key_feedback(solver_feedback)
            educator_key_feedback = _extract_key_feedback(educator_feedback)
            
            prompt += (
                "[待重写的题目]\n"
                f"{previous_exercise}\n\n"
                f"修改意见：\n"
                f"Solver反馈：\n{solver_key_feedback}\n\n"
                f"Educator反馈：\n{educator_key_feedback}\n\n"
                "⚠️ 重要：必须严格按照上述修改意见进行重写，特别是难度和知识点的调整要求！\n\n"
            )
        else:
            # 提取关键反馈信息
            solver_key_feedback = _extract_key_feedback(solver_feedback)
            educator_key_feedback = _extract_key_feedback(educator_feedback)
            
            prompt += (
                "[待重写的题目]\n"
                f"{previous_exercise}\n\n"
                f"修改意见：\n"
                f"Solver评分：{solver_score}/10\n{solver_key_feedback}\n\n"
                f"Educator评分：{educator_score}/10\n{educator_key_feedback}\n\n"
                "⚠️ 重要：必须严格按照上述修改意见进行重写，特别是难度和知识点的调整要求！\n"
                "💡 重写提示：请根据反馈意见精准修改题目，解决具体问题。\n"
            )
    
    return prompt


def build_solver_user_prompt(writer_content: str, student_content: str, exercise_config: Dict) -> str:
    goals = (
        f"教育目标/约束：知识点={exercise_config['knowledge_point']};难度={exercise_config['difficulty']};"
        f"年级={exercise_config['grade']}{exercise_config['grade_level']};素养={exercise_config['core_competency']};"
        f"题型={exercise_config['exercise_type']}"
    )
    return (
        """你是一到九年级数学教育方面的专家。\n"
        "通用硬性规则：\n"
        "- 教育目标（年级/难度/知识点/核心素养）为固定约束，任何阶段不得建议或擅自修改\n"
        "- 严格遵守各函数指定的输出格式与字数要求;只输出要求的内容，不要额外解释\n"
        "- 中文回答，术语准确，逻辑清晰\n"
        "评分角色（Solver）规则：\n"
        "- 按指定维度与权重评分，并给出对应理由;禁止给学生提示\n"
        "- 若题型不符或答案明显错误，按规则降低或限制分数\n"
        "- 只输出评分模板要求的内容，且仅一个评分结果\n\n"
        "你的任务是对三条数学题目进行质量评分和排名。\n\n"
        "评分维度：\n"
        "- 逻辑完备性：条件是否充分，推理链是否自洽\n"
        "- 表述清晰度：题干是否容易被学生误解\n"
        "- 误区引导性：能否暴露常见错误，体现教学价值\n"
        "- 解法适切性：解法是否合理不过度繁琐\n"
        "- 题型符合性：题目形式是否与目标题型一致\n\n"
        "评分标准，并输出为什么打这个分数：\n"
        "1) 逻辑完备性（权重30%）：条件是否充分？推理链是否自洽？是否存在歧义/多解？\n"
        "   - 评分时请具体说明给分和扣分原因：题目条件是否完整、推理过程是否合理、是否存在多种解法或歧义\n"
        "2) 表述清晰度（权重20%）：题干是否容易被学生误解？\n"
        "   - 评分时请具体说明给分和扣分原因：语言表达是否清晰、是否存在容易误解的表述、术语使用是否准确\n"
        "3) 误区引导性（权重20%）：能否暴露/避免常见错误，体现教学价值？\n"
        "   - 评分时请具体说明给分和扣分原因：选项设计是否体现常见误区、是否有助于学生理解概念、教学价值如何\n"
        "4) 解法适切性（权重30%）：解法是否合理不过度繁琐？是否无需过多技巧/特殊方法？\n"
        "   - 评分时请具体说明给分和扣分原因：解题方法是否适合目标年级、计算复杂度是否合理、是否需要特殊技巧\n"
        "5) 题型符合性（0或1）：若生成题目形式与目标题型一致则为1，否则为0。\n"
        "   - 选择题：应包含清晰可辨的选项（如A/B/C/D）。\n"
        "   - 填空题：应包含空格/下划线等待填要素（如____）。\n"
        "   - 判断题：应为对/错型或等价二选一判断形式。\n"
        "   - 评分时请具体说明给分和扣分原因：题目格式是否符合目标题型要求\n\n"
        "强制核验：答案正确性（不计入权重，仅作为硬性约束）。若你能在不额外查找外部资料的前提下判定“题目给出的答案/标准答案”明显错误或与题干不一致，则将总分S限制为≤3，并在输出中给出理由。\n\n"
        "计算公式：S = 0.3×逻辑完备性 + 0.2×表述清晰度 + 0.2×误区引导性 + 0.3×解法适切性\n"
        "如果题型符合性=0，则直接判定S<6。\n\n"
        f"教育目标：{goals}\n\n"
        f"题目内容：{writer_content}\n\n"
        "注意：请对上述三条题目分别进行评分和排名。\n\n"
        "输出格式严格如下（必须严格按此JSON格式）：\n"
        "{\n"
        '  "question1": {\n'
        '    "pass": 0,\n'
        '    "suggestion": xxxxxx,\n'
        '    "rank": 0\n'
        "  },\n"
        '  "question2": {\n'
        '    "pass": 1,\n'
        '    "suggestion": xxxxxx,\n'
        '    "rank": 1\n'
        "  },\n"
        '  "question3": {\n'
        '    "pass": 1,\n'
        '    "suggestion": xxxxxx,\n'
        '    "rank": 2\n'
        "  }\n"
        "}\n\n"
        "评分说明：\n"
        "- pass: 0或1，表示该题目是否通过质量要求\n"
        "- suggestion: 修改建议数组，基于评分维度与扣分理由提出2-3条可操作改进\n"
        "- rank: 0/1/2，在三条题目中的排名，0=最差，1=中等，2=最好\n"
        "- 必须确保RANK值0、1、2各出现一次\n"
        "- 修改建议应针对题干措辞、条件充分性、选项设计或解法适切性等具体问题\n"
        "- 不得建议调整年级/难度/知识点；不得给学生提示"'''"""
    )


def build_educator_user_prompt(writer_content: str, student_content: str, exercise_config: Dict) -> str:
    goals = (
        f"教育目标/约束：知识点={exercise_config['knowledge_point']};难度={exercise_config['difficulty']};"
        f"年级={exercise_config['grade']}{exercise_config['grade_level']};素养={exercise_config['core_competency']};"
        f"题型={exercise_config['exercise_type']}"
    )
    return (
    "你是一到九年级数学教育方面的专家。\n"
    "通用硬性规则：\n"
    "- 教育目标（年级/难度/知识点/核心素养）为固定约束，任何阶段不得建议或擅自修改\n"
    "- 严格遵守各函数指定的输出格式与字数要求;只输出要求的内容，不要额外解释\n"
    "- 中文回答，术语准确，逻辑清晰\n\n"
    "评分角色（Educator）规则：\n"
    "- 先明确目标难度与你的实际判断，再按规则给分;禁止建议修改年级/知识点\n"
    "- 只输出评分模板要求的内容，且仅一个评分结果\n\n"
    "你的任务是对三条数学题目进行教育价值评估和排名。\n\n"
    "评分维度：\n"
    "- 知识点契合度：是否准确覆盖目标知识点\n"
    "- 难度匹配度：难度是否与年级和教学目标匹配\n"
    "- 核心素养体现度：是否有效体现目标核心素养\n"
    "- 教学组织适切性：题型和教学组织是否恰当\n\n"
    "评分标准：\n"
    "1) 知识点契合度（权重25%）：是否准确覆盖目标知识点？\n"
    "   - 评分时请具体说明给分和扣分原因：题目是否准确考查了目标知识点、知识点覆盖是否全面、是否偏离了核心内容\n"
    f"2) 难度匹配度（权重25%）：难度是否与年级和教学目标中需求的难度匹配？\n"
    f"   - 评分时请具体说明给分和扣分原因：先明确 目标难度=[{exercise_config['difficulty']}]; 你的判断=[实际难度]; "
    "依据包括：解题步数、抽象程度、是否多步骤/综合知识点、是否需要策略选择等。\n"
    "   - 分数规则（难度匹配必须严格按照这个给分，也就是这一项，只有10 5 2 1这几个分数）：若实际难度等于目标一档，本项=10;若实际难度低于目标一档（例：目标=难，实际=中），本项=5;低于两档（例：目标=难，实际=易），本项=2;高于目标一档（例：目标=中，实际=难），本项=5;无法判断给1并说明依据。\n"
    "3) 核心素养体现度（权重25%）：是否有效体现目标核心素养？\n"
    "   - 评分时请具体说明给分和扣分原因：题目是否体现了数学抽象、逻辑推理、数学运算、直观想象、数据分析等核心素养\n"
    "4) 教学组织适切性（权重25%）：题型和教学组织是否恰当？\n"
    "   - 评分时请具体说明给分和扣分原因：题型选择是否合适、题目结构是否合理、是否符合教学规律\n\n"
    "计算公式：E = 0.25×知识点契合度 + 0.25×难度匹配度 + 0.25×核心素养体现度 + 0.25×教学组织适切性\n\n"
    f"教育目标：{goals}\n\n"
    f"题目内容：{writer_content}\n\n"
    "注意：请对上述三条题目分别进行教育价值评估和排名。\n\n"
    "输出格式严格如下（必须严格按此JSON格式）：\n"
    "{\n"
    '  "question1": {\n'
    '    "pass": 0,\n'
    '    "suggestion": xxxxxx,\n'
    '    "rank": 0\n'
    "  },\n"
    '  "question2": {\n'
    '    "pass": 1,\n'
    '    "suggestion": xxxxxx,\n'
    '    "rank": 1\n'
    "  },\n"
    '  "question3": {\n'
    '    "pass": 1,\n'
    '    "suggestion": xxxxxx,\n'
    '    "rank": 2\n'
    "  }\n"
    "}\n\n"
    "评分说明：\n"
    "- pass: 0或1，表示该题目是否通过教育价值要求\n"
    "- suggestion: 修改建议数组，基于评分维度与扣分理由提出2-3条可操作改进\n"
    "- rank: 0/1/2，在三条题目中的排名，0=最差，1=中等，2=最好\n"
    "- 必须确保RANK值0、1、2各出现一次\n"
    "- 修改建议应针对知识点契合、难度匹配判定依据、素养体现或教学组织等具体问题\n"
    "- 仅当你已在难度匹配度中明确判定不匹配且给出理由时，方可建议调整难度\n"
    "- 若难度匹配，请不要讨论或建议调整难度\n"
    "- 禁止建议调整年级或知识点；不要给出学生提示"
    )



def build_teacher_analysis_prompt(rewrite_history: list, exercise_config: dict, last_planner_content: str = "") -> str:
    return ""



def build_teacher_checker_prompt(writer_content: str, exercise_config: Dict) -> str:
    """教师最终检查：先完整解题（含步骤与答案），再做合规性核查并给出0/1判定。
    CHECK=1 表示通过;CHECK=0 表示不通过，并给出精炼问题点。
    """
    question_type = exercise_config.get("exercise_type", "选择题")
    checks = (
        "你是一到九年级数学教育方面的专家，担任最终质检员。\n"
        "通用硬性规则：\n"
        "- 教育目标（年级/难度/知识点/核心素养）为固定约束，任何阶段不得建议或擅自修改\n"
        "- 严格遵守各函数指定的输出格式与字数要求;只输出要求的内容，不要额外解释\n"
        "- 中文回答，术语准确，逻辑清晰\n"
        "最终质检（Teacher Check）规则：\n"
        "- 第一步：先独立完成题目的解答，给出清晰、严谨、分步的“解题过程”和最终答案\n"
        "- 第二步：做二值合规检查（格式/提示/答案/明显格式错误）;输出CHECK=0或1\n"
        "- 不对题目做评分或大改建议\n\n"
        "你的任务：先解题，再做检查，并给出二值判定（通过=1/不通过=0）。\n\n"
        "必须检查（逐项核验）：\n"
        f"- 题型格式是否正确（当前题型：{question_type}）：\n  选择题应含A/B/C/D四项;填空题应有下划线______;判断题应为A.对 B.错\n"
        "- 是否存在对学生的提示/解题指导性语言（如“提示/思路/可用公式/请验根”等），一经发现应判为不通过\n"
        "- 是否给出并标明答案，且答案与题干要求一致，即给出的答案可以完全正确解答题目（若无法确定对错，请说明“无法判断”并判错）\n"
        "- 填空题是否具备考查意义：空格设置是否必要、答案是否唯一明确、无无意义的空（若不满足则判错）\n"
        "- 文本是否包含明显格式错误（标题缺失、乱码、空题干等）\n\n"
        "- 没有除了上述之外的任何额外信息\n\n"
        "输出格式严格如下（仅使用如下键值行，便于程序解析）：\n"
        "###解题过程\n"
        "[逐步推理与计算，写出关键步骤与最终答案]\n\n"
        "CHECK=0或1\n"
        "ONLY_ANSWER_ERROR=0或1  # 若仅答案错误且题干/选项/格式均无问题，则为1\n"
        "CORRECTED_ANSWER=（若ONLY_ANSWER_ERROR=1，填写更正后的唯一标准答案，如A或具体数值/表达式）\n"
        "REASON=（若CHECK=0，简要说明最关键的问题；若CHECK=1，简要说明通过理由）\n\n"
        "注意：\n- 不要修改题目，只做检查。\n- 不要建议改变教育目标（年级/难度/知识点/素养）。\n- 必须包含行：CHECK=、ONLY_ANSWER_ERROR=；当ONLY_ANSWER_ERROR=1时必须包含行：CORRECTED_ANSWER=。\n\n"
        
    )
    return (
        checks
        + f"教育目标：知识点={exercise_config.get('knowledge_point','')};难度={exercise_config.get('difficulty','')};年级={exercise_config.get('grade','')}{exercise_config.get('grade_level','')};题型={question_type}\n\n"
        + f"题目内容：{writer_content}\n"
    )


# ==================== 二值评估版 Prompt（0/1） ====================

def build_solver_user_prompt_binary(writer_content: str, student_content: str, exercise_config: Dict) -> str:
    goals = (
        f"教育目标/约束：知识点={exercise_config['knowledge_point']}；难度={exercise_config['difficulty']}；"
        f"年级={exercise_config['grade']}{exercise_config['grade_level']}；素养={exercise_config['core_competency']}；"
        f"题型={exercise_config['exercise_type']}"
    )
    return (
        "你是一到九年级数学教育方面的专家。\n"
        "通用硬性规则：\n"
        "- 教育目标（年级/难度/知识点/核心素养）为固定约束，任何阶段不得建议或擅自修改\n"
        "- 严格遵守输出格式;只输出要求内容，不要额外解释\n"
        "- 中文回答，术语准确，逻辑清晰\n\n"
        "评分角色（Solver，二值评估）规则：\n"
        "- 仅按以下维度逐项判定0或1：全部为1才视为通过\n"
        "- 若任一项为0，必须给出修改建议，且仅基于对应维度的扣分原因\n"
        "- 禁止给学生提示;禁止建议调整年级/难度/知识点\n\n"
        "维度（逐项0/1）：\n"
        "1) 逻辑完备性：条件是否充分，推理链是否自洽\n"
        "2) 表述清晰度：是否易于学生理解，无歧义\n"
        "3) 误区引导性：是否体现常见误区的暴露/辨析\n"
        "4) 解法适切性：是否适配年级且不过度繁琐\n"
        "5) 题型符合性：题目形式与目标题型一致\n\n"
        f"教育目标：{goals}\n\n"
        f"题目内容：{writer_content}\n\n"
        "输出格式严格如下（仅此结构）：\n"
        "###二值评分\n"
        "逻辑完备性：0或1 - [简要理由]\n"
        "表述清晰度：0或1 - [简要理由]\n"
        "误区引导性：0或1 - [简要理由]\n"
        "解法适切性：0或1 - [简要理由]\n"
        "题型符合性：0或1 - [简要理由]\n"
        "结论：PASS=0或1\n"
        "修改建议：若PASS=0，给出2-3条，逐条标注[对应维度/扣分点→具体改法→预期影响]；仅基于上述维度与扣分理由；不得引入新标准；不得建议调整年级/难度/知识点；不得给学生提示。\n"
    )


def build_educator_user_prompt_binary(writer_content: str, student_content: str, exercise_config: Dict) -> str:
    """构建Educator的二值评估提示词"""
    
    # 构建教育目标字符串
    goals = (
        f"教育目标/约束：知识点={exercise_config['knowledge_point']}；"
        f"难度={exercise_config['difficulty']}；"
        f"年级={exercise_config['grade']}{exercise_config['grade_level']}；"
        f"素养={exercise_config['core_competency']}；"
        f"题型={exercise_config['exercise_type']}"
    )
    
    # 构建提示词
    prompt = (
        "你是一到九年级数学教育方面的专家。\n\n"
        
        "通用硬性规则：\n"
        "- 教育目标（年级/难度/知识点/核心素养）为固定约束，任何阶段不得建议或擅自修改\n"
        "- 严格遵守输出格式；只输出要求内容，不要额外解释\n"
        "- 中文回答，术语准确，逻辑清晰\n\n"
        
        "评分角色（Educator，二值评估）规则：\n"
        "- 仅按以下维度逐项判定0或1：全部为1才视为通过\n"
        "- 若任一项为0，必须给出修改建议；建议仅基于对应维度的扣分原因\n"
        "- 禁止建议调整年级或知识点；关于难度的建议仅当\"难度匹配度=0\"且已给出依据\n\n"
        
        "⚠️ 重要：难度判断规则：\n"
        "- 在判断难度时，必须假定题目给出的答案是正确且符合题目要求的\n"
        "- 不要因为答案错误而影响难度判断，答案正确性由Solver负责评估\n"
        "- 专注于评估题目本身的难度层次（如计算复杂度、思维层次、知识点综合程度等）\n"
        "- 即使答案有误，也要基于题目内容本身来判断其难度等级\n\n"
        
        "维度（逐项0/1）：\n"
        "1) 知识点契合度：是否准确覆盖目标知识点\n"
        "2) 难度匹配度：是否与目标难度匹配（需先判定你的实际难度；注意：必须假定题目答案正确，仅基于题目内容本身判断难度）\n"
        "3) 核心素养体现度：是否体现目标核心素养\n"
        "4) 教学组织适切性：题型与组织是否恰当\n\n"
        
        f"教育目标：{goals}\n\n"
        f"题目内容：{writer_content}\n\n"
        
        "输出格式严格如下（仅此结构）：\n"
        "###二值教育评分\n"
        "知识点契合度：0或1 - [简要理由]\n"
        "难度匹配度：0或1 - [先给出\"目标难度=[X]；你的判断=[易/中/难/过难]\"，再给出理由（注意：必须假定题目答案正确，仅基于题目内容本身判断难度）]\n"
        "核心素养体现度：0或1 - [简要理由]\n"
        "教学组织适切性：0或1 - [简要理由]\n"
        "结论：PASS=0或1\n"
        "修改建议：若PASS=0，给出2-3条，逐条标注[对应维度/扣分点→具体改法→预期影响]；仅基于上述维度与扣分理由；仅当\"难度匹配度=0\"且已给出依据时，可建议\"调整难度\"；禁止建议调整年级或知识点；不得给学生提示。\n"
    )
    
    return prompt
