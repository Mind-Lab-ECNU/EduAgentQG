#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Dict
import re
import random


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


def _format_three_questions_content(writer_content) -> str:
    """格式化三条题目内容，支持数组和字符串输入"""
    if isinstance(writer_content, list):
        # 如果是数组，格式化为三条题目
        content = "题目内容：\n"
        for i, question in enumerate(writer_content, 1):
            if isinstance(question, dict):
                # 如果是字典，提取题目内容
                question_text = question.get('question', '')
                options = question.get('options', [])
                answer = question.get('answer', '')
                
                content += f"\n题目{i}：\n{question_text}\n"
                if options:
                    for option in options:
                        content += f"{option}\n"
                if answer:
                    content += f"答案为{answer}\n"
            else:
                # 如果是字符串，直接使用
                content += f"\n题目{i}：\n{question}\n"
        return content
    else:
        # 如果是字符串，直接返回
        return f"题目内容：{writer_content}\n\n"

def _select_reference_examples(text: str, max_count: int = 3) -> str:
    try:
        if not isinstance(text, str):
            return str(text)
        cleaned = _sanitize_reference_text(text)
        # 按"参考样例X"块状提取或用空行分块回退
        pattern = r"(参考样例\d+.*?)(?=参考样例\d+|$)"
        matches = re.findall(pattern, cleaned, re.DOTALL)
        blocks = matches if matches else re.split(r"\n{2,}", cleaned)
        def is_fill_in(block: str) -> bool:
            return "____" in block or "______" in block or "用______" in block
        def is_choice(block: str) -> bool:
            return bool(re.search(r"(^|\n)\s*[ABCD]\s*[\.、\)]\s*", block))
        eligible = [b.strip() for b in blocks if b.strip() and (is_fill_in(b) or is_choice(b))]
        if not eligible:
            return cleaned
        k = min(max_count, len(eligible))
        sampled = random.sample(eligible, k)
        return "\n\n".join(sampled)
    except Exception:
        return text if isinstance(text, str) else str(text)


def build_solver_user_prompt_binary(writer_content: str, student_content: str, exercise_config: Dict, reference_examples: str = "", solution: str = "") -> str:
    goals = (
        f"教育目标/约束：知识点={exercise_config['knowledge_point']}；难度={exercise_config['difficulty']}；"
        f"年级={exercise_config['grade']}{exercise_config['grade_level']}；素养={exercise_config['core_competency']}；"
        f"题型={exercise_config['exercise_type']}"
    )

    # 1. 角色设定
    prompt = (
        "你是一到九年级数学教育方面的专家。\n"
        "通用硬性规则：\n"
        "- 教育目标（年级/难度/知识点/核心素养）为固定约束，不得擅自修改\n"
        "- 严格遵守输出格式，只输出要求内容，不要额外解释\n"
        "- 中文回答，术语准确，逻辑清晰\n\n"
        "评分角色（Solver，二值评估）规则：\n"
        "- 专注于题目逻辑性、表述清晰度、解法合理性、题型符合性\n"
        "- 不评估知识点契合度、难度匹配度、核心素养体现度（这些由Educator负责）\n"
        "- 每个维度逐项判定0或1，全部为1才算通过\n"
        "- 若任一维度为0，必须给出修改建议，仅基于对应扣分原因\n"
        "- 禁止给学生提示，不得调整年级/难度/知识点/题型\n"
        "- 严禁修改题目结构（如选项数量、形式等）\n\n"
        "⚠️ 答案错误处理规则：\n"
        "- 若发现答案错误，修改建议中必须明确正确答案，格式：'将答案修改为[正确答案]'\n"
        "- 在corrected_answer字段中同时提供正确答案\n"
        "- 只有题目存在逻辑问题或表述不清时，才可提出修改建议\n\n"
    )

    # 2. RAG参考
    if reference_examples:
        prompt += (
            "参考题目样例（仅供结构和表达方式参考）：\n"
            + _select_reference_examples(reference_examples)
            + "\n\n"
        )

    # 3. 评分维度和输出格式
    common_dims = (
        "1) 逻辑完备性：条件充分，推理链自洽，符号完整（如度°、三角形△等），前后文对应\n"
        "2) 表述清晰度：易于理解，无歧义，题目中不出现答案\n"
        "3) 解法适切性：步骤合理、清晰，不过度繁琐，答案正确性\n"
        "4) 题型符合性：题目形式与题型要求一致\n"
    )

    qtype = exercise_config.get('exercise_type', '选择题')
    if qtype == '选择题':
        dims = f"维度（逐项0/1）：\n{common_dims}5) 误区引导性：体现常见误区暴露/辨析，选项设计合理\n\n"
    else:
        dims = f"维度（逐项0/1）：\n{common_dims}\n"

    # JSON输出模板
    outfmt = (
        "## 输出格式（必须严格按此JSON格式）：\n"
        "{\n"
        '  "question1": {\n'
        '    "pass": 0,\n'
        '    "suggestion": ["修改建议1", "修改建议2"],\n'
        '    "rank": 0\n'
        "  },\n"
        '  "question2": {\n'
        '    "pass": 1,\n'
        '    "suggestion": ["修改建议1", "修改建议2"],\n'
        '    "rank": 1\n'
        "  },\n"
        '  "question3": {\n'
        '    "pass": 1,\n'
        '    "suggestion": ["修改建议1", "修改建议2"],\n'
        '    "rank": 2\n'
        "  }\n"
        "}\n\n"
        "评分说明：\n"
        "- pass: 0或1，任一维度为0则整体为0\n"
        "- suggestion: 针对扣分原因提出2-3条可操作改进\n"
        "- rank: 0/1/2，分别代表三题中最差/中等/最好，必须各出现一次\n"
        "- 修改建议不得引入新标准，不得给学生提示，不得改变年级/难度/知识点/题型\n"
    )

    prompt += (
        dims +
        outfmt +
        f"教育目标：{goals}\n\n" +
        _format_three_questions_content(writer_content) +
        "注意：请对三条题目分别进行评分和排名。\n\n"
    )

    return prompt



def build_educator_user_prompt_binary(writer_content: str, student_content: str, exercise_config: Dict, reference_examples: str = "") -> str:
    goals = (
        f"教育目标/约束：知识点={exercise_config['knowledge_point']}；难度={exercise_config['difficulty']}；"
        f"年级={exercise_config['grade']}{exercise_config['grade_level']}；素养={exercise_config['core_competency']}；"
        f"题型={exercise_config['exercise_type']}"
    )

    # 1. 角色设定
    prompt = (
        "你是一到九年级数学教育方面的专家。\n"
        "通用硬性规则：\n"
        "- 教育目标（年级/难度/知识点/核心素养）为固定约束，任何阶段不得擅自修改\n"
        "- 严格遵守输出格式；只输出要求内容，不要额外解释\n"
        "- 中文回答，术语准确，逻辑清晰\n\n"
        "评分角色（Educator，二值评估）规则：\n"
        "- 专注于教育目标的匹配度：知识点契合度、难度匹配度、核心素养体现度、教学组织适切性\n"
        "- 不评估题目本身的逻辑性、表述清晰度、解法合理性（这些由Solver负责）\n"
        "- 若任一维度为0，必须给出修改建议，仅基于扣分原因\n"
        "- 禁止调整年级/难度/知识点/题型，不得给学生提示\n"
        "- 严禁修改题目结构（如选项数量、形式），题型必须保持一致\n\n"
        "⚠️ PASS判定硬性规则：\n"
        "- 以下四个维度（知识点契合度、难度匹配度、核心素养体现度、教学组织适切性）任一为0，则pass必须为0，禁止给1\n"
        "- 若难度判断中‘你的判断’≠‘目标难度’，则该维度为0，且整体pass=0\n\n"
        "⚠️ 难度判断规则：\n"
        "- 结合给出的三个参考题目进行逐一对比，判定本题的实际难度层次\n"
        "- 维度包括计算复杂度、思维层次、知识点综合程度等\n"
        "- 输出必须包含：目标难度=；你的判断=；并说明与每个参考示例的差异\n\n"
    )

    # 2. RAG参考
    if reference_examples:
        prompt += (
            "参考题目样例（用于帮助判断难度）：\n"
            + _select_reference_examples(reference_examples)
            + "\n\n"
        )

    # 3. 评分维度与输出格式
    prompt += (
        "维度（逐项0/1）：\n"
        "1) 知识点契合度：是否覆盖目标知识点\n"
        "2) 难度匹配度：是否符合目标难度（基于参考示例逐一对比）\n"
        "3) 核心素养体现度：是否体现目标核心素养\n"
        "4) 教学组织适切性：题型与组织是否恰当\n\n"
        "## 输出格式（必须严格按此JSON格式）：\n"
        "{\n"
        '  "question1": {\n'
        '    "pass": 0,\n'
        '    "difficulty_analysis": "目标难度=XX，你的判断=易/中/难/过难，只能输出这四类，与参考示例逐一对比说明理由",\n'
        '    "suggestion": ["修改建议1", "修改建议2"],\n'
        '    "rank": 0\n'
        "  },\n"
        '  "question2": {\n'
        '    "pass": 1,\n'
        '    "difficulty_analysis": "目标难度=XX，你的判断=XX，只能输出这四类，与参考示例逐一对比说明理由",\n'
        '    "suggestion": ["修改建议1", "修改建议2"],\n'
        '    "rank": 1\n'
        "  },\n"
        '  "question3": {\n'
        '    "pass": 1,\n'
        '    "difficulty_analysis": "目标难度=XX，你的判断=XX，只能输出这四类，与参考示例逐一对比说明理由",\n'
        '    "suggestion": ["修改建议1", "修改建议2"],\n'
        '    "rank": 2\n'
        "  }\n"
        "}\n\n"
        "评分说明：\n"
        "- pass: 0或1，任一维度为0则整体为0\n"
        "- suggestion: 针对扣分原因提出2-3条具体改进，不得引入新标准\n"
        "- rank: 0/1/2，分别代表三题中最差/中等/最好，必须各出现一次\n\n"
        f"教育目标：{goals}\n\n"
        + _format_three_questions_content(writer_content) +
        "注意：请对三条题目分别进行教育价值评估和排名。\n\n"
    )

    return prompt


