#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一测评脚本 - 填空题版本（整合多样性、一致性和Win Rate测评）
"""

import json
import time
import re
import os
import sys
from datetime import datetime
from pathlib import Path
from openai import OpenAI, OpenAIError, AuthenticationError
from bert_score import score as bert_score
import jieba

# ==================== 配置 ====================

# OpenAI 客户端配置
client = OpenAI(
    api_key="sk-yiIIWRemdBaOfnzU9dxg3B5NRbaH5yf1lzNzG83MANDwoqy2",
    base_url="https://xiaoai.plus/v1"
)

# 测评参数
MAX_RETRIES = 5
TIMEOUT = 300
model_name = ["deepseek-v3" ,"gpt-4o"] # 可为字符串或数组
MAX_WINRATE_QUESTIONS = None  # Win Rate测评最大题量，None表示全量

# 测评开关配置
EVALUATION_SWITCHES = {
    "diversity": False,      # 多样性测评开关
    "consistency": True,    # 一致性测评开关
    "winrate": False        # Win Rate测评开关
}

# 索引范围配置（可选）
INDEX_RANGE = {
    "enabled": False,       # 是否启用索引范围
    "start_index": 0,       # 开始索引（从0开始）
    "end_index": 100         # 结束索引（不包含）
}

# 文件路径配置（默认值，可通过参数覆盖）
file1_path = r"D:\CODE\three_0921\data\blank_unique.jsonl"  # 金标准数据
file2_path = r"D:\CODE\three_0921\outputs\generated_questions_20250930_015527_writer_0-500_eval_V3填空题生成_deepseek_v3_input_1_20250930_040446.json"  # 生成数据1
file3_path = r"D:\CODE\three_0921\outputs\generated_questions_20250930_040449_writer_0-500_eval_V3填空题生成_deepseek_v3_input_2_20250930_040446.json"  # 生成数据2（多样性比较用，可以设置不同文件）

# 输出配置
output_dir = Path(r"D:\CODE\three_0921\eval\unified_results")
output_dir.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# ==================== 工具函数 ====================

def safe_json_parse(text: str):
    """安全的JSON解析，具有强大的容错能力"""
    if not text or not isinstance(text, str):
        return None
    
    # 清理文本
    text = text.strip()
    if not text:
        return None
    
    # 移除代码块标记
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*$", "", text)
    text = text.strip()
    
    # 移除BOM和零宽字符
    text = text.replace("\ufeff", "").replace("\u200b", "").replace("\u200c", "").replace("\u200d", "")
    
    # 尝试直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"[safe_json_parse] 直接解析失败: {e}")
        
        # 尝试修复常见的JSON问题
        try:
            # 1. 移除尾随逗号
            text = re.sub(r',\s*}', '}', text)
            text = re.sub(r',\s*]', ']', text)
            
            # 2. 修复单引号为双引号
            text = re.sub(r"'([^']*)':", r'"\1":', text)
            text = re.sub(r":\s*'([^']*)'", r': "\1"', text)
            
            # 3. 修复True/False/null
            text = re.sub(r'\bTrue\b', 'true', text)
            text = re.sub(r'\bFalse\b', 'false', text)
            text = re.sub(r'\bNone\b', 'null', text)
            
            # 4. 移除注释
            text = re.sub(r'//.*$', '', text, flags=re.MULTILINE)
            text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
            
            return json.loads(text)
        except json.JSONDecodeError as e2:
            print(f"[safe_json_parse] 修复后解析失败: {e2}")
            
            # 尝试提取JSON对象
            try:
                # 查找第一个完整的JSON对象
                brace_count = 0
                start_idx = -1
                end_idx = -1
                
                for i, char in enumerate(text):
                    if char == '{':
                        if start_idx == -1:
                            start_idx = i
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0 and start_idx != -1:
                            end_idx = i
                            break
                
                if start_idx != -1 and end_idx != -1:
                    json_text = text[start_idx:end_idx + 1]
                    return json.loads(json_text)
                else:
                    print(f"[safe_json_parse] 无法找到完整的JSON对象")
                    return None
                    
            except json.JSONDecodeError as e3:
                print(f"[safe_json_parse] 提取JSON对象失败: {e3}")
                
                # 最后尝试：逐行解析
                try:
                    lines = text.split('\n')
                    json_lines = []
                    in_json = False
                    
                    for line in lines:
                        line = line.strip()
                        if line.startswith('{') or in_json:
                            in_json = True
                            json_lines.append(line)
                            if line.endswith('}') and line.count('{') == line.count('}'):
                                break
                    
                    if json_lines:
                        json_text = '\n'.join(json_lines)
                        return json.loads(json_text)
                    else:
                        print(f"[safe_json_parse] 无法从文本中提取JSON")
                        return None
                        
                except json.JSONDecodeError as e4:
                    print(f"[safe_json_parse] 所有解析方法都失败了: {e4}")
                    print(f"[safe_json_parse] 原始文本前200字符: {text[:200]}")
                    return None

def load_data_with_format_check(file_path):
    """加载数据并处理不同格式"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ JSON解析错误: {e}")
        print(f"📁 尝试修复文件: {file_path}")
        
        # 尝试修复JSON格式
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # 检查是否是JSONL格式（每行一个JSON对象）
            if file_path.endswith('.jsonl') or ('\n' in content and content.count('{') > 1):
                print("🔧 检测到JSONL格式，转换为JSON数组...")
                lines = content.strip().split('\n')
                json_objects = []
                
                for i, line in enumerate(lines):
                    line = line.strip()
                    if line:
                        try:
                            obj = json.loads(line)
                            json_objects.append(obj)
                        except json.JSONDecodeError as line_e:
                            print(f"⚠️ 跳过第{i+1}行: {line_e}")
                            continue
                
                if json_objects:
                    data = json_objects
                    print(f"✅ 转换成功，提取到 {len(json_objects)} 个JSON对象")
                else:
                    raise ValueError("无法解析JSONL内容")
            
            # 尝试处理多个JSON对象的情况
            elif content.strip().startswith('{') and content.count('{') > 1:
                print("🔧 检测到多个JSON对象，尝试包装成数组...")
                lines = content.strip().split('\n')
                json_objects = []
                for line in lines:
                    line = line.strip()
                    if line and (line.startswith('{') or line.startswith('[')):
                        try:
                            obj = json.loads(line)
                            json_objects.append(obj)
                        except:
                            continue
                
                if json_objects:
                    data = json_objects
                    print(f"✅ 修复成功，提取到 {len(json_objects)} 个JSON对象")
                else:
                    raise ValueError("无法解析JSON内容")
            else:
                raise ValueError("无法修复JSON格式")
                
        except Exception as e2:
            print(f"❌ 修复失败: {e2}")
            raise ValueError(f"无法加载文件 {file_path}: {e}")
    
    # 处理新的文件格式，提取 results 字段
    if isinstance(data, dict) and "results" in data:
        items = data["results"]
        print(f"📁 检测到新格式，提取 {len(items)} 条结果")
    elif isinstance(data, list):
        items = data
        print(f"📁 检测到数组格式，包含 {len(items)} 条结果")
    else:
        raise ValueError(f"不支持的文件格式: {file_path}")
    
    return items

def apply_index_range(data, data_name="数据"):
    """根据索引范围过滤数据"""
    if not INDEX_RANGE["enabled"]:
        return data
    
    start_idx = INDEX_RANGE["start_index"]
    end_idx = INDEX_RANGE["end_index"]
    
    if start_idx < 0 or end_idx <= start_idx:
        print(f"⚠️ 索引范围无效: {start_idx}-{end_idx}，跳过过滤")
        return data
    
    if end_idx > len(data):
        end_idx = len(data)
        print(f"⚠️ 结束索引超出数据范围，调整为: {start_idx}-{end_idx}")
    
    filtered_data = data[start_idx:end_idx]
    print(f"🔧 {data_name}索引过滤: {len(data)} -> {len(filtered_data)} (索引 {start_idx}-{end_idx-1})")
    
    return filtered_data

# ==================== 多样性测评 ====================

def calculate_diversity_metrics(file2_data, file3_data):
    """计算多样性指标 - 比较相同que_id的题目（使用多种指标）"""
    print("\n🔍 开始多样性测评...")
    
    # 构建que_id到题目的映射
    print("📝 构建题目映射...")
    data2_dict = {}
    data3_dict = {}
    
    for item in file2_data:
        if "question" in item and isinstance(item["question"], dict):
            que_id = item.get("que_id", "")
            q = item["question"].get("question", "")
            if que_id and q:
                data2_dict[que_id] = f"题目：{q}"
    
    for item in file3_data:
        if "question" in item and isinstance(item["question"], dict):
            que_id = item.get("que_id", "")
            q = item["question"].get("question", "")
            if que_id and q:
                data3_dict[que_id] = f"题目：{q}"
    
    print(f"📊 文件2题目数: {len(data2_dict)}, 文件3题目数: {len(data3_dict)}")
    
    # 找到共同的que_id
    common_ids = set(data2_dict.keys()) & set(data3_dict.keys())
    print(f"📊 共同que_id数量: {len(common_ids)}")
    
    if not common_ids:
        return {"error": "没有找到共同的que_id"}
    
    # 准备比较数据
    refs = [data2_dict[qid] for qid in common_ids]
    cands = [data3_dict[qid] for qid in common_ids]
    
    print(f"🔧 比较所有共同题目: {len(common_ids)} 个")
    print(f"   ⏱️ 预计耗时: {len(common_ids) * 0.5}秒")
    
    # 计算多种多样性指标
    print("\n📊 计算多样性指标...")
    print("   ⏳ 正在加载BERT模型，请稍候...")
    
    try:
        # BERTScore
        print("   🔄 计算BERTScore...")
        P, R, F1 = bert_score(cands, refs, lang="zh", verbose=False)
        avg_bert_f1 = F1.mean().item()
        print(f"   ✅ BERTScore完成: {avg_bert_f1:.4f}")
        
        # BLEU
        print("   🔄 计算BLEU...")
        from sacrebleu.metrics import BLEU
        bleu_metric = BLEU()
        cands_tok = [" ".join(jieba.cut(t)) for t in cands]
        refs_tok = [[" ".join(jieba.cut(t)) for t in refs]]
        bleu = bleu_metric.corpus_score(cands_tok, refs_tok).score
        print(f"   ✅ BLEU完成: {bleu:.4f}")
        
        # METEOR
        print("   🔄 计算METEOR...")
        from nltk.translate.meteor_score import meteor_score as nltk_meteor
        meteor_scores = []
        for r, c in zip(refs, cands):
            r_tok = list(jieba.cut(r))
            c_tok = list(jieba.cut(c))
            meteor_scores.append(nltk_meteor([r_tok], c_tok))
        avg_meteor = sum(meteor_scores) / len(meteor_scores)
        print(f"   ✅ METEOR完成: {avg_meteor:.4f}")
        
        # ROUGE-L
        print("   🔄 计算ROUGE-L...")
        from rouge_score import rouge_scorer
        rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        rouge_scores = []
        for r, c in zip(refs, cands):
            r_tok = "".join(jieba.cut(r))
            c_tok = "".join(jieba.cut(c))
            rouge_scores.append(rouge.score(r_tok, c_tok)["rougeL"].fmeasure)
        avg_rougeL = sum(rouge_scores) / len(rouge_scores)
        print(f"   ✅ ROUGE-L完成: {avg_rougeL:.4f}")
        
        # 计算最终结果
        diversity_metrics = {
            "avg_BLEU": bleu,
            "avg_METEOR": avg_meteor,
            "avg_ROUGE_L": avg_rougeL,
            "avg_BERTScore_F1": avg_bert_f1,
            "matched_question_count": len(common_ids),
            "tokenize_mode": "jieba",
            "evaluation_model": "bert_score+sacrebleu+nltk+rouge_score"  # 多样性使用多种工具
        }
        
        print(f"\n✅ 多样性测评完成!")
        print(f"   📊 BLEU: {bleu:.4f}")
        print(f"   📊 METEOR: {avg_meteor:.4f}")
        print(f"   📊 ROUGE-L: {avg_rougeL:.4f}")
        print(f"   📊 BERTScore-F1: {avg_bert_f1:.4f}")
        print(f"   📊 比较题目数: {len(common_ids)}")
        
        return diversity_metrics
        
    except Exception as e:
        print(f"   ❌ 多样性计算失败: {e}")
        return {"error": f"多样性计算失败: {str(e)}"}

# ==================== 一致性测评 ====================

def calculate_consistency_metrics(file2_data, model_name=None):
    """计算一致性指标"""
    print("\n🎯 开始一致性测评...")
    
    # 当前评测模型
    current_model = model_name if model_name else globals().get('model_name', 'gpt-4o')

    results = []
    total_questions = len(file2_data)
    processed_count = 0  # 实际处理的题目计数
    
    for i, item in enumerate(file2_data, 1):
        if "question" not in item:
            continue

        question_data = item["question"]
        # 兼容V3（字符串）与baseline（对象）
        if isinstance(question_data, str):
            q = question_data
        elif isinstance(question_data, dict):
            q = question_data.get("question", "")
        else:
            q = ""
        if not q:
            continue
        
        processed_count += 1  # 增加实际处理计数
            
        # 构建教育目标
        input_data = item.get("input", {})
        education_goals = {
            "grade": input_data.get("grade", ""),
            "difficulty": input_data.get("difficulty", ""),
            "competence": input_data.get("competence", []),
            "knowledge": input_data.get("knowledge", ""),
            "question_type": input_data.get("question_type", "")
        }
        
        prompt = f"""
将以下任务视为数学教育评估作业。你将作为一名高级数学教育专家，严格评估给定的数学问题及其相关的教育目标。你将根据以下三个关键维度进行评估，仔细检查每一项。给出0-10之间的评分，可以有一位小数，比如8.2：
关键评估维度：
1. 知识点覆盖：验证所有必需概念的完整覆盖，任何遗漏或引入未提及的点都是不合规的
2. 难度适配性：分析认知水平与目标年级和难度的一致性，不得超过或低于目标级别
3. 题目准确率：确认对所有目标技能的综合评估
4. 素养导向性：评估培养：几何直观、基础能力、空间观念、应用意识、量感、数感、推理意识、符号意识、创新意识、运算能力、数据意识、模型意识、模型观念、推理能力、抽象能力、数据观念
严格要求：在整个评估过程中保持客观性和严谨性

题目: {q}
教育目标: {json.dumps(education_goals, ensure_ascii=False)}

请严格输出 JSON，不要添加任何文字或代码块，格式：
{{
    "que_id": "{item.get('que_id', '')}",
    "评测结果": {{
        "知识点覆盖": {{"score": 0.0, "reason": ""}},
        "难度适配性": {{"score": 0.0, "reason": ""}},
        "题目准确性": {{"score": 0.0, "reason": ""}},
        "素养导向性": {{"score": 0.0, "reason": ""}}
    }},
    "Education_Goals": {json.dumps(education_goals, ensure_ascii=False)},
    "Question": {json.dumps(q, ensure_ascii=False)}
}}
"""
        
        for attempt in range(MAX_RETRIES):
            try:
                response = client.chat.completions.create(
                    model=current_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    timeout=TIMEOUT
                )
                
                result_text = response.choices[0].message.content
                result_json = safe_json_parse(result_text)
                
                if result_json is not None:
                    # 保留小数一位
                    for dim in ["知识点匹配度", "难度适配性", "题目准确性", "素养导向性"]:
                        if dim in result_json.get("评测结果", {}) and "score" in result_json["评测结果"][dim]:
                            score_val = result_json["评测结果"][dim]["score"]
                            result_json["评测结果"][dim]["score"] = round(float(score_val), 1)
                    results.append(result_json)
                break
                
            except (AuthenticationError, OpenAIError, Exception) as e:
                print(f"[{item.get('que_id', '')}] 调用出错: {e}, 尝试 {attempt+1}/{MAX_RETRIES}")
                time.sleep(3)
        
        if processed_count % 10 == 0:
            print(f"📊 一致性测评进度: {processed_count} (原始索引: {i}/{total_questions})")
    
    # 计算统计信息
    if results:
        stats = {}
        for dim in ["知识点覆盖", "难度适配性", "题目准确性", "素养导向性"]:
            valid_scores = [
                r["评测结果"][dim]["score"] for r in results
                if isinstance(r, dict)
                and isinstance(r.get("评测结果"), dict)
                and isinstance(r["评测结果"].get(dim), dict)
                and isinstance(r["评测结果"][dim].get("score"), (int, float))
            ]
            if valid_scores:
                stats[dim] = round(sum(valid_scores) / len(valid_scores), 1)
            else:
                stats[dim] = 0.0
        
        stats["总平均分"] = round(sum(stats[dim] for dim in ["知识点覆盖", "难度适配性", "题目准确性", "素养导向性"]) / 4, 1)
        
        consistency_metrics = {
            "statistics": stats,
            "total_evaluated": len(results),
            "total_processed": processed_count,
            "total_questions": total_questions,
            "evaluation_model": current_model,
            "detailed_results": results  # 添加详细的题目结果
        }
        
        print(f"✅ 一致性测评完成 - 总平均分: {stats['总平均分']}")
        return consistency_metrics
    
    return {"error": "无法完成一致性测评"}

# ==================== Win Rate测评 ====================

def calculate_winrate_metrics(file1_data, file2_data, model_name=None):
    """计算Win Rate指标"""
    print("\n🏆 开始Win Rate测评...")
    
    # 当前评测模型与题量上限
    current_model = model_name if model_name else globals().get('model_name', 'gpt-4o')
    max_questions = globals().get('MAX_WINRATE_QUESTIONS', 50)
    
    # 构建数据字典
    data1 = {item["que_id"]: item for item in file1_data}
    data2 = {item["que_id"]: item for item in file2_data}
    
    # 找到共同题目
    common_ids = set(data1.keys()) & set(data2.keys())
    if not common_ids:
        return {"error": "没有匹配到相同的 que_id"}
    
    print(f"📊 找到 {len(common_ids)} 道共同题目")
    
    # 选择测评题目
    if max_questions is None:
        selected_ids = list(common_ids)
        print(f"📊 将测评所有 {len(selected_ids)} 道题目")
    else:
        selected_ids = list(common_ids)[:max_questions]
        print(f"📊 将测评前 {len(selected_ids)} 道题目（限制：{max_questions}道）")
    
    results = []
    for i, que_id in enumerate(selected_ids, 1):
        item1 = data1[que_id]
        item2 = data2[que_id]
        
        # 构建题目文本
        def build_question_text(item):
            # 检查数据结构
            if "question" in item:
                qdata = item["question"]
                if isinstance(qdata, str):
                    q = qdata
                    a = item.get("answer", "")
                elif isinstance(qdata, dict):
                    q = qdata.get("question", "")
                    a = qdata.get("answer", "")
                else:
                    q = item.get("question", "")
                    a = item.get("answer", "")
            elif "content" in item:
                # file1_data 的格式：直接的content和answer字段
                q = item.get("content", "")
                a = item.get("answer", "")
            else:
                # 其他格式的兼容处理
                q = item.get("question", "")
                a = item.get("answer", "")
            return f"题目：{q}\n答案：{a}"
        
        question1 = build_question_text(item1)
        question2 = build_question_text(item2)
        
        input_data = item2.get("input", {}) if isinstance(item2, dict) else {}
        education_goals = {
            "grade": input_data.get("grade", ""),
            "difficulty": input_data.get("difficulty", ""),
            "competence": input_data.get("competence", []),
            "knowledge": input_data.get("knowledge", []),
            "question_type": input_data.get("question_type", "")
        }

        prompt = f"""
作为一名资深数学教育专家，请对以下问题进行严格的评价和比较。在评估过程中，根据以下维度对每个问题进行分析，并确定哪个问题更符合教育目标。
评估维度：
1. 概念覆盖的完备性：分析所需概念的覆盖面，检查缺失或冗余点
2. 难度水平的匹配：评估与指定难度水平的一致性，验证适当的目标一致性
3. 与能力发展的相关性：确认符合该年级学生的水平，与概述的要求一致
4. 数学素养的发展：分析对数学素养发展的贡献
5. 结构的科学设计：评估问题结构合理性，组织和指导质量
6. 文本清晰度和连贯性：评估措辞清晰度和简明性，有效交流解题信息

输出格式：{{"Better_Quest": 1或2, "原因": "详细的评估理由，解释为什么选择的问题更好，并说明哪个维度(S)表现更好。"}}
必输项：Education_Goals：{json.dumps(education_goals, ensure_ascii=False)}
问题对：
问题1：
{question1}

问题2：
{question2}
"""
        
        for attempt in range(MAX_RETRIES):
            try:
                response = client.chat.completions.create(
                    model=current_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    timeout=TIMEOUT
                )
                
                result_text = response.choices[0].message.content
                result_json = safe_json_parse(result_text)

                winner_val = None
                reason_val = ""
                scores_val = {}
                if isinstance(result_json, dict):
                    if "winner" in result_json:
                        winner_val = result_json.get("winner")
                        reason_val = result_json.get("reason", "")
                        scores_val = result_json.get("scores", {})
                    elif "Better_Quest" in result_json or "better_quest" in result_json:
                        bq = result_json.get("Better_Quest", result_json.get("better_quest"))
                        if bq in [1, 2, "1", "2"]:
                            winner_val = "A" if str(bq) == "1" else "B"
                        reason_val = result_json.get("原因", result_json.get("reason", ""))

                if winner_val in ["A", "B"]:
                    results.append({
                        "que_id": que_id,
                        "winner": winner_val,
                        "reason": reason_val,
                        "scores": scores_val,
                        "question_A": question1,
                        "question_B": question2
                    })
                    break
                    
            except (AuthenticationError, OpenAIError, Exception) as e:
                print(f"[{que_id}] Win Rate测评出错: {e}, 尝试 {attempt+1}/{MAX_RETRIES}")
                time.sleep(3)
        
        if i % 5 == 0:
            print(f"📊 Win Rate测评进度: {i}/{len(selected_ids)}")
    
    # 计算Win Rate
    if results:
        wins = sum(1 for r in results if r["winner"] == "B")  # B是生成题目
        total = len(results)
        win_rate = wins / total if total > 0 else 0
        
        winrate_metrics = {
            "win_rate": round(win_rate, 4),
            "wins": wins,
            "total": total,
            "evaluated_questions": results,
            "evaluation_model": current_model
        }
        
        print(f"✅ Win Rate测评完成 - Win Rate: {win_rate:.4f} ({wins}/{total})")
        return winrate_metrics
    
    return {"error": "无法完成Win Rate测评"}

# ==================== 主函数 ====================

def main(file1=None, file2=None, file3=None, eval_model=None, start_idx=None, end_idx=None, max_winrate_questions=None):
    """主测评函数"""
    global file1_path, file2_path, file3_path, model_name, INDEX_RANGE, MAX_WINRATE_QUESTIONS
    
    # 如果传入了参数，则使用传入的参数
    if file1:
        file1_path = file1
    if file2:
        file2_path = file2
    if file3:
        file3_path = file3
    if eval_model:
        model_name = eval_model
    if start_idx is not None and end_idx is not None:
        INDEX_RANGE["enabled"] = True
        INDEX_RANGE["start_index"] = start_idx
        INDEX_RANGE["end_index"] = end_idx
    if max_winrate_questions is not None:
        MAX_WINRATE_QUESTIONS = max_winrate_questions
    
    # 处理模型名称（支持字符串或数组）
    if isinstance(model_name, str):
        models = [model_name]
    elif isinstance(model_name, list):
        models = model_name
    else:
        models = [str(model_name)]

    print("🚀 开始统一测评（填空题版本）...")
    print(f"📁 参考数据: {file1_path}")
    print(f"📁 生成数据: {file2_path}")
    if file3_path:
        print(f"📁 生成数据2: {file3_path}")
    print(f"🤖 测评模型: {models}")
    print("=" * 60)
    
    try:
        # 加载数据
        print("📖 加载数据...")
        file1_data = load_data_with_format_check(file1_path)
        file2_data = load_data_with_format_check(file2_path)
        
        # 加载第三个文件（多样性比较用）
        print("📖 加载第三个文件（多样性比较用）...")
        file3_data = load_data_with_format_check(file3_path)
        
        # 应用索引范围过滤
        if INDEX_RANGE["enabled"]:
            print(f"\n🔧 应用索引范围过滤: {INDEX_RANGE['start_index']}-{INDEX_RANGE['end_index']-1}")
            file1_data = apply_index_range(file1_data, "金标准数据")
            file2_data = apply_index_range(file2_data, "生成数据1")
            file3_data = apply_index_range(file3_data, "生成数据2")
        
        # 为每个模型分别执行测评
        all_results = {}
        for model_idx, current_model in enumerate(models):
            print(f"\n{'='*60}")
            print(f"🤖 使用模型 {model_idx + 1}/{len(models)}: {current_model}")
            print(f"{'='*60}")

            evaluation_results = {
                "timestamp": timestamp,
                "model_name": current_model,
                "evaluation_model": current_model,
                "file1_path": file1_path,
                "file2_path": file2_path,
                "file3_path": file3_path,
                "file1_count": len(file1_data),
                "file2_count": len(file2_data),
                "file3_count": len(file3_data),
                "index_range": INDEX_RANGE if INDEX_RANGE["enabled"] else None,
                "evaluation_switches": EVALUATION_SWITCHES
            }

            if EVALUATION_SWITCHES["diversity"]:
                print(f"\n🔍 执行多样性测评 (模型: {current_model})...")
                diversity_results = calculate_diversity_metrics(file2_data, file3_data)
                evaluation_results["diversity"] = diversity_results
            else:
                print(f"\n⏭️ 跳过多样性测评 (模型: {current_model})")
                evaluation_results["diversity"] = {"skipped": True, "reason": "用户关闭"}

            if EVALUATION_SWITCHES["consistency"]:
                print(f"\n🎯 执行一致性测评 (模型: {current_model})...")
                consistency_results = calculate_consistency_metrics(file2_data, current_model)
                evaluation_results["consistency"] = consistency_results
            else:
                print(f"\n⏭️ 跳过一致性测评 (模型: {current_model})")
                evaluation_results["consistency"] = {"skipped": True, "reason": "用户关闭"}

            if EVALUATION_SWITCHES["winrate"]:
                print(f"\n🏆 执行Win Rate测评 (模型: {current_model})...")
                winrate_results = calculate_winrate_metrics(file1_data, file2_data, current_model)
                evaluation_results["winrate"] = winrate_results
            else:
                print(f"\n⏭️ 跳过Win Rate测评 (模型: {current_model})")
                evaluation_results["winrate"] = {"skipped": True, "reason": "用户关闭"}

            all_results[current_model] = evaluation_results

            # 每个模型单独保存
            output_filename = f"unified_evaluation_blank_{timestamp}_{current_model}.json"
            output_path = output_dir / output_filename
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(evaluation_results, f, ensure_ascii=False, indent=2)
            print(f"✅ 模型 {current_model} 的测评结果已保存到: {output_path}")

        # 多模型汇总
        if len(models) > 1:
            summary = {
                "timestamp": timestamp,
                "models": models,
                "file1_path": file1_path,
                "file2_path": file2_path,
                "file3_path": file3_path,
                "file1_count": len(file1_data),
                "file2_count": len(file2_data),
                "file3_count": len(file3_data),
                "index_range": INDEX_RANGE if INDEX_RANGE["enabled"] else None,
                "evaluation_switches": EVALUATION_SWITCHES,
                "results_by_model": all_results
            }
            summary_file = output_dir / f"unified_evaluation_blank_summary_{timestamp}.json"
            with open(summary_file, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            print(f"\n📊 多模型汇总结果已保存到: {summary_file}")
        else:
            # 单模型情况下保留兼容总结输出
            single = list(all_results.values())[0]
            print("\n" + "=" * 60)
            print("📊 测评结果汇总:")
            # 显示测评开关状态
            print(f"\n🔧 测评开关状态:")
            for switch, enabled in EVALUATION_SWITCHES.items():
                status = "✅ 开启" if enabled else "❌ 关闭"
                print(f"  {switch}: {status}")
            # 摘要
            if "diversity" in single and "error" not in single["diversity"] and "skipped" not in single["diversity"]:
                div = single["diversity"]
                print(f"\n🔍 多样性: BLEU={div['avg_BLEU']:.4f}, METEOR={div['avg_METEOR']:.4f}")
            if "consistency" in single and "error" not in single["consistency"] and "skipped" not in single["consistency"]:
                cons = single["consistency"]
                print(f"   🎯 一致性: 总平均分 {cons['statistics']['总平均分']}")
            if "winrate" in single and "error" not in single["winrate"] and "skipped" not in single["winrate"]:
                wr = single["winrate"]
                print(f"   🏆 Win Rate: {wr['win_rate']:.4f} ({wr['wins']}/{wr['total']})")

        print("\n✅ 统一测评完成！")
        
    except Exception as e:
        print(f"❌ 测评过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
