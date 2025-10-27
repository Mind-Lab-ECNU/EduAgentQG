import json
import time
import os
from datetime import datetime
from openai import OpenAI, OpenAIError, AuthenticationError

# -------------------------------
# 配置 openai client
# -------------------------------
client = OpenAI(
    api_key="sk-yiIIWRemdBaOfnzU9dxg3B5NRbaH5yf1lzNzG83MANDwoqy2",
    base_url="https://xiaoai.plus/v1"
)

# -------------------------------
# 参数配置
# -------------------------------
MAX_RETRIES = 5
TIMEOUT = 300  # 秒

file1_path = r"D:\CODE\three_0921\data\choice_unquie_500.jsonl"
file2_path = r"D:\CODE\three_0921\baseline\COT_outputs\merged_COT_choice_dedup.json"
# queid_file = r"D:\CODE\three_0921\eval\failed_questions.json"
queid_file = r" "

model_name = "deepseek-v3"

# -------------------------------
# 读取文件
# -------------------------------
data1 = {}
with open(file1_path, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        data1[obj["que_id"]] = obj

with open(file2_path, "r", encoding="utf-8") as f:
    file2_data = json.load(f)
    # 处理新的文件格式，提取 results 字段
    if isinstance(file2_data, dict) and "results" in file2_data:
        items = file2_data["results"]
    else:
        items = file2_data  # 兼容旧格式
    data2 = {obj["que_id"]: obj for obj in items}

# -------------------------------
# 确定测评题目集合
# -------------------------------
common_ids = set(data1.keys()) & set(data2.keys())
if not common_ids:
    raise ValueError("没有匹配到相同的 que_id，请检查数据文件！")

selected_ids = None
queid_used = False  # 标记是否使用了 que_id.json

if os.path.exists(queid_file):
    with open(queid_file, "r", encoding="utf-8") as f:
        try:
            selected_ids = json.load(f)
            if isinstance(selected_ids, list) and selected_ids:
                eval_ids = common_ids & set(selected_ids)
                queid_used = True
                print(f"检测到 {queid_file}，包含 {len(selected_ids)} 个ID，最终交集 {len(eval_ids)} 个题目将被测评")
            else:
                print(f"{queid_file} 为空或格式不正确，忽略，使用全部交集")
                eval_ids = common_ids
        except json.JSONDecodeError:
            print(f"{queid_file} 解析失败，忽略，使用全部交集")
            eval_ids = common_ids
else:
    eval_ids = common_ids
    print(f"未找到 {queid_file}，默认使用全部交集，共 {len(eval_ids)} 个题目")

print(f"本次最终需测评 {len(eval_ids)} 道题目")

# -------------------------------
# 开始测评
# -------------------------------
results = []
failed_ids = []
wins = 0
total = 0

for idx, qid in enumerate(eval_ids, 1):
    q1 = data1[qid]
    q2 = data2[qid]
    print(f"正在测评 {idx}/{len(eval_ids)} que_id = {qid}")

    try:
        # 保证 q2['question'] 是字典，否则直接失败
        if not isinstance(q2.get("question"), dict):
            raise ValueError("q2['question'] 不是字典")

        prompt = f"""
你是一名资深数学教育专家。请严格评估两个问题哪个更符合教育目标。

教育目标：年级={q1['grade']}, 难度={q1['difficulty']}, 
知识点={q1['knowledge']}, 核心素养={','.join(q1['competence'])}

问题1：
题目：{q1['content']}
答案：{q1['answer']}

问题2：
题目：{q2['question']['question']}
选项：{'；'.join(q2['question'].get('options', []))}
答案：{q2['question']['answer']}

请根据以下维度分析：
1. 概念覆盖的完备性：分析所需概念的覆盖面，检查缺失或冗余点
2. 难度水平的匹配：评估与指定难度水平的一致性，验证适当的目标一致性
3. 与能力发展的相关性：确认符合该年级学生的水平，与概述的要求一致
4. 数学素养的发展：分析对数学素养发展的贡献
5. 结构的科学设计：评估问题结构合理性，组织和指导质量
6. 文本清晰度和连贯性：评估措辞清晰度和简明性，有效交流解题信息

重要：请严格按照以下JSON格式输出，不要添加任何其他文字或代码块标记：
{{"Better_Quest": 1或2, "原因": "详细理由，请将所有内容放在一行内，不要换行"}}
"""

    except Exception as e:
        print(f"[{qid}] 构造 prompt 失败: {e}")
        failed_ids.append(qid)
        continue

    success = False
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                timeout=TIMEOUT
            )

            result_text = response.choices[0].message.content
            print(f"[{qid}] API返回内容: {result_text[:200]}...")

            # 提取JSON内容
            if "```json" in result_text:
                import re
                json_match = re.search(r'```json\s*\n(.*?)\n```', result_text, re.DOTALL)
                if json_match:
                    result_text = json_match.group(1)
                else:
                    result_text = result_text.split('```json')[1].split('```')[0].strip()

            try:
                result_json = json.loads(result_text)
            except json.JSONDecodeError as e:
                print(f"[{qid}] JSON解析失败: {e}")
                continue

            results.append({"que_id": qid, **result_json})
            if result_json.get("Better_Quest") == 2:
                wins += 1
            total += 1
            success = True
            break

        except (AuthenticationError, OpenAIError, Exception) as e:
            print(f"[{qid}] 调用出错: {e}, 尝试 {attempt+1}/{MAX_RETRIES}")
            time.sleep(3)
            continue

    if not success:
        failed_ids.append(qid)
        print(f"[{qid}] 最终失败，加入失败列表")

# -------------------------------
# 输出汇总结果
# -------------------------------
win_rate = wins / total if total > 0 else 0
coverage_rate = total / len(eval_ids) if len(eval_ids) > 0 else 0
print(f"文件二胜出的比例: {win_rate:.2%}")
print(f"测评覆盖率: {coverage_rate:.2%}")
print(f"成功测评题目数: {total}")
print(f"文件二胜出题目数: {wins}")
print(f"总失败题目数: {len(failed_ids)}")

# -------------------------------
# 保存结果
# -------------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

final_output = {
    "statistics": {
        "win_rate": win_rate,
        "coverage_rate": coverage_rate,
        "total_success": total,
        "wins": wins,
        "failed_count": len(failed_ids),
        "queid_file_used": queid_used,
        "total_eval": len(eval_ids)
    },
    "results": results,
    "failed": failed_ids
}

out_put_win_rate = f"Win_Rate_our_COT_choice_{timestamp}_{model_name}.json"
with open(out_put_win_rate, "w", encoding="utf-8") as f:
    json.dump(final_output, f, ensure_ascii=False, indent=2)

# 单独保存失败题目 ID
failed_file = f"failed_questions_{timestamp}_{model_name}.json"
with open(failed_file, "w", encoding="utf-8") as f:
    json.dump(failed_ids, f, ensure_ascii=False, indent=2)

print(f"结果已保存到 {out_put_win_rate}")
print(f"失败题目ID已单独保存到 {failed_file}")
