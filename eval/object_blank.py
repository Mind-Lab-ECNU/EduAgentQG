import json
import time
import re
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
model_name = "deepseek-v3"
input_file = r"D:\CODE\three_0921\eval\data\blank2\merged_blank2.json"

# 运行开始时间戳与实时落盘文件
START_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
realtime_jsonl = f"Objective_consistency_blank2_{model_name}_{START_TS}.jsonl"
snapshot_path = f"Objective_consistency_blank2_{model_name}_{START_TS}_snapshot.json"

# -------------------------------
# 安全 JSON 解析函数
# -------------------------------
def safe_json_parse(text: str):
    try:
        text = re.sub(r"```json\s*", "", text)
        text = text.replace("```", "").strip("\ufeff \n")
        return json.loads(text)
    except json.JSONDecodeError as e:
        snippet = text[:200].replace("\n", "\\n")
        print(f"[safe_json_parse] JSON解析失败: {e}")
        print(f"[safe_json_parse] 原始内容: {snippet}...")
        return None

# -------------------------------
# 实时落盘工具
# -------------------------------
def append_realtime(entry: dict):
    try:
        with open(realtime_jsonl, "a", encoding="utf-8") as rf:
            rf.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[realtime] 追加实时日志失败: {e}")

def write_snapshot(results: list, stats: dict | None = None):
    try:
        payload = {"evaluations": results}
        if stats is not None:
            payload["statistics"] = stats
        with open(snapshot_path, "w", encoding="utf-8") as sf:
            json.dump(payload, sf, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[snapshot] 写入快照失败: {e}")

# -------------------------------
# 统一解析题目
# -------------------------------
def parse_questions(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    data = {}
    for obj in items:
        que_id = obj.get("que_id")
        if not que_id:
            continue
        if que_id in data:
            continue  # 避免重复

        if "input" in obj and "question" in obj:
            q_input = obj["input"]
            q_question = obj["question"]
            edu_goals = {
                "grade": q_input.get("grade", ""),
                "difficulty": q_input.get("difficulty", ""),
                "competence": q_input.get("competence", []),
                "knowledge": q_input.get("knowledge", ""),
                "question_type": q_input.get("question_type", "")
            }
            content = q_question.get("question", "")
            answer = q_question.get("answer", "")
            options = q_question.get("options", [])
            text = f"题目：{content}\n答案：{answer}"
            if options:
                text = f"题目：{content}\n选项：{'；'.join(options)}\n答案：{answer}"
            data[que_id] = {"text": text, "education_goals": edu_goals}
        elif "reasoning" in obj and "question" in obj:
            q_input = obj.get("input", {})
            q_question = obj.get("question", {})
            edu_goals = {
                "grade": q_input.get("grade", ""),
                "difficulty": q_input.get("difficulty", ""),
                "competence": q_input.get("competence", []),
                "knowledge": q_input.get("knowledge", ""),
                "question_type": q_input.get("question_type", "")
            }
            content = q_question.get("question", "")
            answer = q_question.get("answer", "")
            options = q_question.get("options", [])
            text = f"题目：{content}\n答案：{answer}"
            if options:
                text = f"题目：{content}\n选项：{'；'.join(options)}\n答案：{answer}"
            data[que_id] = {"text": text, "education_goals": edu_goals}

    return data

# -------------------------------
# 加载题目
# -------------------------------
data = parse_questions(input_file)
print(f"加载题目数量: {len(data)}")

# -------------------------------
# 批量评测
# -------------------------------
results = []

for que_id, item in data.items():
    prompt = f"""
将以下任务视为数学教育评估作业。你将作为一名高级数学教育专家，严格评估给定的数学问题及其相关的教育目标。请根据以下四个关键维度进行评估，每个维度给出0-10分，可保留一位小数。

题目: {item['text']}
教育目标: {json.dumps(item['education_goals'], ensure_ascii=False)}

请严格输出 JSON，不要添加任何文字或代码块，格式：
{{
    "que_id": "{que_id}",
    "评测结果": {{
        "知识点匹配度": {{"score": 0.0, "reason": ""}},
        "题目准确性": {{"score": 0.0, "reason": ""}},
        "素养导向性": {{"score": 0.0, "reason": ""}},
        "难度适配性": {{"score": 0.0, "reason": ""}}
    }},
    "Education_Goals": {json.dumps(item['education_goals'], ensure_ascii=False)},
    "Question": {json.dumps(item['text'], ensure_ascii=False)}
}}
"""

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                timeout=TIMEOUT
            )

            result_text = response.choices[0].message.content
            snippet = result_text[:200].replace("\n", " ")
            print(f"[{que_id}] API 返回（前200字符）：{snippet}")

            result_json = safe_json_parse(result_text)
            if result_json is not None:
                # 保留小数一位
                for dim in ["知识点匹配度", "题目准确性", "素养导向性", "难度适配性"]:
                    if dim in result_json.get("评测结果", {}) and "score" in result_json["评测结果"][dim]:
                        score_val = result_json["评测结果"][dim]["score"]
                        result_json["评测结果"][dim]["score"] = round(float(score_val), 1)
                results.append(result_json)
                append_realtime({
                    "que_id": que_id,
                    "status": "parsed",
                    "result": result_json
                })
                write_snapshot(results)
            break
        except (AuthenticationError, OpenAIError, Exception) as e:
            print(f"[{que_id}] 调用出错: {e}, 尝试 {attempt+1}/{MAX_RETRIES}")
            time.sleep(3)
    # 若解析失败（result_json 为 None），也做实时落盘，避免丢失
    if result_json is None:
        append_realtime({
            "que_id": que_id,
            "status": "raw",
            "raw": result_text if 'result_text' in locals() else ""
        })
        write_snapshot(results)

# -------------------------------
# 统计模块
# -------------------------------
stats = {
    "知识点匹配度": 0.0,
    "题目准确性": 0.0,
    "素养导向性": 0.0,
    "难度适配性": 0.0,
    "总平均分": 0.0
}

try:
    if results:
        # 仅使用结构完整的样本进行统计
        for dim in ["知识点匹配度", "题目准确性", "素养导向性", "难度适配性"]:
            valid = [
                r for r in results
                if isinstance(r, dict)
                and isinstance(r.get("评测结果"), dict)
                and isinstance(r["评测结果"].get(dim), dict)
                and isinstance(r["评测结果"][dim].get("score"), (int, float))
            ]
            n_valid = len(valid)
            stats[dim] = round(
                (sum(r["评测结果"][dim]["score"] for r in valid) / n_valid) if n_valid else 0.0,
                1
            )
        stats["总平均分"] = round(sum(stats[dim] for dim in ["知识点匹配度", "题目准确性", "素养导向性", "难度适配性"]) / 4, 1)
except Exception as e:
    print(f"[stats] 统计阶段出错: {e}")
finally:
    # 统计阶段也写一次快照，确保至少保存现状
    write_snapshot(results, stats)

print("四个维度平均分和总平均分:")
print(json.dumps(stats, ensure_ascii=False, indent=2))

# -------------------------------
# 保存结果
# -------------------------------
output_file = f"Objective_consistency_blank2_{model_name}_{START_TS}.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump({"evaluations": results, "statistics": stats, "realtime_log": realtime_jsonl, "snapshot": snapshot_path}, f, ensure_ascii=False, indent=2)

print(f"评测完成，结果已保存至 {output_file}")
