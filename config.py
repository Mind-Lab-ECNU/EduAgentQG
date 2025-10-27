#!/usr/bin/env python3
"""
多智能体系统配置文件
"""

import os
from typing import Dict, Any

# ==================== 环境变量配置 ====================

# OpenAI API配置
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-api-key-here")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://xiaoai.plus/v1")



# 本地模型配置
LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH", "Qwen/Qwen2.5-7B-Instruct")
LOCAL_API_BASE = os.getenv("LOCAL_API_BASE", "https://api.siliconflow.cn")
SC_API_KEY = os.getenv("SC_API_KEY", "your-api-key-here")



# LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH", "Qwen2.5-7B-Instruct")
# LOCAL_API_BASE = os.getenv("LOCAL_API_BASE", "http://172.23.40.13:8000/v1")

# ==================== 模型配置 ====================

# 外部模型配置 (Teacher/通用) 使用 gemini-2.0-flash
EXTERNAL_MODEL_CONFIG = {
    "model": "gemini-2.0-flash",
    "base_url": OPENAI_BASE_URL,
    "api_key": OPENAI_API_KEY,
    "model_info": {
        "vision": False,
        "function_calling": True,
        "json_output": True,
        "family": "gemini-2.0-flash",
    },
    "timeout": 1200.0,  # 增加超时到20分钟
    "max_retries": 10,
}

# 外部模型配置（Writer专用，更发散） - gemini-2.5-flash
EXTERNAL_MODEL_CONFIG_WRITER = {
    "model": "gemini-2.5-flash",
    "base_url": OPENAI_BASE_URL,
    "api_key": OPENAI_API_KEY,
    "model_info": {
        "vision": False,
        "function_calling": True,
        "json_output": True,
        "family": "gemini-2.5-flash",
    },
    "timeout": 1200.0,  # 增加超时到20分钟
    "max_retries": 10,
    # 更发散的默认采样参数（仅Writer使用）
    # 注意：Gemini模型不支持presence_penalty和frequency_penalty
    "temperature": 1.0,
    "top_p": 0.95,
}

# 外部模型配置（Planner专用，发散规划） - gemini-2.0-flash
EXTERNAL_MODEL_CONFIG_PLANNER = {
    "model": "gemini-2.0-flash",
    "base_url": OPENAI_BASE_URL,
    "api_key": OPENAI_API_KEY,
    "model_info": {
        "vision": False,
        "function_calling": True,
        "json_output": True,
        "family": "gemini-2.0-flash",
    },
    "timeout": 1200.0,  # 增加超时到20分钟
    "max_retries": 10,
    # Planner发散参数：鼓励多样化的规划思路
    "temperature": 1.0,
    "top_p": 0.9,
}

# 外部模型配置（Educator专用） - gemini-2.5-flash
EXTERNAL_MODEL_CONFIG_EDUCATOR = {
    "model": "gemini-2.5-flash",
    "base_url": OPENAI_BASE_URL,
    "api_key": OPENAI_API_KEY,
    "model_info": {
        "vision": False,
        "function_calling": True,
        "json_output": True,
        "family": "gemini-2.5-flash",
    },
    "timeout": 1200.0,  # 增加超时到20分钟
    "max_retries": 5,  # 限制重试次数（Educator）
}

# 外部模型配置（Solver专用） - gemini-2.5-flash
EXTERNAL_MODEL_CONFIG_SOLVER = {
    "model": "gemini-2.5-flash",
    "base_url": OPENAI_BASE_URL,
    "api_key": OPENAI_API_KEY,
    "model_info": {
        "vision": False,
        "function_calling": True,
        "json_output": True,
        "family": "gemini-2.5-flash",
    },
    "timeout": 1200.0,  # 增加超时到20分钟
    "max_retries": 5,  # 限制重试次数（Solver）
}

# 本地模型配置 (Qwen2.5-7B-Instruct)
LOCAL_MODEL_CONFIG = {
    "model": LOCAL_MODEL_PATH,
    "base_url": LOCAL_API_BASE,
    "api_key": SC_API_KEY,
    "model_info": {
        "vision": False,
        "function_calling": False,  # 本地模型默认关闭函数调用
        "json_output": False,       # 默认不强制JSON
        "family": "qwen",
    },
    "timeout": 1200.0,  # 增加超时到20分钟
    "max_retries": 10,
    "max_model_len": 8192,
    "max_tokens": 2048,
    # Writer多样性：提高采样随机性
    # 注意：本地模型可能不支持presence_penalty和frequency_penalty
    "temperature": 1.0,
    "top_p": 0.9,
}

# ==================== 智能体配置 ====================

# Planner智能体配置
PLANNER_CONFIG = {
    "name": "Planner",
    "model_type": "external_planner",  # 使用Planner专用外部配置（发散规划）
    "reflect_on_tool_use": True,
    "model_client_stream": True,  # 启用流式响应
}

# Solver智能体配置
SOLVER_CONFIG = {
    "name": "Solver",
    "model_type": "external_solver",  # 使用Solver专用外部配置（gemini-2.0-flash）
    "reflect_on_tool_use": True,
    "model_client_stream": True,  # 启用流式响应
}

# Writer智能体配置
WRITER_CONFIG = {
    "name": "Writer",
    "model_type": "external_writer",  # 使用Writer专用外部配置（更发散）
    "reflect_on_tool_use": True,
    "model_client_stream": False,
}

# Educator智能体配置
EDUCATOR_CONFIG = {
    "name": "Educator",
    "model_type": "external_educator",  # 使用Educator专用外部配置（gemini-2.5-flash）
    "reflect_on_tool_use": True,
    "model_client_stream": False,  # 关闭流式，提升稳定性
}

# Teacher智能体配置
TEACHER_CONFIG = {
    "name": "Teacher",
    "model_type": "external",
    "reflect_on_tool_use": True,
    "model_client_stream": False,  # 关闭流式，提升稳定性
}

# 所有智能体配置
AGENT_CONFIGS = {
    "planner": PLANNER_CONFIG,
    "solver": SOLVER_CONFIG,
    "writer": WRITER_CONFIG,
    "educator": EDUCATOR_CONFIG,
    "teacher": TEACHER_CONFIG,
}

# ==================== 工作流配置 ====================

# 教育题目工作流配置
EDUCATIONAL_WORKFLOW_CONFIG = {
    "name": "教育题目设计工作流",
    "description": "从规划到评审的完整教育题目设计流程",
    "stages": [
        {
            "name": "题目规划",
            "agent": "planner",
            "task": "请为初中数学设计一道关于二次函数的应用题，要求：1)难度适中 2)贴近生活实际 3)包含多个知识点",
            "description": "制定题目设计规划和评分标准"
        },
        {
            "name": "题目编写",
            "agent": "writer",
            "task": "根据规划者的要求，编写一道具体的二次函数应用题",
            "description": "基于规划编写具体题目内容"
        },
        {
            "name": "题目评分",
            "agent": "solver",
            "task": "对编写的题目进行评分（满分10分），分析优缺点并给出具体分数。请在回复中明确写出分数，如：'评分：8.5分'",
            "description": "客观评估题目质量并给出明确分数"
        },
        {
            "name": "教育评审",
            "agent": "educator",
            "task": "从教育角度评估这道题目的教学价值（满分10分），给出具体分数。请在回复中明确写出分数，如：'评分：8.5分'",
            "description": "专业教育角度评估和改进建议并给出明确分数"
        }
    ]
}

# ==================== 系统配置 ====================

# 日志配置
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": "multi_agent_system.log",
}

# 重试配置
RETRY_CONFIG = {
    "max_retries": 3,
    "base_delay": 2.0,
    "max_delay": 30.0,
    "exponential_base": 2,
}

# 超时配置
TIMEOUT_CONFIG = {
    "connect": 300.0,
    "read": 1200.0,  # 增加读取超时到20分钟
    "write": 300.0,
    "pool": 60.0,
}

# ==================== 工具函数 ====================

def get_agent_config(agent_name: str) -> Dict[str, Any]:
    """获取指定智能体的配置"""
    return AGENT_CONFIGS.get(agent_name, {})

def get_model_config(model_type: str) -> Dict[str, Any]:
    """获取指定类型模型的配置"""
    if model_type == "external":
        return EXTERNAL_MODEL_CONFIG
    elif model_type == "external_planner":
        return EXTERNAL_MODEL_CONFIG_PLANNER
    elif model_type == "external_writer":
        return EXTERNAL_MODEL_CONFIG_WRITER
    elif model_type == "external_educator":
        return EXTERNAL_MODEL_CONFIG_EDUCATOR
    elif model_type == "external_solver":
        return EXTERNAL_MODEL_CONFIG_SOLVER
    elif model_type == "local":
        return LOCAL_MODEL_CONFIG
    else:
        raise ValueError(f"未知的模型类型: {model_type}")

def validate_config() -> bool:
    """验证配置的有效性"""
    try:
        # 检查必要的环境变量
        if OPENAI_API_KEY == "your-openai-api-key":
            print("警告: 请设置OPENAI_API_KEY环境变量")
            return False         
        return True
    except Exception as e:
        print(f"配置验证失败: {e}")
        return False

# ==================== 配置信息 ====================

if __name__ == "__main__":
    print("多智能体系统配置信息:")
    print(f"外部模型: {EXTERNAL_MODEL_CONFIG['model']}")
    print(f"本地模型: {LOCAL_MODEL_CONFIG['model']}")
    print(f"智能体数量: {len(AGENT_CONFIGS)}")
    print(f"工作流阶段: {len(EDUCATIONAL_WORKFLOW_CONFIG['stages'])}")
    
    if validate_config():
        print("配置验证通过")
    else:
        print("配置验证失败，请检查配置")

def update_model_configs_for_qwen():
    """动态更新所有模型配置为Qwen-72B"""
    global EXTERNAL_MODEL_CONFIG, EXTERNAL_MODEL_CONFIG_WRITER, EXTERNAL_MODEL_CONFIG_PLANNER
    global EXTERNAL_MODEL_CONFIG_EDUCATOR, EXTERNAL_MODEL_CONFIG_SOLVER, LOCAL_MODEL_CONFIG
    
    # 检查是否已经设置为Qwen配置
    if EXTERNAL_MODEL_CONFIG["model"] == "Qwen/Qwen2.5-72B-Instruct":
        return  # 已经配置过了
    
    # 更新所有外部模型配置为Qwen-72B
    qwen_config = {
        "model": "Qwen/Qwen2.5-72B-Instruct",
        "base_url": os.getenv("QWEN_BASE_URL", "your-qwen-base-url"),
        "api_key": os.getenv("QWEN_API_KEY", "your-qwen-api-key"),
        "model_info": {
            "vision": False,
            "function_calling": True,
            "json_output": True,
            "family": "qwen-72b",
        },
        "timeout": 1200.0,  # 增加超时到20分钟
        "max_retries": 10,
        "temperature": 1.2,  # 设置温度为1.0
        "top_p": 0.9
    }
    
    # 更新所有外部模型配置
    EXTERNAL_MODEL_CONFIG.update(qwen_config)
    EXTERNAL_MODEL_CONFIG_WRITER.update(qwen_config)
    EXTERNAL_MODEL_CONFIG_PLANNER.update(qwen_config)
    EXTERNAL_MODEL_CONFIG_EDUCATOR.update(qwen_config)
    EXTERNAL_MODEL_CONFIG_SOLVER.update(qwen_config)
    
    # 更新本地模型配置
    LOCAL_MODEL_CONFIG.update(qwen_config)
    
    print("✅ 所有模型配置已更新为Qwen-72B")

def update_model_configs_for_gemini():
    """恢复所有模型配置为Gemini"""
    global EXTERNAL_MODEL_CONFIG, EXTERNAL_MODEL_CONFIG_WRITER, EXTERNAL_MODEL_CONFIG_PLANNER
    global EXTERNAL_MODEL_CONFIG_EDUCATOR, EXTERNAL_MODEL_CONFIG_SOLVER, LOCAL_MODEL_CONFIG
    
    # 恢复为原始Gemini配置
    EXTERNAL_MODEL_CONFIG.update({
        "model": "gemini-2.0-flash",
        "base_url": OPENAI_BASE_URL,
        "api_key": OPENAI_API_KEY,
        "model_info": {
            "vision": False,
            "function_calling": True,
            "json_output": True,
            "family": "gemini-2.0-flash",
        },
        "timeout": 1200.0,  # 增加超时到20分钟
        "max_retries": 10,
    })
    
    # 其他配置保持原样，只更新base_url和api_key
    for config in [EXTERNAL_MODEL_CONFIG_WRITER, EXTERNAL_MODEL_CONFIG_PLANNER, 
                   EXTERNAL_MODEL_CONFIG_EDUCATOR, EXTERNAL_MODEL_CONFIG_SOLVER]:
        config["base_url"] = OPENAI_BASE_URL
        config["api_key"] = OPENAI_API_KEY
    
    # 恢复本地模型配置
    LOCAL_MODEL_CONFIG.update({
        "model": LOCAL_MODEL_PATH,
        "base_url": LOCAL_API_BASE,
        "api_key": SC_API_KEY,
        "model_info": {
            "vision": False,
            "function_calling": True,
            "json_output": True,
            "family": "qwen-7b",
        },
        "timeout": 1200.0,  # 增加超时到20分钟
        "max_retries": 10,
    })
    
    print("✅ 所有模型配置已恢复为Gemini")

def update_model_configs_for_gpt4o_mini():
    """动态更新所有模型配置为GPT-4o-mini"""
    global EXTERNAL_MODEL_CONFIG, EXTERNAL_MODEL_CONFIG_WRITER, EXTERNAL_MODEL_CONFIG_PLANNER
    global EXTERNAL_MODEL_CONFIG_EDUCATOR, EXTERNAL_MODEL_CONFIG_SOLVER, LOCAL_MODEL_CONFIG
    
    # 检查是否已经设置为GPT-4o-mini配置
    if EXTERNAL_MODEL_CONFIG["model"] == "gpt-4o-mini":
        return  # 已经配置过了
    
    # 更新所有外部模型配置为GPT-4o-mini
    gpt4o_mini_config = {
        "model": "gpt-4o-mini",
        "base_url": OPENAI_BASE_URL,
        "api_key": OPENAI_API_KEY,
        "model_info": {
            "vision": False,
            "function_calling": True,
            "json_output": True,
            "family": "gpt-4o-mini",
        },
        "timeout": 1200.0,  # 增加超时到20分钟
        "max_retries": 10,
        "temperature": 0.7,
    }
    
    # 更新所有外部模型配置
    EXTERNAL_MODEL_CONFIG.update(gpt4o_mini_config)
    EXTERNAL_MODEL_CONFIG_WRITER.update(gpt4o_mini_config)
    EXTERNAL_MODEL_CONFIG_PLANNER.update(gpt4o_mini_config)
    EXTERNAL_MODEL_CONFIG_EDUCATOR.update(gpt4o_mini_config)
    EXTERNAL_MODEL_CONFIG_SOLVER.update(gpt4o_mini_config)
    
    # 更新本地模型配置
    LOCAL_MODEL_CONFIG.update(gpt4o_mini_config)
    
    print("✅ 所有模型配置已更新为GPT-4o-mini")
