#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一生成和测评系统配置文件
"""

# ==================== 默认配置 ====================

# 生成方法配置
DEFAULT_METHOD = "v3_choice"  # 默认使用V3选择题方法

# 数据文件配置
DATA_FILES = {
    "choice": {
        "standard": r"D:\CODE\three_0921\data\choice_unquie_500.jsonl",
        "small": r"D:\CODE\three_0921\data\choice_sample10.jsonl",
        "test": r"D:\CODE\three_0921\data\choice_sample10_2.jsonl"
    },
    "blank": {
        "standard": r"D:\CODE\three_0921\data\blank_unique.jsonl",
        "small": r"D:\CODE\three_0921\data\sample_from_all_choice_blank_10.jsonl",
        "test": r"D:\CODE\three_0921\data\sample_from_all_choice_blank_100.jsonl"
    }
}

# 索引范围配置
INDEX_RANGES = {
    "small": {"start": 0, "end": 10},      # 小规模测试
    "medium": {"start": 0, "end": 50},     # 中等规模
    "large": {"start": 0, "end": 100},     # 大规模
    "custom": {"start": 0, "end": None}    # 自定义
}

# 生成参数配置
GENERATION_CONFIGS = {
    "fast": {
        "batch_size": 1,
        "delay_between_batches": 0.2,
        "use_rag": False,
        "evaluation_mode": "binary"
    },
    "standard": {
        "batch_size": 5,
        "delay_between_batches": 1.0,
        "use_rag": True,
        "rag_mode": "planner",
        "evaluation_mode": "binary"
    },
    "thorough": {
        "batch_size": 3,
        "delay_between_batches": 2.0,
        "use_rag": True,
        "rag_mode": "writer",
        "evaluation_mode": "score"
    }
}

# 测评配置
EVALUATION_CONFIGS = {
    "basic": {
        "enable_consistency": True,
        "enable_winrate": False,
        "enable_diversity": False
    },
    "standard": {
        "enable_consistency": True,
        "enable_winrate": True,
        "enable_diversity": False
    },
    "comprehensive": {
        "enable_consistency": True,
        "enable_winrate": True,
        "enable_diversity": True
    }
}

# 预设场景配置
PRESET_SCENARIOS = {
    "quick_test": {
        "description": "快速测试 - 生成少量题目进行快速验证",
        "method": "v3_choice",
        "data_file": "choice.small",
        "index_range": "small",
        "generation": "fast",
        "evaluation": "basic"
    },
    "v3_choice_standard": {
        "description": "V3选择题标准测试 - 使用V3方法生成选择题并进行标准测评",
        "method": "v3_choice",
        "data_file": "choice.standard",
        "index_range": "medium",
        "generation": "standard",
        "evaluation": "standard"
    },
    "v3_blank_standard": {
        "description": "V3填空题标准测试 - 使用V3方法生成填空题并进行标准测评",
        "method": "v3_blank",
        "data_file": "blank.standard",
        "index_range": "medium",
        "generation": "standard",
        "evaluation": "standard"
    },
    "baseline_comparison": {
        "description": "基线方法对比 - 生成多个基线方法的结果进行对比",
        "methods": ["cot_choice", "cot_n_choice", "react_choice"],
        "data_file": "choice.standard",
        "index_range": "small",
        "generation": "fast",
        "evaluation": "standard"
    },
    "comprehensive_evaluation": {
        "description": "全面测评 - 生成题目并进行全面测评",
        "method": "v3_choice",
        "data_file": "choice.standard",
        "index_range": "large",
        "generation": "thorough",
        "evaluation": "comprehensive"
    }
}

# ==================== 辅助函数 ====================

def get_data_file(data_type: str, file_size: str = "standard") -> str:
    """获取数据文件路径"""
    return DATA_FILES[data_type][file_size]

def get_index_range(range_name: str) -> tuple:
    """获取索引范围"""
    if range_name not in INDEX_RANGES:
        raise ValueError(f"未知的索引范围: {range_name}")
    
    range_config = INDEX_RANGES[range_name]
    return range_config["start"], range_config["end"]

def get_generation_config(config_name: str) -> dict:
    """获取生成配置"""
    if config_name not in GENERATION_CONFIGS:
        raise ValueError(f"未知的生成配置: {config_name}")
    
    return GENERATION_CONFIGS[config_name].copy()

def get_evaluation_config(config_name: str) -> dict:
    """获取测评配置"""
    if config_name not in EVALUATION_CONFIGS:
        raise ValueError(f"未知的测评配置: {config_name}")
    
    return EVALUATION_CONFIGS[config_name].copy()

def get_preset_scenario(scenario_name: str) -> dict:
    """获取预设场景配置"""
    if scenario_name not in PRESET_SCENARIOS:
        raise ValueError(f"未知的预设场景: {scenario_name}")
    
    return PRESET_SCENARIOS[scenario_name].copy()

def list_preset_scenarios():
    """列出所有预设场景"""
    print("=" * 80)
    print("📋 预设场景配置:")
    print("=" * 80)
    
    for scenario_name, scenario_config in PRESET_SCENARIOS.items():
        print(f"🔹 {scenario_name}")
        print(f"   描述: {scenario_config['description']}")
        if 'method' in scenario_config:
            print(f"   方法: {scenario_config['method']}")
        elif 'methods' in scenario_config:
            print(f"   方法: {', '.join(scenario_config['methods'])}")
        print(f"   数据文件: {scenario_config['data_file']}")
        print(f"   索引范围: {scenario_config['index_range']}")
        print(f"   生成配置: {scenario_config['generation']}")
        print(f"   测评配置: {scenario_config['evaluation']}")
        print()

def list_available_configs():
    """列出所有可用配置"""
    print("=" * 80)
    print("📋 可用配置选项:")
    print("=" * 80)
    
    print("🔹 数据文件类型:")
    for data_type, files in DATA_FILES.items():
        print(f"   {data_type}: {list(files.keys())}")
    
    print("\n🔹 索引范围:")
    for range_name, range_config in INDEX_RANGES.items():
        print(f"   {range_name}: {range_config['start']}-{range_config['end']}")
    
    print("\n🔹 生成配置:")
    for config_name in GENERATION_CONFIGS.keys():
        print(f"   {config_name}")
    
    print("\n🔹 测评配置:")
    for config_name in EVALUATION_CONFIGS.keys():
        print(f"   {config_name}")
    
    print()

if __name__ == "__main__":
    # 显示所有配置选项
    list_available_configs()
    list_preset_scenarios()
