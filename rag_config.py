#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG模式配置
"""

# RAG模式选择
RAG_MODES = {
    "planner_full": {
        "name": "Planner完整RAG",
        "description": "Planner获得知识图谱+课程标准+题目模式分析+题目样例",
        "use_rag": True,
        "rag_mode": "planner",
        "include_question_patterns": True,
        "include_question_samples": True
    },
    "planner_basic": {
        "name": "Planner基础RAG", 
        "description": "Planner只获得知识图谱+课程标准，无题目模式分析",
        "use_rag": True,
        "rag_mode": "planner",
        "include_question_patterns": False,
        "include_question_samples": False
    },
    "writer_rag": {
        "name": "Writer RAG",
        "description": "Planner获得知识图谱+课程标准，Writer获得题目参考样例",
        "use_rag": True,
        "rag_mode": "writer",
        "include_question_patterns": False,
        "include_question_samples": False
    },
    "no_rag": {
        "name": "禁用RAG",
        "description": "不使用任何RAG功能，完全基于基础配置",
        "use_rag": False,
        "rag_mode": "none",
        "include_question_patterns": False,
        "include_question_samples": False
    }
}

# 默认RAG模式
DEFAULT_RAG_MODE = "planner_basic"

def get_rag_config(mode: str = None):
    """获取RAG配置"""
    if mode is None:
        mode = DEFAULT_RAG_MODE
    
    if mode not in RAG_MODES:
        print(f"⚠️ 未知的RAG模式: {mode}")
        print(f"可用模式: {list(RAG_MODES.keys())}")
        mode = DEFAULT_RAG_MODE
    
    return RAG_MODES[mode]

def list_rag_modes():
    """列出所有可用的RAG模式"""
    print("🎯 可用的RAG模式:")
    print("="*60)
    for mode, config in RAG_MODES.items():
        print(f"模式: {mode}")
        print(f"名称: {config['name']}")
        print(f"描述: {config['description']}")
        print(f"RAG开关: {config['use_rag']}")
        print(f"RAG模式: {config['rag_mode']}")
        print(f"包含题目模式分析: {config['include_question_patterns']}")
        print(f"包含题目样例: {config['include_question_samples']}")
        print("-" * 60)

if __name__ == "__main__":
    list_rag_modes()
