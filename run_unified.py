#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一生成和测评系统启动脚本
使用配置文件简化操作
"""

import asyncio
import sys
import os
from unified_generation_evaluation import UnifiedGenerationEvaluation
from config_unified import (
    get_data_file, get_index_range, get_generation_config, 
    get_evaluation_config, get_preset_scenario,
    list_preset_scenarios, list_available_configs
)

async def run_preset_scenario(scenario_name: str):
    """运行预设场景"""
    print(f"🚀 运行预设场景: {scenario_name}")
    print("=" * 80)
    
    # 获取场景配置
    scenario = get_preset_scenario(scenario_name)
    print(f"📋 场景描述: {scenario['description']}")
    
    system = UnifiedGenerationEvaluation()
    
    # 解析数据文件
    data_file_parts = scenario['data_file'].split('.')
    data_type = data_file_parts[0]
    file_size = data_file_parts[1]
    data_file = get_data_file(data_type, file_size)
    
    # 解析索引范围
    start_index, end_index = get_index_range(scenario['index_range'])
    
    # 获取生成和测评配置
    generation_config = get_generation_config(scenario['generation'])
    evaluation_config = get_evaluation_config(scenario['evaluation'])
    
    # 检查是否是多个方法的对比
    if 'methods' in scenario:
        print(f"🔄 运行多个方法对比: {', '.join(scenario['methods'])}")
        
        results = {}
        for method_id in scenario['methods']:
            print(f"\n📝 运行方法: {method_id}")
            
            result = await system.run_complete_pipeline(
                method_id=method_id,
                data_file=data_file,
                start_index=start_index,
                end_index=end_index,
                **generation_config,
                **evaluation_config
            )
            
            results[method_id] = result
            
            if result["success"]:
                print(f"✅ {method_id} 完成")
                print(f"   📝 生成文件: {result['generation']['output_file']}")
                if result["evaluation"].get("evaluation_file"):
                    print(f"   📊 测评文件: {result['evaluation']['evaluation_file']}")
            else:
                print(f"❌ {method_id} 失败: {result['error']}")
        
        return results
    else:
        # 单个方法
        method_id = scenario['method']
        print(f"📝 运行方法: {method_id}")
        
        result = await system.run_complete_pipeline(
            method_id=method_id,
            data_file=data_file,
            start_index=start_index,
            end_index=end_index,
            **generation_config,
            **evaluation_config
        )
        
        if result["success"]:
            print("✅ 场景运行完成!")
            print(f"📝 生成文件: {result['generation']['output_file']}")
            if result["generation"].get("workflow_file"):
                print(f"📋 Workflow文件: {result['generation']['workflow_file']}")
            if result["evaluation"].get("evaluation_file"):
                print(f"📊 测评文件: {result['evaluation']['evaluation_file']}")
        else:
            print(f"❌ 场景运行失败: {result['error']}")
        
        return result

async def run_custom_config(method_id: str, data_type: str, file_size: str = "standard",
                          range_name: str = "medium", generation_config: str = "standard",
                          evaluation_config: str = "standard"):
    """运行自定义配置"""
    print(f"🚀 运行自定义配置")
    print("=" * 80)
    
    # 获取配置
    data_file = get_data_file(data_type, file_size)
    start_index, end_index = get_index_range(range_name)
    gen_config = get_generation_config(generation_config)
    eval_config = get_evaluation_config(evaluation_config)
    
    print(f"📝 方法: {method_id}")
    print(f"📁 数据文件: {data_file}")
    print(f"📊 索引范围: {start_index}-{end_index}")
    print(f"⚙️ 生成配置: {generation_config}")
    print(f"📊 测评配置: {evaluation_config}")
    
    system = UnifiedGenerationEvaluation()
    
    result = await system.run_complete_pipeline(
        method_id=method_id,
        data_file=data_file,
        start_index=start_index,
        end_index=end_index,
        **gen_config,
        **eval_config
    )
    
    if result["success"]:
        print("✅ 自定义配置运行完成!")
        print(f"📝 生成文件: {result['generation']['output_file']}")
        if result["generation"].get("workflow_file"):
            print(f"📋 Workflow文件: {result['generation']['workflow_file']}")
        if result["evaluation"].get("evaluation_file"):
            print(f"📊 测评文件: {result['evaluation']['evaluation_file']}")
    else:
        print(f"❌ 自定义配置运行失败: {result['error']}")
    
    return result

def show_help():
    """显示帮助信息"""
    print("=" * 80)
    print("🚀 统一生成和测评系统启动脚本")
    print("=" * 80)
    print()
    print("使用方法:")
    print("  python run_unified.py <模式> [参数]")
    print()
    print("模式:")
    print("  preset <场景名>     - 运行预设场景")
    print("  custom <方法> <题型> [参数...] - 运行自定义配置")
    print("  list-presets       - 列出所有预设场景")
    print("  list-configs       - 列出所有配置选项")
    print("  help               - 显示此帮助信息")
    print()
    print("预设场景示例:")
    print("  python run_unified.py preset quick_test")
    print("  python run_unified.py preset v3_choice_standard")
    print("  python run_unified.py preset baseline_comparison")
    print()
    print("自定义配置示例:")
    print("  python run_unified.py custom v3_choice choice")
    print("  python run_unified.py custom v3_blank blank medium standard")
    print("  python run_unified.py custom cot_choice choice small fast basic")
    print()
    print("参数说明:")
    print("  题型: choice (选择题) 或 blank (填空题)")
    print("  文件大小: standard, small, test")
    print("  索引范围: small (0-10), medium (0-50), large (0-100)")
    print("  生成配置: fast, standard, thorough")
    print("  测评配置: basic, standard, comprehensive")

async def main():
    """主函数"""
    if len(sys.argv) < 2:
        show_help()
        return
    
    mode = sys.argv[1].lower()
    
    if mode == "help":
        show_help()
    elif mode == "list-presets":
        list_preset_scenarios()
    elif mode == "list-configs":
        list_available_configs()
    elif mode == "preset":
        if len(sys.argv) < 3:
            print("❌ 错误: 请指定预设场景名称")
            print("使用 'python run_unified.py list-presets' 查看所有预设场景")
            return
        
        scenario_name = sys.argv[2]
        try:
            await run_preset_scenario(scenario_name)
        except ValueError as e:
            print(f"❌ 错误: {e}")
            print("使用 'python run_unified.py list-presets' 查看所有预设场景")
    elif mode == "custom":
        if len(sys.argv) < 4:
            print("❌ 错误: 自定义配置需要至少指定方法和题型")
            print("示例: python run_unified.py custom v3_choice choice")
            return
        
        method_id = sys.argv[2]
        data_type = sys.argv[3]
        file_size = sys.argv[4] if len(sys.argv) > 4 else "standard"
        range_name = sys.argv[5] if len(sys.argv) > 5 else "medium"
        generation_config = sys.argv[6] if len(sys.argv) > 6 else "standard"
        evaluation_config = sys.argv[7] if len(sys.argv) > 7 else "standard"
        
        try:
            await run_custom_config(
                method_id, data_type, file_size, 
                range_name, generation_config, evaluation_config
            )
        except ValueError as e:
            print(f"❌ 错误: {e}")
            print("使用 'python run_unified.py list-configs' 查看所有配置选项")
    else:
        print(f"❌ 错误: 未知模式 '{mode}'")
        show_help()

if __name__ == "__main__":
    asyncio.run(main())
