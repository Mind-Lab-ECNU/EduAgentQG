#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一生成和测评系统
支持多种生成方法和测评选项的统一调用
"""

import asyncio
import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import argparse
import sys

# 添加当前目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# 导入各个生成方法
from multi_agent_system_v3 import MultiAgentSystemV3
from baseline.COT import main as cot_main
from baseline.COT_n import main as cot_n_main
from baseline.COT_blank import main as cot_blank_main
from baseline.COT_n_blank import main as cot_n_blank_main
from baseline.ReACT import main as react_main
from baseline.ReACT_blank import main as react_blank_main

# 导入测评模块
from eval.unified_evaluation import main as eval_choice_main
from eval.unified_evaluation_blank import main as eval_blank_main

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('unified_generation_evaluation.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class UnifiedGenerationEvaluation:
    """统一生成和测评系统"""
    
    def __init__(self):
        self.methods = {
            # V3方法（我们的方法）
            "v3_choice": {
                "name": "V3选择题生成",
                "description": "多智能体系统V3 - 选择题生成",
                "type": "v3",
                "question_type": "choice"
            },
            "v3_blank": {
                "name": "V3填空题生成", 
                "description": "多智能体系统V3 - 填空题生成",
                "type": "v3",
                "question_type": "blank"
            },
            
            # 基线方法 - 选择题
            "cot_choice": {
                "name": "COT选择题",
                "description": "Chain of Thought - 选择题生成",
                "type": "baseline",
                "question_type": "choice"
            },
            "cot_n_choice": {
                "name": "COT-N选择题",
                "description": "Chain of Thought Best-of-N - 选择题生成", 
                "type": "baseline",
                "question_type": "choice"
            },
            "react_choice": {
                "name": "ReACT选择题",
                "description": "ReACT - 选择题生成",
                "type": "baseline", 
                "question_type": "choice"
            },
            
            # 基线方法 - 填空题
            "cot_blank": {
                "name": "COT填空题",
                "description": "Chain of Thought - 填空题生成",
                "type": "baseline",
                "question_type": "blank"
            },
            "cot_n_blank": {
                "name": "COT-N填空题", 
                "description": "Chain of Thought Best-of-N - 填空题生成",
                "type": "baseline",
                "question_type": "blank"
            },
            "react_blank": {
                "name": "ReACT填空题",
                "description": "ReACT - 填空题生成", 
                "type": "baseline",
                "question_type": "blank"
            }
        }
        
        self.data_files = {
            "choice": {
                "standard": r"D:\CODE\three_0921\data\choice_unquie_500.jsonl",
                "description": "选择题金标准数据"
            },
            "blank": {
                "standard": r"D:\CODE\three_0921\data\blank_unique.jsonl", 
                "description": "填空题金标准数据"
            }
        }
        
        self.output_dir = Path(r"D:\CODE\three_0921\unified_outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def list_methods(self):
        """列出所有可用的生成方法"""
        print("=" * 80)
        print("📋 可用的生成方法:")
        print("=" * 80)
        
        for method_id, method_info in self.methods.items():
            print(f"🔹 {method_id}")
            print(f"   名称: {method_info['name']}")
            print(f"   描述: {method_info['description']}")
            print(f"   类型: {method_info['type']}")
            print(f"   题型: {method_info['question_type']}")
            print()
    
    def list_data_files(self):
        """列出可用的数据文件"""
        print("=" * 80)
        print("📁 可用的数据文件:")
        print("=" * 80)
        
        for data_type, file_info in self.data_files.items():
            print(f"🔹 {data_type}")
            print(f"   路径: {file_info['standard']}")
            print(f"   描述: {file_info['description']}")
            print(f"   存在: {'✅' if os.path.exists(file_info['standard']) else '❌'}")
            print()
    
    def validate_inputs(self, method_id: str, data_file: str = None, 
                       start_index: int = 0, end_index: int = None) -> Tuple[bool, str]:
        """验证输入参数"""
        # 验证方法ID
        if method_id not in self.methods:
            return False, f"未知的方法ID: {method_id}"
        
        # 验证数据文件
        if data_file and not os.path.exists(data_file):
            return False, f"数据文件不存在: {data_file}"
        
        # 验证索引范围
        if start_index < 0:
            return False, "开始索引不能为负数"
        
        if end_index is not None and end_index <= start_index:
            return False, "结束索引必须大于开始索引"
        
        return True, "参数验证通过"
    
    async def run_with_params(self, method_id: str, start_index: int = 0, end_index: int = 10,
                             data_file: str = None, batch_size: int = 1, 
                             delay_between_batches: float = 2.0, use_rag: bool = True,
                             rag_mode: str = "planner", evaluation_mode: str = "binary",
                             enable_consistency: bool = True, enable_winrate: bool = True,
                             enable_diversity: bool = False, run_twice_for_consistency: bool = True,
                             evaluation_model = "gpt-4o", evaluation_models = None, 
                             base_model_name = "gemini") -> Dict[str, Any]:
        """通过代码参数直接运行完整流程"""
        logger.info("🚀 通过代码参数运行完整流程")
        logger.info(f"📋 方法: {method_id}, 范围: {start_index}-{end_index}")
        
        # 处理测评模型参数
        if evaluation_models is not None:
            # 使用多个测评模型
            return await self.run_complete_pipeline(
                method_id=method_id,
                data_file=data_file,
                start_index=start_index,
                end_index=end_index,
                batch_size=batch_size,
                delay_between_batches=delay_between_batches,
                use_rag=use_rag,
                rag_mode=rag_mode,
                evaluation_mode=evaluation_mode,
                enable_consistency=enable_consistency,
                enable_winrate=enable_winrate,
                enable_diversity=enable_diversity,
                run_twice_for_consistency=run_twice_for_consistency,
                evaluation_models=evaluation_models,
                base_model_name=base_model_name
            )
        else:
            # 使用单个测评模型
            return await self.run_complete_pipeline(
                method_id=method_id,
                data_file=data_file,
                start_index=start_index,
                end_index=end_index,
                batch_size=batch_size,
                delay_between_batches=delay_between_batches,
                use_rag=use_rag,
                rag_mode=rag_mode,
                evaluation_mode=evaluation_mode,
                enable_consistency=enable_consistency,
                enable_winrate=enable_winrate,
                enable_diversity=enable_diversity,
                run_twice_for_consistency=run_twice_for_consistency,
                evaluation_model=evaluation_model,
                base_model_name=base_model_name
        )
    
    async def generate_questions(self, method_id: str, data_file: str = None,
                               start_index: int = 0, end_index: int = None,
                               batch_size: int = 5, delay_between_batches: float = 1.0,
                               use_rag: bool = True, rag_mode: str = "planner",
                               evaluation_mode: str = "binary", base_model_name: str = "gemini") -> Dict[str, Any]:
        """生成题目"""
        method_info = self.methods[method_id]
        question_type = method_info["question_type"]
        
        # 如果没有指定数据文件，使用默认的
        if not data_file:
            data_file = self.data_files[question_type]["standard"]
        
        logger.info(f"🚀 开始使用 {method_info['name']} 生成题目")
        logger.info(f"📁 数据文件: {data_file}")
        logger.info(f"📊 索引范围: {start_index} - {end_index}")
        
        # 生成时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            if method_info["type"] == "v3":
                # V3方法
                return await self._generate_v3_questions(
                    method_id, data_file, start_index, end_index,
                    batch_size, delay_between_batches, use_rag, rag_mode, evaluation_mode, base_model_name
                )
            else:
                # 基线方法
                return await self._generate_baseline_questions(
                    method_id, data_file, start_index, end_index, timestamp, base_model_name, 1
                )
                
        except Exception as e:
            logger.error(f"生成题目失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "output_file": None,
                "method": method_info["name"]
            }
    
    async def _generate_v3_questions(self, method_id: str, data_file: str,
                                   start_index: int, end_index: int,
                                   batch_size: int, delay_between_batches: float,
                                   use_rag: bool, rag_mode: str, evaluation_mode: str, base_model_name: str) -> Dict[str, Any]:
        """使用V3方法生成题目"""
        from datetime import datetime
        start_time = datetime.now()
        
        try:
            # 根据base_model_name设置模型配置
            if base_model_name == "qwen-72b":
                # 设置Qwen-72B配置 - 更新所有5个智能体的配置
                from config import update_model_configs_for_qwen
                update_model_configs_for_qwen()
                logger.info("🤖 所有5个智能体已切换到Qwen-72B模型")
            elif base_model_name == "gemini":
                # 使用默认Gemini配置
                from config import update_model_configs_for_gemini
                update_model_configs_for_gemini()
                logger.info("🤖 所有5个智能体已切换到Gemini模型")
            elif base_model_name == "gpt-4o-mini":
                # 使用GPT-4o-mini配置
                from config import update_model_configs_for_gpt4o_mini
                update_model_configs_for_gpt4o_mini()
                logger.info("🤖 所有5个智能体已切换到GPT-4o-mini模型")
            else:
                logger.warning(f"⚠️ 未知的base_model_name: {base_model_name}，使用默认Gemini配置")
            
            # 创建V3系统
            system = MultiAgentSystemV3(
                use_rag=use_rag,
                rag_mode=rag_mode, 
                evaluation_mode=evaluation_mode
            )
            
            # 初始化系统
            await system.initialize()
            
            # 生成题目
            result = await system.generate_questions_from_data(
                jsonl_file_path=data_file,
                batch_size=batch_size,
                delay_between_batches=delay_between_batches,
                start_index=start_index,
                end_index=end_index
            )
            
            # 关闭系统
            await system.shutdown()
            
            return {
                "success": True,
                "output_file": result["generated"],
                "workflow_file": result["workflow"],
                "method": self.methods[method_id]["name"],
                "start_time": start_time
            }
            
        except Exception as e:
            logger.error(f"V3生成失败: {e}")
            raise
    
    async def _generate_baseline_questions(self, method_id: str, data_file: str,
                                         start_index: int, end_index: int,
                                         timestamp: str, base_model_name: str, 
                                         generation_attempt: int = 1) -> Dict[str, Any]:
        """使用基线方法生成题目"""
        from datetime import datetime
        start_time = datetime.now()
        
        try:
            # 根据base_model_name设置模型配置
            if base_model_name == "qwen-72b":
                # 设置Qwen-72B配置
                import os
                os.environ["OPENAI_BASE_URL"] = "https://notebook-inspire.sii.edu.cn/ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6/project-b795c114-135a-40db-b3d0-19b60f25237b/user-304c6bd0-a3e9-4e9d-826c-dace2a1d04bd/vscode/62fd4373-8a6a-40fd-86c8-4077fa381f74/49942f40-349f-4291-b3a0-d6886c8d2da5/proxy/33001/v1"
                os.environ["OPENAI_API_KEY"] = "sk-pjtlgoubuigtgpxneosvmivvvkopxflxfncnhorzenbasdyb"
                os.environ["OPENAI_MODEL"] = "Qwen/Qwen2.5-72B-Instruct"
                logger.info("🤖 基线方法使用Qwen-72B模型配置")
            elif base_model_name == "gemini":
                # 使用默认Gemini配置
                import os
                os.environ["OPENAI_MODEL"] = "gemini-2.5-flash"
                logger.info("🤖 基线方法使用Gemini模型配置")
            elif base_model_name == "gpt-4o-mini":
                # 使用GPT-4o-mini配置
                import os
                os.environ["OPENAI_MODEL"] = "gpt-4o-mini"
                logger.info("🤖 基线方法使用GPT-4o-mini模型配置")
            else:
                logger.warning(f"⚠️ 未知的base_model_name: {base_model_name}，基线方法使用默认Gemini配置")
            
            # 根据方法ID选择对应的函数
            method_functions = {
                "cot_choice": cot_main,
                "cot_n_choice": cot_n_main,
                "cot_blank": cot_blank_main,
                "cot_n_blank": cot_n_blank_main,
                "react_choice": react_main,
                "react_blank": react_blank_main
            }
            
            if method_id not in method_functions:
                raise ValueError(f"不支持的基线方法: {method_id}")
            
            # 为第二次生成添加不同的时间戳后缀
            if generation_attempt > 1:
                timestamp = f"{timestamp}_attempt_{generation_attempt}"
            
            # 调用对应的基线方法
            output_file = method_functions[method_id](
                data_file=data_file,
                start_idx=start_index,
                end_idx=end_index,
                timestamp=timestamp
            )
            
            return {
                "success": True,
                "output_file": output_file,
                "workflow_file": None,
                "method": self.methods[method_id]["name"],
                "start_time": start_time
            }
            
        except Exception as e:
            logger.error(f"基线方法生成失败: {e}")
            raise
    
    def evaluate_questions(self, method_id: str, generated_file: str,
                          second_generated_file: str = None,
                          start_index: int = 0, end_index: int = None,
                          enable_consistency: bool = True, enable_winrate: bool = True,
                          enable_diversity: bool = False, evaluation_model = "gpt-4o") -> Dict[str, Any]:
        """测评生成的题目"""
        method_info = self.methods[method_id]
        question_type = method_info["question_type"]
        
        logger.info(f"📊 开始测评 {method_info['name']} 生成的题目")
        logger.info(f"📁 生成文件: {generated_file}")
        
        try:
            if question_type == "choice":
                # 选择题测评
                return self._evaluate_choice_questions(
                    generated_file, second_generated_file, start_index, end_index,
                    enable_consistency, enable_winrate, enable_diversity, evaluation_model
                )
            else:
                # 填空题测评
                return self._evaluate_blank_questions(
                    generated_file, second_generated_file, start_index, end_index,
                    enable_consistency, enable_winrate, enable_diversity, evaluation_model
                )
                
        except Exception as e:
            logger.error(f"测评失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "evaluation_file": None
            }
    
    def _evaluate_choice_questions(self, generated_file: str, second_generated_file: str,
                                 start_index: int, end_index: int, enable_consistency: bool,
                                 enable_winrate: bool, enable_diversity: bool, evaluation_model: str = "gpt-4o") -> Dict[str, Any]:
        """测评选择题"""
        try:
            # 动态设置测评模块的文件路径
            import eval.unified_evaluation as eval_module
            
            # 设置文件路径
            eval_module.file1_path = self.data_files["choice"]["standard"]  # 金标准数据（用于Win Rate测评）
            eval_module.file2_path = generated_file  # 第一次生成的文件（用于一致性测评和Win Rate测评）
            eval_module.file3_path = second_generated_file if second_generated_file else generated_file  # 第二次生成的文件（用于多样性测评）
            
            # 设置索引范围 - 对于生成的文件，索引应该从0开始
            eval_module.INDEX_RANGE = {
                "enabled": False,  # 禁用索引过滤，因为生成的文件已经是对应范围的数据
                "start_index": 0,
                "end_index": end_index - start_index if end_index else None
            }
            
            # 设置测评选项
            eval_module.EVALUATION_SWITCHES = {
                "diversity": enable_diversity,
                "consistency": enable_consistency,
                "winrate": enable_winrate
            }
            
            # 调用选择题测评，传入正确的文件路径
            eval_choice_main(
                file1=self.data_files["choice"]["standard"],  # 金标准数据
                file2=generated_file,  # 第一次生成的数据
                file3=second_generated_file,  # 第二次生成的数据（多样性比较用）
                eval_model=evaluation_model,
                start_idx=0,  # 生成的文件索引从0开始
                end_idx=end_index - start_index if end_index else None  # 调整结束索引
            )
            return {
                "success": True,
                "evaluation_file": None,  # 测评模块会直接保存文件，我们稍后获取路径
                "method": "选择题测评"
            }
        except Exception as e:
            logger.error(f"选择题测评失败: {e}")
            raise
    
    def _evaluate_blank_questions(self, generated_file: str, second_generated_file: str,
                                start_index: int, end_index: int, enable_consistency: bool,
                                enable_winrate: bool, enable_diversity: bool, evaluation_model: str = "gpt-4o") -> Dict[str, Any]:
        """测评填空题"""
        try:
            # 动态设置测评模块的文件路径
            import eval.unified_evaluation_blank as eval_module
            
            # 设置文件路径
            eval_module.file1_path = self.data_files["blank"]["standard"]  # 金标准数据（用于Win Rate测评）
            eval_module.file2_path = generated_file  # 第一次生成的文件（用于一致性测评和Win Rate测评）
            eval_module.file3_path = second_generated_file if second_generated_file else generated_file  # 第二次生成的文件（用于多样性测评）
            
            # 设置索引范围 - 对于生成的文件，索引应该从0开始
            eval_module.INDEX_RANGE = {
                "enabled": False,  # 禁用索引过滤，因为生成的文件已经是对应范围的数据
                "start_index": 0,
                "end_index": end_index - start_index if end_index else None
            }
            
            # 设置测评选项
            eval_module.EVALUATION_SWITCHES = {
                "diversity": enable_diversity,
                "consistency": enable_consistency,
                "winrate": enable_winrate
            }
            
            # 调用填空题测评，传入正确的文件路径
            eval_blank_main(
                file1=self.data_files["blank"]["standard"],  # 金标准数据
                file2=generated_file,  # 第一次生成的数据
                file3=second_generated_file,  # 第二次生成的数据（多样性比较用）
                eval_model=evaluation_model,
                start_idx=0,  # 生成的文件索引从0开始
                end_idx=end_index - start_index if end_index else None  # 调整结束索引
            )
            return {
                "success": True,
                "evaluation_file": None,  # 测评模块会直接保存文件，我们稍后获取路径
                "method": "填空题测评"
            }
        except Exception as e:
            logger.error(f"填空题测评失败: {e}")
            raise
    
    def _normalize_evaluation_models(self, evaluation_model):
        """标准化测评模型参数，支持单个或多个模型"""
        if isinstance(evaluation_model, str):
            return [evaluation_model]
        elif isinstance(evaluation_model, (list, tuple)):
            return list(evaluation_model)
        else:
            return ["gpt-4o"]  # 默认值
    
    def _generate_evaluation_file_path(self, base_file: str, model_name: str, timestamp: str, method_name: str = None) -> str:
        """为不同测评模型生成独立的测评文件路径"""
        import os
        
        # 获取基础文件名和目录
        base_dir = os.path.dirname(base_file)
        base_name = os.path.basename(base_file)
        name_without_ext = os.path.splitext(base_name)[0]
        ext = os.path.splitext(base_name)[1]
        
        # 清理模型名称，移除特殊字符
        clean_model_name = model_name.replace(":", "_").replace("/", "_").replace("-", "_")
        
        # 清理方法名称，移除特殊字符
        clean_method_name = ""
        if method_name:
            clean_method_name = method_name.replace(":", "_").replace("/", "_").replace("-", "_").replace(" ", "_")
            clean_method_name = f"{clean_method_name}_"
        
        # 生成新的文件名
        new_filename = f"{name_without_ext}_eval_{clean_method_name}{clean_model_name}_{timestamp}{ext}"
        return os.path.join(base_dir, new_filename)
    
    async def run_complete_pipeline(self, method_id: str, data_file: str = None,
                                  start_index: int = 0, end_index: int = None,
                                  batch_size: int = 1, delay_between_batches: float = 2.0,
                                  use_rag: bool = True, rag_mode: str = "planner",
                                  evaluation_mode: str = "binary",
                                  enable_consistency: bool = True, enable_winrate: bool = True,
                                  enable_diversity: bool = False, run_twice_for_consistency: bool = True,
                                  evaluation_model = "gpt-4o", evaluation_models = None, 
                                  base_model_name = "gemini") -> Dict[str, Any]:
        """运行完整的生成+测评流程"""
        logger.info("=" * 80)
        logger.info("🚀 开始完整流程：生成 + 测评")
        logger.info("=" * 80)
        
        # 验证参数
        is_valid, error_msg = self.validate_inputs(method_id, data_file, start_index, end_index)
        if not is_valid:
            return {"success": False, "error": error_msg}
        
        # 生成题目
        logger.info("📝 步骤1: 生成题目")
        generation_result = await self.generate_questions(
            method_id=method_id,
            data_file=data_file,
            start_index=start_index,
            end_index=end_index,
            batch_size=batch_size,
            delay_between_batches=delay_between_batches,
            use_rag=use_rag,
            rag_mode=rag_mode,
            evaluation_mode=evaluation_mode,
            base_model_name=base_model_name
        )
        
        if not generation_result["success"]:
            return generation_result
        
        # 生成时间戳用于文件命名（需要在第二次生成之前定义）
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 如果需要多样性测评，进行第二次生成
        second_generation_result = None
        if enable_diversity and run_twice_for_consistency:
            logger.info("📝 步骤1.5: 第二次生成（用于多样性测评）")
            
            # 判断是否为V3方法
            if method_id in ["v3_choice", "v3_blank"]:
                # V3方法：直接调用generate_questions
                second_generation_result = await self.generate_questions(
                    method_id=method_id,
                    data_file=data_file,
                    start_index=start_index,
                    end_index=end_index,
                    batch_size=batch_size,
                    delay_between_batches=delay_between_batches,
                    use_rag=use_rag,
                    rag_mode=rag_mode,
                    evaluation_mode=evaluation_mode,
                    base_model_name=base_model_name
                )
            else:
                # 基线方法：直接调用_generate_baseline_questions，传入generation_attempt=2
                second_generation_result = await self._generate_baseline_questions(
                    method_id=method_id,
                    data_file=data_file,
                    start_index=start_index,
                    end_index=end_index,
                    timestamp=timestamp,
                    base_model_name=base_model_name,
                    generation_attempt=2
                )
            
            if not second_generation_result["success"]:
                logger.warning("⚠️ 第二次生成失败，将只使用第一次生成的结果进行测评")
                second_generation_result = None
        
        # 测评题目 - 支持多模型测评
        logger.info("📊 步骤2: 测评题目")
        
        # 标准化测评模型列表
        if evaluation_models is not None:
            # 使用提供的多个测评模型
            evaluation_models = self._normalize_evaluation_models(evaluation_models)
        else:
            # 使用单个测评模型
            evaluation_models = self._normalize_evaluation_models(evaluation_model)
        logger.info(f"🤖 将使用 {len(evaluation_models)} 个测评模型: {evaluation_models}")
        
        # 存储所有模型的测评结果
        evaluation_results = {}
        
        for i, model in enumerate(evaluation_models, 1):
            logger.info(f"📊 步骤2.{i}: 使用测评模型 {model}")
            
            try:
                # 为每个模型生成独立的测评文件路径
                base_generated_file = generation_result["output_file"]
                base_second_file = second_generation_result["output_file"] if second_generation_result else None
                
                # 获取方法名称
                method_name = self.methods[method_id]["name"]
                
                # 生成带模型名称的输入文件路径
                model_generated_file = self._generate_evaluation_file_path(
                    base_generated_file, f"{model}_input_1", timestamp, method_name
                )
                model_second_file = self._generate_evaluation_file_path(
                    base_second_file, f"{model}_input_2", timestamp, method_name
                ) if base_second_file else None
                
                # 复制输入文件到新路径（避免修改原文件）
                import shutil
                shutil.copy2(base_generated_file, model_generated_file)
                if model_second_file:
                    shutil.copy2(base_second_file, model_second_file)
                
                # 使用当前模型进行测评
                model_evaluation_result = self.evaluate_questions(
                    method_id=method_id,
                    generated_file=model_generated_file,
                    second_generated_file=model_second_file,
                    start_index=start_index,
                    end_index=end_index,
                    enable_consistency=enable_consistency,
                    enable_winrate=enable_winrate,
                    enable_diversity=enable_diversity,
                    evaluation_model=model
                )
                
                # 测评完成后，自动获取生成的测评文件路径
                if model_evaluation_result["success"]:
                    # 查找最新生成的测评文件
                    eval_dir = Path("eval/unified_results")
                    if eval_dir.exists():
                        # 按修改时间排序，获取最新的文件
                        eval_files = list(eval_dir.glob("unified_evaluation_*.json"))
                        if eval_files:
                            latest_eval_file = max(eval_files, key=lambda x: x.stat().st_mtime)
                            
                            # 为测评结果文件生成带模型名称的路径
                            model_eval_file = self._generate_evaluation_file_path(
                                str(latest_eval_file), model, timestamp, method_name
                            )
                            
                            # 复制测评结果文件到新路径
                            shutil.copy2(str(latest_eval_file), model_eval_file)
                            model_evaluation_result["evaluation_file"] = model_eval_file
                            
                            logger.info(f"📁 找到测评文件: {latest_eval_file} -> {model_eval_file}")
                        else:
                            logger.warning(f"⚠️ 未找到测评文件，模型: {model}")
                            model_evaluation_result["success"] = False
                            model_evaluation_result["error"] = "未找到测评文件"
                    else:
                        logger.warning(f"⚠️ 测评目录不存在: {eval_dir}")
                        model_evaluation_result["success"] = False
                        model_evaluation_result["error"] = "测评目录不存在"
                
                # 存储结果
                evaluation_results[model] = {
                    "success": model_evaluation_result["success"],
                    "evaluation_file": model_evaluation_result.get("evaluation_file"),
                    "method": f"测评模型: {model}",
                    "error": model_evaluation_result.get("error")
                }
                
                if model_evaluation_result["success"]:
                    logger.info(f"✅ 测评模型 {model} 完成: {model_evaluation_result.get('evaluation_file')}")
                else:
                    logger.error(f"❌ 测评模型 {model} 失败: {model_evaluation_result.get('error', '未知错误')}")
                    
            except Exception as e:
                logger.error(f"❌ 测评模型 {model} 异常: {e}")
                evaluation_results[model] = {
                    "success": False,
                    "error": str(e),
                    "method": f"测评模型: {model}"
                }
        
        # 检查是否有成功的测评
        successful_evaluations = [model for model, result in evaluation_results.items() if result["success"]]
        if not successful_evaluations:
            return {
                "success": False,
                "error": "所有测评模型的测评都失败了",
                "evaluation_results": evaluation_results
            }
        
        # 返回完整结果
        result = {
            "success": True,
            "generation": generation_result,
            "evaluation_results": evaluation_results,  # 多模型测评结果
            "summary": {
                "method": self.methods[method_id]["name"],
                "data_file": data_file or self.data_files[self.methods[method_id]["question_type"]]["standard"],
                "index_range": f"{start_index}-{end_index}",
                "generated_file": generation_result["output_file"],
                "evaluation_models": evaluation_models,
                "successful_evaluations": successful_evaluations,
                "evaluation_files": {
                    model: result["evaluation_file"] 
                    for model, result in evaluation_results.items() 
                    if result["success"]
                }
            }
        }
        
        # 如果有第二次生成，添加到结果中
        if second_generation_result:
            result["second_generation"] = second_generation_result
            result["summary"]["second_generated_file"] = second_generation_result["output_file"]
        
        # 生成统计文件
        try:
            stats_file = self._generate_statistics_file(
                method_id=method_id,
                base_model_name=base_model_name,
                generation_result=generation_result,
                second_generation_result=second_generation_result,
                evaluation_results=evaluation_results,
                start_index=start_index,
                end_index=end_index,
                timestamp=timestamp
            )
            result["statistics_file"] = stats_file
            logger.info(f"📊 统计文件已保存: {stats_file}")
        except Exception as e:
            logger.warning(f"⚠️ 生成统计文件失败: {e}")
        
        # 输出测评结果摘要
        logger.info("=" * 80)
        logger.info("📊 测评结果摘要")
        logger.info("=" * 80)
        for model, eval_result in evaluation_results.items():
            if eval_result["success"]:
                logger.info(f"✅ {model}: {eval_result['evaluation_file']}")
            else:
                logger.error(f"❌ {model}: {eval_result.get('error', '测评失败')}")
        logger.info("=" * 80)
        
        return result

    def _generate_statistics_file(self, method_id: str, base_model_name: str, 
                                generation_result: Dict[str, Any], 
                                second_generation_result: Dict[str, Any],
                                evaluation_results: Dict[str, Any],
                                start_index: int, end_index: int, 
                                timestamp: str) -> str:
        """生成统计文件，包含token使用量、生成题目数量、花费时间等信息"""
        from datetime import datetime
        import json
        import os
        
        # 计算时间统计
        start_time = generation_result.get("start_time")
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds() if start_time else 0
        
        # 计算生成题目数量
        generated_count = 0
        if generation_result.get("success") and generation_result.get("output_file"):
            try:
                with open(generation_result["output_file"], "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict) and "results" in data:
                        generated_count = len(data["results"])
                    elif isinstance(data, list):
                        generated_count = len(data)
            except Exception as e:
                logger.warning(f"⚠️ 无法读取生成文件统计题目数量: {e}")
        
        # 计算第二次生成题目数量
        second_generated_count = 0
        if second_generation_result and second_generation_result.get("success") and second_generation_result.get("output_file"):
            try:
                with open(second_generation_result["output_file"], "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict) and "results" in data:
                        second_generated_count = len(data["results"])
                    elif isinstance(data, list):
                        second_generated_count = len(data)
            except Exception as e:
                logger.warning(f"⚠️ 无法读取第二次生成文件统计题目数量: {e}")
        
        # 计算token使用量（从V3系统获取）
        token_stats = self._extract_token_statistics(generation_result, second_generation_result)
        
        # 计算测评统计
        evaluation_stats = {
            "total_models": len(evaluation_results),
            "successful_models": len([r for r in evaluation_results.values() if r.get("success", False)]),
            "failed_models": len([r for r in evaluation_results.values() if not r.get("success", False)])
        }
        
        # 构建统计信息
        statistics = {
            "run_info": {
                "method_id": method_id,
                "method_name": self.methods[method_id]["name"],
                "base_model_name": base_model_name,
                "timestamp": timestamp,
                "start_time": start_time.isoformat() if start_time else None,
                "end_time": end_time.isoformat(),
                "total_duration_seconds": total_duration,
                "total_duration_formatted": f"{int(total_duration // 3600):02d}:{int((total_duration % 3600) // 60):02d}:{int(total_duration % 60):02d}"
            },
            "generation_stats": {
                "first_generation": {
                    "success": generation_result.get("success", False),
                    "output_file": generation_result.get("output_file"),
                    "questions_generated": generated_count,
                    "start_index": start_index,
                    "end_index": end_index,
                    "range_size": (end_index - start_index + 1) if end_index is not None else "unlimited"
                },
                "second_generation": {
                    "enabled": second_generation_result is not None,
                    "success": second_generation_result.get("success", False) if second_generation_result else False,
                    "output_file": second_generation_result.get("output_file") if second_generation_result else None,
                    "questions_generated": second_generated_count
                },
                "total_questions_generated": generated_count + second_generated_count
            },
            "token_usage": token_stats,
            "evaluation_stats": evaluation_stats,
            "evaluation_files": {
                model: result.get("evaluation_file") 
                for model, result in evaluation_results.items() 
                if result.get("success", False)
            }
        }
        
        # 保存统计文件
        stats_filename = f"statistics_{method_id}_{base_model_name}_{timestamp}.json"
        stats_file_path = os.path.join("outputs", stats_filename)
        
        # 确保outputs目录存在
        os.makedirs("outputs", exist_ok=True)
        
        with open(stats_file_path, "w", encoding="utf-8") as f:
            json.dump(statistics, f, ensure_ascii=False, indent=2)
        
        return stats_file_path
    
    def _extract_token_statistics(self, generation_result: Dict[str, Any], 
                                second_generation_result: Dict[str, Any]) -> Dict[str, Any]:
        """从生成结果中提取token统计信息"""
        token_stats = {
            "first_generation": {
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0,
                "total_tokens": 0,
                "total_cost": 0.0
            },
            "second_generation": {
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0,
                "total_tokens": 0,
                "total_cost": 0.0
            },
            "combined": {
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0,
                "total_tokens": 0,
                "total_cost": 0.0
            }
        }
        
        # 从第一次生成结果提取token信息
        if generation_result.get("success") and generation_result.get("output_file"):
            try:
                with open(generation_result["output_file"], "r", encoding="utf-8") as f:
                    data = json.load(f)
                    
                    # 检查是否是V3格式（包含cost_info字段的数组）
                    if isinstance(data, list) and len(data) > 0:
                        # V3格式：从每个题目的cost_info字段累计
                        total_prompt = 0
                        total_completion = 0
                        total_tokens = 0
                        total_cost = 0.0
                        
                        for item in data:
                            if isinstance(item, dict) and "cost_info" in item:
                                cost_info = item["cost_info"]
                                total_prompt += cost_info.get("prompt_tokens", 0)
                                total_completion += cost_info.get("completion_tokens", 0)
                                total_tokens += cost_info.get("total_tokens", 0)
                                total_cost += cost_info.get("total_cost", 0.0)
                        
                        token_stats["first_generation"] = {
                            "total_prompt_tokens": total_prompt,
                            "total_completion_tokens": total_completion,
                            "total_tokens": total_tokens,
                            "total_cost": total_cost
                        }
                    
                    # 检查是否是baseline格式（包含token_usage字段的字典）
                    elif isinstance(data, dict) and "token_usage" in data:
                        usage = data["token_usage"]
                        token_stats["first_generation"] = {
                            "total_prompt_tokens": usage.get("total_prompt_tokens", 0),
                            "total_completion_tokens": usage.get("total_completion_tokens", 0),
                            "total_tokens": usage.get("total_tokens", 0),
                            "total_cost": usage.get("total_cost", 0.0)
                        }
                        
            except Exception as e:
                logger.warning(f"⚠️ 无法从第一次生成文件提取token统计: {e}")
        
        # 从第二次生成结果提取token信息
        if second_generation_result and second_generation_result.get("success") and second_generation_result.get("output_file"):
            try:
                with open(second_generation_result["output_file"], "r", encoding="utf-8") as f:
                    data = json.load(f)
                    
                    # 检查是否是V3格式（包含cost_info字段的数组）
                    if isinstance(data, list) and len(data) > 0:
                        # V3格式：从每个题目的cost_info字段累计
                        total_prompt = 0
                        total_completion = 0
                        total_tokens = 0
                        total_cost = 0.0
                        
                        for item in data:
                            if isinstance(item, dict) and "cost_info" in item:
                                cost_info = item["cost_info"]
                                total_prompt += cost_info.get("prompt_tokens", 0)
                                total_completion += cost_info.get("completion_tokens", 0)
                                total_tokens += cost_info.get("total_tokens", 0)
                                total_cost += cost_info.get("total_cost", 0.0)
                        
                        token_stats["second_generation"] = {
                            "total_prompt_tokens": total_prompt,
                            "total_completion_tokens": total_completion,
                            "total_tokens": total_tokens,
                            "total_cost": total_cost
                        }
                    
                    # 检查是否是baseline格式（包含token_usage字段的字典）
                    elif isinstance(data, dict) and "token_usage" in data:
                        usage = data["token_usage"]
                        token_stats["second_generation"] = {
                            "total_prompt_tokens": usage.get("total_prompt_tokens", 0),
                            "total_completion_tokens": usage.get("total_completion_tokens", 0),
                            "total_tokens": usage.get("total_tokens", 0),
                            "total_cost": usage.get("total_cost", 0.0)
                        }
                        
            except Exception as e:
                logger.warning(f"⚠️ 无法从第二次生成文件提取token统计: {e}")
        
        # 计算合计
        token_stats["combined"] = {
            "total_prompt_tokens": token_stats["first_generation"]["total_prompt_tokens"] + token_stats["second_generation"]["total_prompt_tokens"],
            "total_completion_tokens": token_stats["first_generation"]["total_completion_tokens"] + token_stats["second_generation"]["total_completion_tokens"],
            "total_tokens": token_stats["first_generation"]["total_tokens"] + token_stats["second_generation"]["total_tokens"],
            "total_cost": token_stats["first_generation"]["total_cost"] + token_stats["second_generation"]["total_cost"]
        }
        
        return token_stats
    

def create_cli_interface():
    """创建命令行界面"""
    parser = argparse.ArgumentParser(description="统一生成和测评系统")
    
    # 基本参数
    parser.add_argument("--method", "-m", required=True, 
                       help="生成方法ID (使用 --list-methods 查看所有方法)")
    parser.add_argument("--data-file", "-d", 
                       help="数据文件路径 (可选，默认使用对应题型的标准数据)")
    parser.add_argument("--start-index", "-s", type=int, default=0,
                       help="开始索引 (默认: 0)")
    parser.add_argument("--end-index", "-e", type=int,
                       help="结束索引 (可选)")
    
    # 生成参数
    parser.add_argument("--batch-size", "-b", type=int, default=1,
                       help="批处理大小 (默认: 1, 适配Qwen-72B rate limit)")
    parser.add_argument("--delay", type=float, default=2.0,
                       help="批次间延迟秒数 (默认: 2.0, RPM:1000, TPM:20000)")
    parser.add_argument("--no-rag", action="store_true",
                       help="禁用RAG")
    parser.add_argument("--rag-mode", choices=["planner", "writer", "writer_only", "planner_kg"],
                       default="planner", help="RAG模式 (默认: planner)")
    parser.add_argument("--evaluation-mode", choices=["score", "binary"],
                       default="binary", help="评估模式 (默认: binary)")
    
    # 测评参数
    parser.add_argument("--no-consistency", action="store_true",
                       help="禁用一致性测评")
    parser.add_argument("--no-winrate", action="store_true",
                       help="禁用Win Rate测评")
    parser.add_argument("--enable-diversity", action="store_true",
                       help="启用多样性测评")
    parser.add_argument("--evaluation-model", nargs="+", default=["gpt-4o"],
                       help="测评使用的模型名称，支持多个模型 (默认: gpt-4o)")
    parser.add_argument("--base-model-name", choices=["gemini", "qwen-72b", "gpt-4o-mini"], default="gemini",
                       help="基础模型名称 (默认: gemini)")
    
    # 操作模式
    parser.add_argument("--generate-only", action="store_true",
                       help="仅生成题目，不进行测评")
    parser.add_argument("--evaluate-only", action="store_true",
                       help="仅测评已有文件")
    parser.add_argument("--generated-file", 
                       help="要测评的生成文件路径 (用于 --evaluate-only)")
    
    # 信息查看
    parser.add_argument("--list-methods", action="store_true",
                       help="列出所有可用的生成方法")
    parser.add_argument("--list-data", action="store_true",
                       help="列出所有可用的数据文件")
    
    return parser

async def main():
    """主函数"""
    parser = create_cli_interface()
    args = parser.parse_args()
    
    # 创建统一系统
    system = UnifiedGenerationEvaluation()
    
    # 处理信息查看命令
    if args.list_methods:
        system.list_methods()
        return
    
    if args.list_data:
        system.list_data_files()
        return
    
    # 验证必要参数
    if not args.method:
        print("❌ 错误: 必须指定生成方法 (--method)")
        print("使用 --list-methods 查看所有可用方法")
        return
    
    # 仅测评模式
    if args.evaluate_only:
        if not args.generated_file:
            print("❌ 错误: --evaluate-only 需要指定 --generated-file")
            return
        
        result = system.evaluate_questions(
            method_id=args.method,
            generated_file=args.generated_file,
            start_index=args.start_index,
            end_index=args.end_index,
            enable_consistency=not args.no_consistency,
            enable_winrate=not args.no_winrate,
            enable_diversity=args.enable_diversity,
            evaluation_model=args.evaluation_model
        )
        
        if result["success"]:
            print("✅ 测评完成!")
            print(f"📊 测评文件: {result['evaluation_file']}")
        else:
            print(f"❌ 测评失败: {result['error']}")
        return
    
    # 生成模式
    if args.generate_only:
        result = await system.generate_questions(
            method_id=args.method,
            data_file=args.data_file,
            start_index=args.start_index,
            end_index=args.end_index,
            batch_size=args.batch_size,
            delay_between_batches=args.delay,
            use_rag=not args.no_rag,
            rag_mode=args.rag_mode,
            evaluation_mode=args.evaluation_mode,
            base_model_name=args.base_model_name
        )
        
        if result["success"]:
            print("✅ 生成完成!")
            print(f"📝 生成文件: {result['output_file']}")
            if result.get("workflow_file"):
                print(f"📋 Workflow文件: {result['workflow_file']}")
        else:
            print(f"❌ 生成失败: {result['error']}")
        return
    
    # 完整流程
    result = await system.run_complete_pipeline(
        method_id=args.method,
        data_file=args.data_file,
        start_index=args.start_index,
        end_index=args.end_index,
        batch_size=args.batch_size,
        delay_between_batches=args.delay,
        use_rag=not args.no_rag,
        rag_mode=args.rag_mode,
        evaluation_mode=args.evaluation_mode,
        enable_consistency=not args.no_consistency,
        enable_winrate=not args.no_winrate,
        enable_diversity=args.enable_diversity,
        evaluation_model=args.evaluation_model,
        base_model_name=args.base_model_name
    )
    
    if result["success"]:
        print("✅ 完整流程完成!")
        print(f"🤖 使用的基础模型: {args.base_model_name}")
        print(f"📝 生成文件: {result['generation']['output_file']}")
        if result["generation"].get("workflow_file"):
            print(f"📋 Workflow文件: {result['generation']['workflow_file']}")
        
        # 显示多模型测评结果
        if "evaluation_results" in result:
            print("📊 测评结果:")
            for model, eval_result in result["evaluation_results"].items():
                if eval_result["success"]:
                    print(f"  ✅ {model}: {eval_result['evaluation_file']}")
                else:
                    print(f"  ❌ {model}: {eval_result.get('error', '测评失败')}")
        elif "evaluation" in result and result["evaluation"].get("evaluation_file"):
            print(f"📊 测评文件: {result['evaluation']['evaluation_file']}")
        
        # 显示统计文件
        if "statistics_file" in result:
            print(f"📊 统计文件: {result['statistics_file']}")
    else:
        print(f"❌ 流程失败: {result['error']}")
        if "evaluation_results" in result:
            print("📊 测评失败详情:")
            for model, eval_result in result["evaluation_results"].items():
                print(f"  ❌ {model}: {eval_result.get('error', '测评失败')}")

if __name__ == "__main__":
    print("🔍 调试信息: 开始执行主程序")
    print("🔍 调试信息: 这是 unified_generation_evaluation.py 文件")
    
    try:
        # ===== 手动设置参数区域 =====
        # 如果你想直接在编辑器中运行，可以修改下面的参数
        MANUAL_RUN = True  # 设置为 True 启用手动参数模式
        
        print(f"🔍 调试信息: MANUAL_RUN = {MANUAL_RUN}")
        
        if MANUAL_RUN:
            print("🔍 调试信息: 进入手动参数模式")
            # ===== 手动设置参数区域 =====
            # 你可以修改下面的参数来定制运行行为
            manual_params = {
                # === 基本参数 ===
                "method_id": "react_choice",  # 生成方法ID
                # 可选值: v3_choice, v3_blank, cot_choice, cot_blank, cot_n_choice, cot_n_blank, react_choice, react_blank
                
                "start_index": 0,          # 开始索引 (从第几道题开始)
                "end_index": 2,           # 结束索引 (到第几道题结束，None表示处理到最后)
                
                # === 生成参数 ===
                "batch_size": 5,           # 批处理大小 (每次处理多少道题) - 适配Qwen-72B rate limit
                "delay_between_batches": 1.0,  # 批次间延迟秒数 (避免API限制) - RPM:1000, TPM:20000
                
                # === RAG参数 ===
                "use_rag": True,           # 是否使用RAG (检索增强生成)
                "rag_mode": "writer",     # RAG模式: planner, writer, writer_only, planner_kg
                "evaluation_mode": "binary",  # 评估模式: score, binary
                
                # === 测评参数 ===
                "enable_consistency": True,    # 是否启用一致性测评
                "enable_winrate": True,       # 是否启用Win Rate测评
                "enable_diversity": True,     # 是否启用多样性测评
                "run_twice_for_consistency": True,  # 是否进行两次生成以测评多样性
                "evaluation_models": ["gpt-4o", "deepseek-v3"],  # 测评使用的模型，支持多个模型
                "base_model_name": "gpt-4o-mini",  # 基础模型名称: gemini, qwen-72b 或 gpt-4o-mini
                
                # === 数据文件 ===
                "data_file": None  # 数据文件路径，None表示使用默认数据文件
                # 选择题默认: D:\CODE\three_0921\data\choice_unquie_500.jsonl
                # 填空题默认: D:\CODE\three_0921\data\blank_unique.jsonl
            }
            
            async def manual_run():
                """手动运行模式"""
                print("🚀 使用手动参数模式运行...")
                print(f"📋 方法: {manual_params['method_id']}")
                print(f"📊 范围: {manual_params['start_index']}-{manual_params['end_index']}")
                print(f"🔧 批处理大小: {manual_params['batch_size']}")
                print(f"🔧 RAG模式: {manual_params['rag_mode']}")
                print("=" * 50)
                
                system = UnifiedGenerationEvaluation()
                result = await system.run_with_params(**manual_params)
                
                if result["success"]:
                    print("✅ 完整流程完成!")
                    print(f"🤖 使用的基础模型: {manual_params['base_model_name']}")
                    print(f"📝 第一次生成文件: {result['generation']['output_file']}")
                    if result.get("second_generation"):
                        print(f"📝 第二次生成文件: {result['second_generation']['output_file']}")
                    if result["generation"].get("workflow_file"):
                        print(f"📋 Workflow文件: {result['generation']['workflow_file']}")
                    
                    # 显示多模型测评结果
                    if "evaluation_results" in result:
                        print("📊 测评结果:")
                        for model, eval_result in result["evaluation_results"].items():
                            if eval_result["success"]:
                                print(f"  ✅ {model}: {eval_result['evaluation_file']}")
                            else:
                                print(f"  ❌ {model}: {eval_result.get('error', '测评失败')}")
                    elif "evaluation" in result and result["evaluation"].get("evaluation_file"):
                        print(f"📊 测评文件: {result['evaluation']['evaluation_file']}")
                    
                    # 显示统计文件
                    if "statistics_file" in result:
                        print(f"📊 统计文件: {result['statistics_file']}")
                else:
                    print(f"❌ 流程失败: {result['error']}")
                    if "evaluation_results" in result:
                        print("📊 测评失败详情:")
                        for model, eval_result in result["evaluation_results"].items():
                            print(f"  ❌ {model}: {eval_result.get('error', '测评失败')}")
                
                return result
            
            asyncio.run(manual_run())
        else:
            # 使用命令行参数模式
            asyncio.run(main())
    except Exception as e:
        print(f"❌ 调试信息: 主程序执行出错: {e}")
        import traceback
        traceback.print_exc()
