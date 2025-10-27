#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多智能体系统V3 - 基于100条数据批量生成题目
"""

import asyncio
import random
import httpx
import logging
import os
import re
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from tqdm.asyncio import tqdm

from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

# 导入配置
from config import (
    AGENT_CONFIGS, 
    get_model_config, 
    LOGGING_CONFIG,
    RETRY_CONFIG,
    validate_config
)
from prompts import (
    build_planner_user_prompt,
    build_planner_user_prompt_with_teacher_feedback,
    build_writer_user_prompt_for_first,
    build_writer_user_prompt_for_rewrite,
    build_teacher_checker_prompt,
)
from lightweight_parser import LightweightParser
# 评分模式已拆分到两个独立文件：
# - prompts_score.py（数值评分模式：S/E 为浮点分数）
# - prompts_binary.py（二值评估模式：各维度0/1，且需 PASS=1）

from prompts_score import (
    build_solver_user_prompt,
    build_educator_user_prompt,
)
from prompts_binary import (
    build_solver_user_prompt_binary,
    build_educator_user_prompt_binary,
)

# 导入知识检索器
from knowledge_retriever import KnowledgeRetriever


# 配置日志
logging.basicConfig(
    level=getattr(logging, LOGGING_CONFIG["level"]),
    format=LOGGING_CONFIG["format"],
    handlers=[
        logging.FileHandler(LOGGING_CONFIG["file"], encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)

# 降低第三方库在控制台的输出噪音（仅写入文件）
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("autogen_core").setLevel(logging.WARNING)
logging.getLogger("autogen_ext").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)

# 控制台最小流程输出：仅本模块的关键信息
_console_handler = logging.StreamHandler()
_console_handler.setLevel(logging.INFO)
_console_handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
logger.addHandler(_console_handler)
logger.propagate = False


# ==================== 模型客户端管理器 ====================

class ModelClientManager:
    """模型客户端管理器"""
    
    def __init__(self):
        self.clients = {}
        self._initialize_clients()
    
    def _initialize_clients(self):
        """初始化所有模型客户端"""
        try:
            # 外部模型客户端
            external_config = get_model_config("external")
            self.clients["external"] = OpenAIChatCompletionClient(
                model=external_config["model"],
                base_url=external_config["base_url"],
                api_key=external_config["api_key"],
                model_info=external_config["model_info"],
                timeout=httpx.Timeout(
                    external_config["timeout"],
                    connect=300.0,
                    read=600.0
                ),
                max_retries=0,
                temperature=external_config.get("temperature", 1.0),
                top_p=external_config.get("top_p", 1.0)
            )
            logger.info("外部模型客户端初始化成功")
            
            # 外部模型客户端（Planner专用，发散规划） - 若配置存在则初始化
            try:
                external_planner_config = get_model_config("external_planner")
                self.clients["external_planner"] = OpenAIChatCompletionClient(
                    model=external_planner_config["model"],
                    base_url=external_planner_config["base_url"],
                    api_key=external_planner_config["api_key"],
                    model_info=external_planner_config["model_info"],
                    timeout=httpx.Timeout(
                        external_planner_config["timeout"],
                        connect=300.0,
                        read=600.0
                    ),
                    max_retries=0,
                    # 传递采样参数（仅支持Gemini模型的参数）
                    temperature=external_planner_config.get("temperature"),
                    top_p=external_planner_config.get("top_p"),
                )
                logger.info("外部模型客户端（Planner）初始化成功")
            except Exception:
                # 允许无此配置时跳过
                pass
            
            # 外部模型客户端（Writer专用，更发散） - 若配置存在则初始化
            try:
                external_writer_config = get_model_config("external_writer")
                self.clients["external_writer"] = OpenAIChatCompletionClient(
                    model=external_writer_config["model"],
                    base_url=external_writer_config["base_url"],
                    api_key=external_writer_config["api_key"],
                    model_info=external_writer_config["model_info"],
                    timeout=httpx.Timeout(
                        external_writer_config["timeout"],
                        connect=300.0,
                        read=600.0
                    ),
                    max_retries=0,
                    # 传递采样参数（仅支持Gemini模型的参数）
                    temperature=external_writer_config.get("temperature"),
                    top_p=external_writer_config.get("top_p"),
                )
                logger.info("外部模型客户端（Writer）初始化成功")
            except Exception:
                # 允许无此配置时跳过
                pass

            # 外部模型客户端（Educator专用，gemini-2.5-flash） - 若配置存在则初始化
            try:
                external_educator_config = get_model_config("external_educator")
                self.clients["external_educator"] = OpenAIChatCompletionClient(
                    model=external_educator_config["model"],
                    base_url=external_educator_config["base_url"],
                    api_key=external_educator_config["api_key"],
                    model_info=external_educator_config["model_info"],
                    timeout=httpx.Timeout(
                        external_educator_config["timeout"],
                        connect=300.0,
                        read=600.0
                    ),
                    max_retries=0,
                    temperature=external_educator_config.get("temperature", 1.0),
                    top_p=external_educator_config.get("top_p", 1.0)
                )
                logger.info("外部模型客户端（Educator）初始化成功")
            except Exception:
                # 允许无此配置时跳过
                pass

            # 外部模型客户端（Solver专用，gemini-2.5-flash） - 若配置存在则初始化
            try:
                external_solver_config = get_model_config("external_solver")
                self.clients["external_solver"] = OpenAIChatCompletionClient(
                    model=external_solver_config["model"],
                    base_url=external_solver_config["base_url"],
                    api_key=external_solver_config["api_key"],
                    model_info=external_solver_config["model_info"],
                    timeout=httpx.Timeout(
                        external_solver_config["timeout"],
                        connect=180.0,  # 增加连接超时到3分钟
                        read=300.0      # 增加读取超时到5分钟
                    ),
                    max_retries=0,
                    temperature=external_solver_config.get("temperature", 1.0),
                    top_p=external_solver_config.get("top_p", 1.0)
                )
                logger.info("外部模型客户端（Solver）初始化成功")
            except Exception:
                # 允许无此配置时跳过
                pass

            # 本地模型客户端
            local_config = get_model_config("local")
            self.clients["local"] = OpenAIChatCompletionClient(
                model=local_config["model"],
                base_url=local_config["base_url"],
                api_key=local_config["api_key"],
                model_info=local_config["model_info"],
                timeout=httpx.Timeout(
                    local_config["timeout"],
                    connect=300.0,
                    read=600.0
                ),
                max_retries=0,
                temperature=local_config.get("temperature", 1.0),
                top_p=local_config.get("top_p", 1.0)
            )
            logger.info("本地模型客户端初始化成功")
            
        except Exception as e:
            logger.error(f"模型客户端初始化失败: {e}")
            # 如果是 API Key 相关错误，记录但不抛出异常，让调用方处理重试
            if "API Key not found" in str(e) or "API key" in str(e).lower() or "authentication" in str(e).lower():
                logger.warning("检测到 API Key 认证问题，将在重试时重新初始化客户端")
                # 不抛出异常，让调用方可以重试
                return
            raise
    
    def get_client(self, model_type: str) -> OpenAIChatCompletionClient:
        """获取指定类型的模型客户端"""
        if model_type not in self.clients:
            # 尝试重新初始化客户端
            logger.info(f"客户端 {model_type} 不存在，尝试重新初始化...")
            self._initialize_clients()
            
            if model_type not in self.clients:
                raise ValueError(f"未知的模型类型: {model_type}")
        
        return self.clients[model_type]
    
    async def close_all(self):
        """关闭所有客户端连接"""
        for client in self.clients.values():
            try:
                await client.close()
                logger.info("模型客户端连接已关闭")
            except Exception as e:
                logger.error(f"关闭客户端连接时出错: {e}")

# ==================== 智能体管理器 ====================

class AgentManager:
    """智能体管理器"""
    
    def __init__(self, model_manager: ModelClientManager):
        self.model_manager = model_manager
        self.agents = {}
        self._initialize_agents()
    
    def _initialize_agents(self):
        """初始化所有智能体"""
        try:
            for agent_name, agent_config in AGENT_CONFIGS.items():
                # 获取对应的模型客户端
                model_type = agent_config["model_type"]
                model_client = self.model_manager.get_client(model_type)
                
                # 创建智能体（不再使用独立system prompt，统一合并到各自的user prompt中）
                system_msg = ""
                agent = AssistantAgent(
                    name=agent_config["name"],
                    model_client=model_client,
                    system_message=system_msg,
                    reflect_on_tool_use=agent_config["reflect_on_tool_use"],
                    model_client_stream=agent_config["model_client_stream"],
                )
                
                self.agents[agent_name] = agent
                logger.info(f"智能体 {agent_config['name']} 初始化成功")
                
        except Exception as e:
            logger.error(f"智能体初始化失败: {e}")
            raise
    
    def get_agent(self, agent_name: str) -> AssistantAgent:
        """获取指定名称的智能体"""
        if agent_name not in self.agents:
            raise ValueError(f"未知的智能体: {agent_name}")
        return self.agents[agent_name]
    
    def recreate_agent(self, agent_name: str):
        """重新创建指定智能体，确保对话无历史（单轮）。"""
        if agent_name not in AGENT_CONFIGS:
            raise ValueError(f"未知的智能体: {agent_name}")
        agent_config = AGENT_CONFIGS[agent_name]
        model_type = agent_config["model_type"]
        model_client = self.model_manager.get_client(model_type)
        system_msg = ""
        agent = AssistantAgent(
            name=agent_config["name"],
            model_client=model_client,
            system_message=system_msg,
            reflect_on_tool_use=agent_config["reflect_on_tool_use"],
            model_client_stream=agent_config["model_client_stream"],
        )
        self.agents[agent_name] = agent
        logger.info(f"智能体 {agent_config['name']} 已重建（单轮对话，无历史）")

    def recreate_all_agents(self):
        """重建所有已配置的智能体。"""
        for agent_name in AGENT_CONFIGS.keys():
            try:
                self.recreate_agent(agent_name)
            except Exception as e:
                logger.warning(f"重建智能体 {agent_name} 失败：{e}")
    
    def get_all_agents(self) -> Dict[str, AssistantAgent]:
        """获取所有智能体"""
        return self.agents


# ==================== 批量题目生成器 ====================

class BatchQuestionGenerator:
    """批量题目生成器 - 基于100条数据生成题目"""
    
    def __init__(self, agent_manager: AgentManager, use_rag: bool = True, rag_mode: str = "planner", evaluation_mode: str = "score", start_timestamp: str = None):
        self.agent_manager = agent_manager
        self.use_rag = use_rag  # RAG开关（True=开启知识检索，False=关闭）
        # RAG模式：
        # - "planner"：规划阶段检索（默认）
        # - "planner_kg"：仅用知识图谱/课标
        # - "writer"：编写阶段检索题目样例
        # - "writer_only"：跳过规划，直接由Writer+样例生成
        self.rag_mode = rag_mode
        # 评分模式：
        # - "score"：数值评分（Solver/Educator 输出 S/E 分数）
        # - "binary"：二值评估（各维度0/1，且 PASS=1 才通过）
        self.evaluation_mode = evaluation_mode
        self.knowledge_retriever = KnowledgeRetriever() if use_rag else None
        self.generated_questions = []
        self.workflow_data = []  # 保存workflow信息
        
        # 轻量级解析器
        self.lightweight_parser = LightweightParser()
        self.use_lightweight_parsing = True  # 开关，可以控制是否使用轻量级解析
        
        # 使用相对于当前文件的路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.output_dir = os.path.join(current_dir, "outputs")
        self._ensure_output_dir()
        
        # 使用传入的开始时间戳，如果没有则使用当前时间
        if start_timestamp is None:
            start_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.start_timestamp = start_timestamp
        
        # 实时题目输出文件（必须在 output_dir 初始化之后设置）
        self.generated_realtime_path = os.path.join(self.output_dir, f"generated_questions_realtime_{self.start_timestamp}.json")
        
        # 统计信息
        self.stats = {
            "total_questions": 0,
            "successful_questions": 0,
            "failed_questions": 0,
            "rewrite_attempts": 0,
            "replan_attempts": 0,
            "questions_requiring_rewrite": 0,
            "questions_requiring_replan": 0
        }
    
    
    def _ensure_output_dir(self):
        """确保输出目录存在"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    
    def _save_generated_questions(self, questions: List[Dict[str, Any]], output_file: str):
        """实时保存生成的题目"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(questions, f, ensure_ascii=False, indent=2)
            logger.debug(f"已保存 {len(questions)} 条题目到 {output_file}")
        except Exception as e:
            logger.error(f"保存题目失败: {e}")

    def _upsert_generated_question(self, que_id: str, entry: Dict[str, Any]):
        """以 que_id 作为键，插入或更新一条题目，并实时写入文件。"""
        try:
            # 去重更新
            self.generated_questions = [q for q in self.generated_questions if q.get("que_id") != que_id]
            self.generated_questions.append(entry)
            # 实时落盘
            self._save_generated_questions(self.generated_questions, self.generated_realtime_path)
        except Exception as e:
            logger.warning(f"实时写入题目失败（忽略继续）：{e}")
    
    def load_questions_data(self, jsonl_file_path: str, limit: int = None, start_index: int = 0, end_index: int = None, indices: List[int] = None) -> List[Dict[str, Any]]:
        """
        加载题目数据
        
        Args:
            jsonl_file_path: 数据文件路径
            limit: 限制数量（优先级最低，仅当其他参数都为None时生效）
            start_index: 开始索引（从0开始）
            end_index: 结束索引（不包含，如果为None则到文件末尾）
            indices: 特定索引列表（如[0,1,3,5]，优先级最高）
        
        优先级：indices > start_index/end_index > limit
        """
        questions = []
        try:
            with open(jsonl_file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    try:
                        question = json.loads(line.strip())
                        
                        # 参数优先级：indices > start_index/end_index > limit
                        if indices is not None:
                            # 使用特定索引模式
                            if line_num in indices:
                                questions.append(question)
                        elif start_index is not None and end_index is not None:
                            # 使用区间模式
                            if line_num >= start_index and line_num < end_index:
                                questions.append(question)
                            elif line_num >= end_index:
                                break
                        elif limit is not None:
                            # 使用limit模式
                            questions.append(question)
                            if len(questions) >= limit:
                                break
                        else:
                            # 默认加载所有
                            questions.append(question)
                            
                    except json.JSONDecodeError:
                        continue
            
            # 记录加载结果
            if indices is not None:
                logger.info(f"成功加载指定索引 {indices} 的题目数据，共 {len(questions)} 条")
            elif start_index is not None and end_index is not None:
                logger.info(f"成功加载第 {start_index+1}-{end_index} 条题目数据，共 {len(questions)} 条")
            elif limit is not None:
                logger.info(f"成功加载前 {len(questions)} 条题目数据（limit={limit}）")
            else:
                logger.info(f"成功加载 {len(questions)} 条题目数据")
            
            return questions
            
        except Exception as e:
            logger.error(f"加载题目数据失败: {e}")
            raise
    
    def extract_input_fields(self, question_data: Dict[str, Any]) -> Dict[str, Any]:
        """提取输入字段：grade, difficulty, competence, knowledge, question_type"""
        # 直接使用数据中提供的题型，如果没有则默认为选择题
        question_type = question_data.get("question_type", "选择题")
        return {
            "grade": question_data.get("grade", ""),
            "difficulty": question_data.get("difficulty", ""),
            "competence": question_data.get("competence", []),
            "knowledge": question_data.get("knowledge", ""),
            "question_type": question_type,
            "que_id": question_data.get("que_id", "")
        }

    def _is_education_goal_complete(self, input_data: Dict[str, Any]) -> bool:
        """校验教育目标是否完整：年级/难度/知识点/核心素养均需存在"""
        grade_ok = bool(input_data.get("grade"))
        difficulty_ok = bool(input_data.get("difficulty"))
        knowledge_ok = bool(input_data.get("knowledge"))
        competence = input_data.get("competence")
        competence_ok = isinstance(competence, (list, tuple)) and len(competence) > 0
        return grade_ok and difficulty_ok and knowledge_ok and competence_ok
    
    def _save_workflow_step(self, que_id: str, stage: str, prompt: str, response: str, token_usage: dict = None):
        """保存workflow步骤"""
        # 计算 prompt 的 SHA1 摘要，便于排查重复/改写
        try:
            import hashlib as _hashlib
            prompt_text = prompt if isinstance(prompt, str) else str(prompt)
            prompt_sha1 = _hashlib.sha1(prompt_text.encode('utf-8', errors='ignore')).hexdigest()
        except Exception:
            prompt_sha1 = ""

        workflow_step = {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "que_id": que_id,
            "stage": stage,
            "prompt": prompt,
            "prompt_sha1": prompt_sha1,
            "response": response
        }
        
        # 添加token使用量信息
        if token_usage:
            workflow_step["token_usage"] = token_usage
        else:
            workflow_step["token_usage"] = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "total_cost": 0.0
            }
        
        self.workflow_data.append(workflow_step)
        
        # 立即保存workflow到文件（实时保存）
        self._save_workflow_immediately()

    def _save_workflow_immediately(self):
        """立即保存workflow到文件（实时保存）"""
        try:
            # 使用带时间戳的实时文件
            realtime_path = os.path.join(self.output_dir, f"workflow_realtime_{self.start_timestamp}.json")
            with open(realtime_path, 'w', encoding='utf-8') as f:
                json.dump(self.workflow_data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.warning(f"实时保存workflow失败: {e}")

    def _extract_token_usage(self, response, stage: str = None) -> dict:
        """从API响应中提取token使用量和成本信息；根据 stage 选择计费档位"""
        try:
            # 检查是否有usage信息
            if hasattr(response, 'usage') and response.usage:
                usage_data = {
                    "prompt_tokens": getattr(response.usage, 'prompt_tokens', 0),
                    "completion_tokens": getattr(response.usage, 'completion_tokens', 0),
                    "total_tokens": getattr(response.usage, 'total_tokens', 0)
                }
                
                # 尝试提取成本信息
                if hasattr(response.usage, 'total_cost'):
                    usage_data["total_cost"] = getattr(response.usage, 'total_cost', 0)
                elif hasattr(response.usage, 'cost'):
                    usage_data["total_cost"] = getattr(response.usage, 'cost', 0)
                else:
                    # 如果没有直接的成本信息，使用估算（按阶段定价）
                    tier = "2.0" if (stage and ("planner" in stage.lower() or "teacher" in stage.lower())) else "2.5"
                    usage_data["total_cost"] = self._calculate_cost_estimate_tier(
                        usage_data["prompt_tokens"], usage_data["completion_tokens"], tier
                    )
                
                return usage_data
            
            # 检查messages中是否有usage信息
            if hasattr(response, 'messages') and response.messages:
                for message in response.messages:
                    # 检查models_usage属性（AutoGen使用这个）
                    if hasattr(message, 'models_usage') and message.models_usage:
                        usage_data = {
                            "prompt_tokens": getattr(message.models_usage, 'prompt_tokens', 0),
                            "completion_tokens": getattr(message.models_usage, 'completion_tokens', 0),
                            "total_tokens": getattr(message.models_usage, 'total_tokens', 0)
                        }
                        
                        # 如果没有total_tokens，计算总和
                        if usage_data["total_tokens"] == 0:
                            usage_data["total_tokens"] = usage_data["prompt_tokens"] + usage_data["completion_tokens"]
                        
                        # 尝试提取成本信息
                        if hasattr(message.models_usage, 'total_cost'):
                            usage_data["total_cost"] = getattr(message.models_usage, 'total_cost', 0)
                        elif hasattr(message.models_usage, 'cost'):
                            usage_data["total_cost"] = getattr(message.models_usage, 'cost', 0)
                        else:
                            tier = "2.0" if (stage and ("planner" in stage.lower() or "teacher" in stage.lower())) else "2.5"
                            usage_data["total_cost"] = self._calculate_cost_estimate_tier(
                                usage_data["prompt_tokens"], usage_data["completion_tokens"], tier
                            )
                        
                        return usage_data
                    
                    # 也检查传统的usage属性（向后兼容）
                    if hasattr(message, 'usage') and message.usage:
                        usage_data = {
                            "prompt_tokens": getattr(message.usage, 'prompt_tokens', 0),
                            "completion_tokens": getattr(message.usage, 'completion_tokens', 0),
                            "total_tokens": getattr(message.usage, 'total_tokens', 0)
                        }
                        
                        # 如果没有total_tokens，计算总和
                        if usage_data["total_tokens"] == 0:
                            usage_data["total_tokens"] = usage_data["prompt_tokens"] + usage_data["completion_tokens"]
                        
                        # 尝试提取成本信息
                        if hasattr(message.usage, 'total_cost'):
                            usage_data["total_cost"] = getattr(message.usage, 'total_cost', 0)
                        elif hasattr(message.usage, 'cost'):
                            usage_data["total_cost"] = getattr(message.usage, 'cost', 0)
                        else:
                            tier = "2.0" if (stage and ("planner" in stage.lower() or "teacher" in stage.lower())) else "2.5"
                            usage_data["total_cost"] = self._calculate_cost_estimate_tier(
                                usage_data["prompt_tokens"], usage_data["completion_tokens"], tier
                            )
                        
                        return usage_data
            
            # 如果没有找到usage信息，返回0
            return {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "total_cost": 0.0
            }
            
        except Exception as e:
            logger.warning(f"提取token使用量失败: {e}")
            return {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "total_cost": 0.0
            }
    
    def _calculate_cost_estimate_tier(self, prompt_tokens: int, completion_tokens: int, tier: str) -> float:
        """按计费档位计算成本估算。
        tier: "2.0" 或 "2.5"
        """
        if tier == "2.0":
            # 2.0 定价：prompt $0.1500 / 1M；completion $0.6000 / 1M
            in_per_1k = 0.00015
            out_per_1k = 0.0006
        else:
            # 2.5 定价：prompt $0.3000 / 1M；completion $2.5000 / 1M
            in_per_1k = 0.0003
            out_per_1k = 0.0025
        return (prompt_tokens / 1000) * in_per_1k + (completion_tokens / 1000) * out_per_1k

    def _extract_solver_score(self, solver_content: str) -> float:
        """从solver回复中提取综合得分S"""
        try:
            # 查找"###评分：S=X.X分"模式
            import re
            score_match = re.search(r'###评分：S=(\d+\.?\d*)分', solver_content)
            if score_match:
                return float(score_match.group(1))
            
            # 查找"S=X.X分"模式
            score_match = re.search(r'S=(\d+\.?\d*)分', solver_content)
            if score_match:
                return float(score_match.group(1))
            
            # 查找数字模式
            score_match = re.search(r'(\d+\.?\d*)分', solver_content)
            if score_match:
                return float(score_match.group(1))
                
            return 0.0
        except:
            return 0.0
    
    def _extract_educator_score(self, educator_content: str) -> float:
        """从educator回复中提取综合得分E"""
        try:
            # 查找"###评分：E=X.X分"模式
            import re
            score_match = re.search(r'###评分：E=(\d+\.?\d*)分', educator_content)
            if score_match:
                return float(score_match.group(1))
            
            # 查找"E=X.X分"模式
            score_match = re.search(r'E=(\d+\.?\d*)分', educator_content)
            if score_match:
                return float(score_match.group(1))
            
            # 查找数字模式
            score_match = re.search(r'(\d+\.?\d*)分', educator_content)
            if score_match:
                return float(score_match.group(1))
                
            return 0.0
        except:
            return 0.0

    def _extract_three_questions_scores(self, solver_content: str, educator_content: str) -> Dict[str, Any]:
        """从solver和educator回复中提取三条题目的评分、PASS和RANK信息"""
        import re
        
        result = {
            "questions": [],
            "selected_question": None,
            "selection_reason": ""
        }
        
        try:
            # 解析新的JSON格式
            import json
            
            # 提取solver数据
            solver_data = {}
            try:
                # 尝试从solver_content中提取JSON
                json_match = re.search(r'\{.*\}', solver_content, re.DOTALL)
                if json_match:
                    solver_json = json.loads(json_match.group())
                    for i in range(1, 4):
                        question_key = f"question{i}"
                        if question_key in solver_json:
                            solver_data[i] = {
                                "score": 0.0,  # 新格式中没有score字段
                                "pass": solver_json[question_key].get("pass", 0),
                                "rank": solver_json[question_key].get("rank", 0),
                                "suggestions": solver_json[question_key].get("suggestion", [])
                            }
            except Exception as e:
                logger.warning(f"解析solver JSON失败: {e}")
            
            # 提取educator数据
            educator_data = {}
            try:
                # 尝试从educator_content中提取JSON
                json_match = re.search(r'\{.*\}', educator_content, re.DOTALL)
                if json_match:
                    educator_json = json.loads(json_match.group())
                    for i in range(1, 4):
                        question_key = f"question{i}"
                        if question_key in educator_json:
                            educator_data[i] = {
                                "score": 0.0,  # 新格式中没有score字段
                                "pass": educator_json[question_key].get("pass", 0),
                                "rank": educator_json[question_key].get("rank", 0),
                                "suggestions": educator_json[question_key].get("suggestion", [])
                            }
            except Exception as e:
                logger.warning(f"解析educator JSON失败: {e}")
            
            # 合并数据
            for i in range(1, 4):  # 题目1, 2, 3
                question_data = {
                    "question_num": i,
                    "solver": solver_data.get(i, {"score": 0.0, "pass": 0, "rank": 0}),
                    "educator": educator_data.get(i, {"score": 0.0, "pass": 0, "rank": 0}),
                    "combined_rank": 0
                }
                
                # 计算综合排名（solver rank + educator rank）
                question_data["combined_rank"] = question_data["solver"]["rank"] + question_data["educator"]["rank"]
                
                result["questions"].append(question_data)
            
            # 选择最佳题目 - 修改优先级逻辑
            # 1. 最高优先级：两个PASS=1的题目，按rank总和排序
            both_pass_questions = [q for q in result["questions"] if q["solver"]["pass"] == 1 and q["educator"]["pass"] == 1]
            
            if both_pass_questions:
                # 选择rank总和最高的题目；若并列，则优先Educator的rank更高者
                best_question = max(
                    both_pass_questions,
                    key=lambda x: (x["combined_rank"], x["educator"].get("rank", 0))
                )
                result["selected_question"] = best_question
                result["selection_reason"] = (
                    f"两个PASS=1，优先选择rank总和最高；若并列，取Educator rank更高。"
                    f" 本次选择题目{best_question['question_num']} (综合rank={best_question['combined_rank']}, "
                    f"Educator rank={best_question['educator'].get('rank', 0)})"
                )
            else:
                # 2. 次优先级：至少一个PASS=1的题目，优先选择两个PASS都较高的
                # 计算每个题目的PASS总分（solver pass + educator pass）
                for q in result["questions"]:
                    q["pass_total"] = q["solver"]["pass"] + q["educator"]["pass"]
                
                # 按PASS总分降序，然后按rank总和降序排序
                best_question = max(result["questions"], key=lambda x: (x["pass_total"], x["combined_rank"]))
                result["selected_question"] = best_question
                
                if best_question["pass_total"] == 2:
                    result["selection_reason"] = f"两个PASS=1，选择rank总和最高({best_question['combined_rank']})的题目{best_question['question_num']}"
                elif best_question["pass_total"] == 1:
                    pass_source = "Solver" if best_question["solver"]["pass"] == 1 else "Educator"
                    result["selection_reason"] = f"仅{pass_source} PASS=1，选择rank总和最高({best_question['combined_rank']})的题目{best_question['question_num']}"
                else:
                    result["selection_reason"] = f"无PASS=1的题目，选择rank总和最高({best_question['combined_rank']})的题目{best_question['question_num']}"
            
            return result

        except Exception as e:
            logger.error(f"解析三条题目评分失败: {e}")
            # 返回默认值
            for i in range(1, 4):
                result["questions"].append({
                    "question_num": i,
                    "solver": {"score": 0.0, "pass": 0, "rank": 0},
                    "educator": {"score": 0.0, "pass": 0, "rank": 0},
                    "combined_rank": 0
                })
            result["selected_question"] = result["questions"][0]
            result["selection_reason"] = "解析失败，使用默认选择"
            return result

    def _extract_corrected_answer(self, solver_parsed: Dict[str, Any], educator_parsed: Dict[str, Any]) -> str:
        """从解析结果中提取被建议的正确答案（优先取Solver，其次Educator）。"""
        try:
            # 先从轻量解析器提供的字段中取
            corrected = self.lightweight_parser.extract_corrected_answer(solver_parsed)
            if corrected:
                return str(corrected).strip()
        except Exception:
            pass
        try:
            corrected = self.lightweight_parser.extract_corrected_answer(educator_parsed) if hasattr(self.lightweight_parser, 'extract_corrected_answer') else ""
            if corrected:
                return str(corrected).strip()
        except Exception:
            pass

        # 兜底：在原始文本里尝试抓取“将答案修改为X/正确答案为X”
        import re as _re_ec
        def _scan(raw: str) -> str:
            if not raw:
                return ""
            m = _re_ec.search(r"(将答案修改为|正确答案为|答案应为)[：: ]?([A-D]|[\u4e00-\u9fa5A-Za-z0-9\.\-\+]+)", str(raw))
            return m.group(2).strip() if m else ""

        if isinstance(solver_parsed, dict):
            raw = solver_parsed.get('raw_output') or str(solver_parsed)
            val = _scan(raw)
            if val:
                return val
        if isinstance(educator_parsed, dict):
            raw = educator_parsed.get('raw_output') or str(educator_parsed)
            val = _scan(raw)
            if val:
                return val
        return ""

    def _is_only_answer_correction(self, rewrite_parsed: Dict[str, Any], solver_parsed: Dict[str, Any], educator_parsed: Dict[str, Any]) -> bool:
        """判断是否仅为“答案更正”型修改（无题干/选项结构变化）。
        规则（启发式）：
        - 解析到的重写结果中 question 为空 且 options 为空；且存在 corrected_answer 建议。
        - 或重写输出仅包含答案字段，且题干与选项为空。
        """
        try:
            corrected = self._extract_corrected_answer(solver_parsed, educator_parsed)
            if not corrected:
                return False
            if not rewrite_parsed or 'raw_output' in rewrite_parsed:
                return True
            q = (rewrite_parsed.get('question') or '').strip()
            opts = rewrite_parsed.get('options') or []
            # 若没有新题干或选项，只是给出了答案/解析，视为仅答案更正
            if (not q) and (not opts or len(opts) == 0):
                return True
            return False
        except Exception:
            return False

    def _apply_answer_correction(self, final_question: Any, corrected_answer: str) -> Any:
        """将纠正的答案应用到最终题文本。
        支持：
        - 文本型题目：覆盖“答案为X”片段；若无显式答案，则在末尾追加。
        - 列表型（不期望在此阶段出现），则对第一个元素应用。
        """
        if not corrected_answer:
            return final_question

        # 若是列表，尽量对第一个应用，避免结构复杂化
        if isinstance(final_question, list) and final_question:
            return [self._apply_answer_correction(final_question[0], corrected_answer)] + list(final_question[1:])

        if not isinstance(final_question, str):
            return final_question

        import re as _re_ap
        text = final_question
        # 常见格式：“答案为X” 或 “答案为：X” 或 “答案：X”
        patterns = [
            r"(答案为[：: ]?)[A-Da-z0-9\.\-\+]+",
            r"(答案[：: ]?)[A-Da-z0-9\.\-\+]+",
        ]
        replaced = False
        for pat in patterns:
            new_text, n = _re_ap.subn(pat, r"\1" + corrected_answer, text)
            if n > 0:
                text = new_text
                replaced = True
                break
        if not replaced:
            # 若未检测到答案字段，则在末尾追加一行
            sep = "\n\n" if not text.endswith("\n") else "\n"
            text = f"{text}{sep}答案为{corrected_answer}"
        return text

    async def generate_single_question(self, input_data: Dict[str, Any], fast_mode: bool = True) -> Dict[str, Any]:
        """为单条数据生成题目（带迭代重写机制）"""
        try:
            logger.info(f"开始为题目ID {input_data['que_id']} 生成新题目")
            self.stats["total_questions"] += 1
            
            # 构建习题配置（将题型纳入配置）
            exercise_config = {
                "grade": input_data["grade"],
                "grade_level": "年级",
                "subject": "数学",
                "knowledge_point": input_data["knowledge"],
                "exercise_type": input_data.get("question_type") or input_data.get("type") or "选择题",
                "difficulty": input_data["difficulty"],
                "core_competency": ", ".join(input_data["competence"]) if input_data["competence"] else "基础能力"
            }
            
            # 保持原有workflow不变，只调整批处理
            max_rewrites = 6  # 重写次数：5次
            max_replans = 2   # 重新规划次数：2次
            score_threshold_s = 9.0  # 保持原有评分标准
            score_threshold_e = 8.0
            
            for replan_attempt in range(max_replans):
                logger.info(f"题目ID {input_data['que_id']} - 第{replan_attempt + 1}次规划")
                if replan_attempt > 0:  # 重新规划
                    self.stats["replan_attempts"] += 1
                    if replan_attempt == 1:  # 第一次重新规划
                        self.stats["questions_requiring_replan"] += 1
                
                # 初始化重写尝试次数
                rewrite_attempt = 0
                
                # 使用知识检索器获取相关知识（如果启用RAG）
                if self.use_rag and self.knowledge_retriever:
                    if self.rag_mode == "planner":
                        # Planner RAG模式：获取完整知识（知识图谱+课程标准+题目模式分析）
                        knowledge_context = self.knowledge_retriever.format_knowledge_for_prompt(
                            knowledge_graph_results, curriculum_results
                        )
                        logger.info(f"题目ID {input_data['que_id']} - 使用Planner RAG模式，上下文长度: {len(knowledge_context)}")
                    elif self.rag_mode == "planner_kg":
                        # Planner-KG模式：Planner仅用知识图谱+课程标准（不含样例）
                        knowledge_graph_results = self.knowledge_retriever.search_knowledge_graph(
                            knowledge_point=exercise_config['knowledge_point'],
                            grade=exercise_config['grade'],
                            grade_level=exercise_config['grade_level']
                        )
                        curriculum_results = self.knowledge_retriever.search_curriculum_standards(
                            knowledge_point=exercise_config['knowledge_point'],
                            grade=exercise_config['grade'],
                            difficulty=exercise_config['difficulty'],
                            core_competency=exercise_config['core_competency']
                        )
                        knowledge_context = self.knowledge_retriever.format_knowledge_for_prompt(
                            knowledge_graph_results, curriculum_results
                        )
                        logger.info(f"题目ID {input_data['que_id']} - 使用Planner-KG模式（无样例），上下文长度: {len(knowledge_context)}")
                    elif self.rag_mode == "writer":
                        # Writer RAG模式：Planner只获取知识图谱+课程标准，不包含题目样例
                        knowledge_graph_results = self.knowledge_retriever.search_knowledge_graph(
                            knowledge_point=exercise_config['knowledge_point'],
                            grade=exercise_config['grade'],
                            grade_level=exercise_config['grade_level']
                        )
                        curriculum_results = self.knowledge_retriever.search_curriculum_standards(
                            knowledge_point=exercise_config['knowledge_point'],
                            grade=exercise_config['grade'],
                            difficulty=exercise_config['difficulty'],
                            core_competency=exercise_config['core_competency']
                        )
                        knowledge_context = self.knowledge_retriever.format_knowledge_for_prompt(
                            knowledge_graph_results, curriculum_results
                        )
                        logger.info(f"题目ID {input_data['que_id']} - 使用Writer RAG模式，Planner获得知识图谱+课程标准，上下文长度: {len(knowledge_context)}")
                    elif self.rag_mode == "writer_only":
                        # Writer Only模式：跳过Planner，直接使用Writer
                        knowledge_context = ""
                        logger.info(f"题目ID {input_data['que_id']} - 使用Writer Only模式，跳过Planner")
                    else:
                        knowledge_context = ""
                        logger.info(f"题目ID {input_data['que_id']} - 未知RAG模式，跳过检索")
                else:
                    knowledge_context = ""
                    logger.info(f"题目ID {input_data['que_id']} - 跳过RAG，直接使用基础配置")
                
                # 如果是writer_only模式，跳过Planner阶段
                if self.rag_mode == "writer_only":
                    # 直接使用输入参数作为规划内容
                    planner_content_clean = f"知识点：{exercise_config['knowledge_point']}，难度：{exercise_config['difficulty']}，年级：{exercise_config['grade']}，素养：{exercise_config['core_competency']}，题型：{exercise_config['exercise_type']}"
                    logger.info(f"题目ID {input_data['que_id']} - Writer Only模式，跳过Planner，直接使用输入参数")
                else:
                    if replan_attempt > 0:  # 重新规划
                        # 重新规划时使用带教师反馈的prompt，禁用RAG
                        # 构建重写历史作为教师分析
                        teacher_analysis = ""
                        if 'planner_content_clean' in locals() and planner_content_clean:
                            teacher_analysis += f"## 之前的规划内容\n{planner_content_clean}\n\n"
                        if rewrite_history:
                            teacher_analysis += "## 重写历史\n"
                            for i, attempt in enumerate(rewrite_history, 1):
                                teacher_analysis += f"第{i}次重写：{attempt.get('question', '')}\n"
                            teacher_analysis += "\n"
                        
                        planner_prompt = build_planner_user_prompt_with_teacher_feedback(
                            exercise_config=exercise_config,
                            knowledge_context="",  # 重新规划时禁用RAG
                            teacher_analysis=teacher_analysis,
                            use_rag=False  # 重新规划时禁用RAG
                        )
                    else:
                        # 首次规划使用普通prompt
                        planner_prompt = build_planner_user_prompt(
                            exercise_config,
                            knowledge_context,
                            use_rag=self.use_rag
                        )
                    
                    # Planner 阶段重试机制
                    planner_response = await self._retry_agent_call(
                        agent_name="planner",
                        task=planner_prompt,
                        stage_name="Planner规划",
                        input_data=input_data,
                        replan_attempt=replan_attempt,
                        rewrite_attempt=rewrite_attempt
                    )
                    
                    # 提取规划内容
                    if hasattr(planner_response, 'messages') and planner_response.messages:
                        last_message = planner_response.messages[-1]
                        planner_content = last_message.content if hasattr(last_message, 'content') else str(last_message)
                    elif hasattr(planner_response, 'content'):
                        planner_content = planner_response.content
                    else:
                        planner_content = str(planner_response)
                    
                    # 提取token使用量
                    planner_token_usage = self._extract_token_usage(planner_response, stage="planner")
                    
                    # 保存planner workflow
                    self._save_workflow_step(
                        input_data['que_id'], 
                        f"planner_attempt_{replan_attempt + 1}", 
                        planner_prompt, 
                        planner_content,
                        planner_token_usage
                    )
                    
                    # 从内容中提取题目规划部分
                    if "### 题目规划：" in planner_content:
                        start_idx = planner_content.find("### 题目规划：")
                        end_idx = planner_content.find("###", start_idx + 1) if "###" in planner_content[start_idx + 1:] else len(planner_content)
                        planner_content = planner_content[start_idx:end_idx].strip()
                    elif "题目规划：" in planner_content:
                        start_idx = planner_content.find("题目规划：")
                        end_idx = planner_content.find("\n\n", start_idx) if "\n\n" in planner_content[start_idx:] else len(planner_content)
                        planner_content = planner_content[start_idx:end_idx].strip()
                    
                    # 仅保留题干与选项，移除解释等
                    planner_content_clean = self._keep_stem_and_options(planner_content)
                
                # 初始化重写历史和RAG内容
                if 'rewrite_history' not in locals():
                    rewrite_history = []
                # 在第一次写题时获取RAG内容，重写时复用
                initial_rag_content = ""
                evaluator_rag_content = ""
                if self.use_rag and self.rag_mode in ["writer", "writer_only"] and self.knowledge_retriever:
                    try:
                        # 为Writer获取RAG内容（在binary模式下不会使用）
                        initial_rag_content = self.knowledge_retriever.retrieve_knowledge_for_writer(exercise_config)
                        # 为Solver/Educator获取三种难度的参考示例
                        evaluator_rag_content = self.knowledge_retriever.retrieve_knowledge_for_evaluators(exercise_config)
                        logger.info(f"题目ID {input_data['que_id']} - 获取Writer RAG内容，长度: {len(initial_rag_content)}")
                        logger.info(f"题目ID {input_data['que_id']} - 获取评分参考示例，长度: {len(evaluator_rag_content)}")
                    except Exception as e:
                        logger.warning(f"获取RAG内容失败: {e}")
                
                for rewrite_attempt in range(max_rewrites):
                    if rewrite_attempt == 0:
                        logger.info(f"题目ID {input_data['que_id']} - 第{replan_attempt + 1}次规划，第1次写题")
                    else:
                        logger.info(f"题目ID {input_data['que_id']} - 第{replan_attempt + 1}次规划，第{rewrite_attempt + 1}次重写")
                        self.stats["rewrite_attempts"] += 1
                        if rewrite_attempt == 1:  # 第一次重写
                            self.stats["questions_requiring_rewrite"] += 1
                    
                    # 构建writer prompt
                    if rewrite_attempt == 0:
                        # 第一次写题：与后续重写共用同一份 initial_rag_content
                        writer_prompt = build_writer_user_prompt_for_first(
                            planner_content_clean,
                            input_data.get("question_type", "选择题"),
                            use_writer_rag=False,
                            knowledge_retriever=None,
                            exercise_config=exercise_config,
                            rag_content=("" if (self.evaluation_mode == "binary" and self.rag_mode == "writer") else initial_rag_content),
                            rag_mode=self.rag_mode
                        )
                    else:
                        # 重写阶段，基于选中的题目和修改建议进行重写
                        # 获取选中题目的修改建议
                        selected_question_num = selected_question["question_num"]
                        solver_feedback = f"题目{selected_question_num}的Solver评分: {solver_score}, PASS: {solver_pass}, RANK: {selected_question['solver']['rank']}"
                        educator_feedback = f"题目{selected_question_num}的Educator评分: {educator_score}, PASS: {educator_pass}, RANK: {selected_question['educator']['rank']}"
                        
                        # 从三条题目数据中提取选中题目的修改建议
                        solver_suggestions = ""
                        educator_suggestions = ""
                        selected_question_data = {}
                        # 从已解析的数据中获取修改建议（使用索引边界检查）
                        try:
                            questions_list = three_questions_data.get("questions", [])
                            idx = int(selected_question_num) - 1
                            if isinstance(questions_list, list) and 0 <= idx < len(questions_list):
                                selected_question_data = questions_list[idx] or {}
                                if "solver" in selected_question_data and "suggestions" in selected_question_data["solver"]:
                                    solver_suggestions = "; ".join(selected_question_data["solver"].get("suggestions", []))
                                if "educator" in selected_question_data and "suggestions" in selected_question_data["educator"]:
                                    educator_suggestions = "; ".join(selected_question_data["educator"].get("suggestions", []))
                        except Exception:
                            selected_question_data = {}
                        
                        # 在binary模式下，使用被选中的单个题目进行重写
                        if self.evaluation_mode == "binary" and isinstance(three_questions_for_rewrite, list):
                            selected_question_num = selected_question["question_num"]
                            if selected_question_num <= len(three_questions_for_rewrite):
                                previous_question = three_questions_for_rewrite[selected_question_num - 1]
                                logger.info(f"题目ID {input_data['que_id']} - 重写使用选中的题目{selected_question_num}")
                            else:
                                previous_question = three_questions_for_rewrite[0]
                                logger.warning(f"题目ID {input_data['que_id']} - 选中题目索引超出范围，使用第一个题目")
                        
                        # 构建重写prompt - 只传递pass、rank和修改建议
                        solver_pass = 0
                        solver_rank = 0
                        educator_pass = 0
                        educator_rank = 0
                        if selected_question_data:
                            if isinstance(selected_question_data.get("solver"), dict):
                                solver_pass = selected_question_data["solver"].get("pass", 0)
                                solver_rank = selected_question_data["solver"].get("rank", 0)
                            if isinstance(selected_question_data.get("educator"), dict):
                                educator_pass = selected_question_data["educator"].get("pass", 0)
                                educator_rank = selected_question_data["educator"].get("rank", 0)
                        
                        solver_feedback_clean = f"PASS: {solver_pass}, RANK: {solver_rank}\n修改建议: {solver_suggestions}"
                        educator_feedback_clean = f"PASS: {educator_pass}, RANK: {educator_rank}\n修改建议: {educator_suggestions}"
                        
                        writer_prompt = build_writer_user_prompt_for_rewrite(
                            previous_question,
                            solver_score,
                            solver_feedback_clean,
                            educator_score,
                            educator_feedback_clean,
                            question_type=input_data.get("question_type", "选择题"),
                            use_writer_rag=False,  # 重写时不重新RAG
                            knowledge_retriever=None,
                            exercise_config=exercise_config,
                            rag_content=("" if (self.evaluation_mode == "binary" and self.rag_mode == "writer") else initial_rag_content),
                            rag_mode=self.rag_mode
                        )
                    
                    # Writer 阶段重试机制
                    writer_response = await self._retry_agent_call(
                        agent_name="writer",
                        task=writer_prompt,
                        stage_name="Writer编写",
                        input_data=input_data,
                        replan_attempt=replan_attempt,
                        rewrite_attempt=rewrite_attempt
                    )
                    
                    # 提取编写的题目内容
                    if hasattr(writer_response, 'messages') and writer_response.messages:
                        last_message = writer_response.messages[-1]
                        question_content = last_message.content if hasattr(last_message, 'content') else str(last_message)
                    elif hasattr(writer_response, 'content'):
                        question_content = writer_response.content
                    else:
                        question_content = str(writer_response)
                    
                    # 提取token使用量
                    writer_token_usage = self._extract_token_usage(writer_response, stage="writer")
                    
                    # 解析Writer输出 - 支持三条题目
                    if self.evaluation_mode == "binary":
                        # Binary模式：解析三条题目的JSON数组
                        try:
                            import json
                            # 尝试解析JSON数组
                            json_match = re.search(r'\[.*\]', question_content, re.DOTALL)
                            if json_match:
                                three_questions = json.loads(json_match.group())
                                if isinstance(three_questions, list) and len(three_questions) >= 3:
                                    # 成功解析三条题目
                                    final_question = three_questions  # 保存为数组
                                    logger.info(f"题目ID {input_data['que_id']} - 成功解析三条题目")
                                else:
                                    # 解析失败，使用第一个题目
                                    final_question = three_questions[0] if three_questions else question_content
                                    logger.warning(f"题目ID {input_data['que_id']} - 解析的题目数量不足，使用第一个题目")
                            else:
                                # 没有找到JSON数组，使用原有逻辑
                                final_question = self._keep_stem_and_options(question_content)
                                logger.warning(f"题目ID {input_data['que_id']} - 未找到JSON数组，使用原有解析逻辑")
                        except Exception as e:
                            logger.error(f"题目ID {input_data['que_id']} - 解析三条题目失败: {e}")
                            final_question = self._keep_stem_and_options(question_content)
                    else:
                        # Score模式：使用原有解析逻辑
                        if self.use_lightweight_parsing:
                            if rewrite_attempt == 0:
                                # 首次写题使用writer类型解析
                                writer_parsed = self.lightweight_parser.parse_agent_output('writer', question_content)
                                final_question = self.lightweight_parser.extract_question_content(writer_parsed)
                            else:
                                # 重写时使用rewrite类型解析
                                writer_parsed = self.lightweight_parser.parse_agent_output('rewrite', question_content)
                                final_question = self.lightweight_parser.extract_rewrite_question_content(writer_parsed)
                        else:
                            # 使用原有解析逻辑
                            final_question = self._keep_stem_and_options(question_content)
                    
                    # 保存writer workflow（包含完整的solution信息）
                    self._save_workflow_step(
                        input_data['que_id'], 
                        f"writer_attempt_{replan_attempt + 1}_{rewrite_attempt + 1}", 
                        writer_prompt, 
                        question_content,
                        writer_token_usage
                    )
                    
                    # 实时输出题目：将当前题目条目 upsert 到 generated_questions_realtime.json
                    try:
                        # 统一结构：que_id、input、question（字符串或数组）、meta
                        entry = {
                            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                            "que_id": input_data['que_id'],
                            "input": {
                                "grade": exercise_config["grade"],
                                "difficulty": exercise_config["difficulty"],
                                "competence": input_data.get("competence", []),
                                "knowledge": exercise_config["knowledge_point"],
                                "question_type": exercise_config.get("exercise_type", "选择题")
                            },
                            "question": final_question,
                            "stage": f"writer_attempt_{replan_attempt + 1}_{rewrite_attempt + 1}"
                        }
                        self._upsert_generated_question(input_data['que_id'], entry)
                    except Exception as _e_rt:
                        logger.warning(f"实时输出题目失败（忽略继续）：{_e_rt}")
                    
                    # 在binary模式下，previous_question应该是被选中的单个题目
                    if self.evaluation_mode == "binary" and isinstance(final_question, list):
                        # 暂时保存三条题目，稍后在重写时使用选中的题目
                        three_questions_for_rewrite = final_question
                        previous_question = None  # 稍后设置
                    else:
                        previous_question = final_question
                    
                    # 执行Solver评分阶段（根据 evaluation_mode 选择对应 prompt 文件）
                    if self.evaluation_mode == "binary":
                        # Binary模式：传递三条题目给Solver
                        if isinstance(final_question, list):
                            # 三条题目的情况
                            solver_prompt = build_solver_user_prompt_binary(final_question, "", exercise_config)
                        else:
                            # 单个题目的情况（重写时）
                            solver_prompt = build_solver_user_prompt_binary(final_question, "", exercise_config)
                    else:
                        solver_prompt = build_solver_user_prompt(final_question, "", exercise_config)
                    # Solver 阶段重试机制
                    solver_response = await self._retry_agent_call(
                        agent_name="solver",
                        task=solver_prompt,
                        stage_name="Solver评分",
                        input_data=input_data,
                        replan_attempt=replan_attempt,
                        rewrite_attempt=rewrite_attempt
                    )
                    
                    # 提取solver内容
                    if hasattr(solver_response, 'messages') and solver_response.messages:
                        last_message = solver_response.messages[-1]
                        solver_content = last_message.content if hasattr(last_message, 'content') else str(last_message)
                    elif hasattr(solver_response, 'content'):
                        solver_content = solver_response.content
                    else:
                        solver_content = str(solver_response)
                    
                    # 提取token使用量
                    solver_token_usage = self._extract_token_usage(solver_response, stage="solver")
                    
                    # 保存solver workflow
                    self._save_workflow_step(
                        input_data['que_id'], 
                        f"solver_attempt_{replan_attempt + 1}_{rewrite_attempt + 1}", 
                        solver_prompt, 
                        solver_content,
                        solver_token_usage
                    )
                    
                    # 使用轻量级解析器处理Solver输出
                    if self.use_lightweight_parsing:
                        solver_parsed = self.lightweight_parser.parse_agent_output('solver', solver_content)
                        correct_answer = self.lightweight_parser.extract_corrected_answer(solver_parsed)
                        
                        if correct_answer:
                            logger.info(f"题目ID {input_data['que_id']} - 从Solver反馈中提取到正确答案：{correct_answer}")
                            final_question = self._replace_answer_in_question(final_question, correct_answer)
                            logger.info(f"题目ID {input_data['que_id']} - 已将正确答案替换到题目中")
                        else:
                            logger.info(f"题目ID {input_data['que_id']} - 未从Solver反馈中提取到正确答案，使用原题目")
                    else:
                        # 使用原有解析逻辑
                        correct_answer = self._extract_correct_answer_from_solver(solver_content)
                        if correct_answer:
                            logger.info(f"题目ID {input_data['que_id']} - 从Solver反馈中提取到正确答案：{correct_answer}")
                            final_question = self._replace_answer_in_question(final_question, correct_answer)
                            logger.info(f"题目ID {input_data['que_id']} - 已将正确答案替换到题目中")
                        else:
                            logger.info(f"题目ID {input_data['que_id']} - 未从Solver反馈中提取到正确答案，使用原题目")
                    
                    # 执行Educator评分阶段（根据 evaluation_mode 选择对应 prompt 文件）
                    if self.evaluation_mode == "binary":
                        # Binary模式：传递三条题目给Educator
                        ref_examples = evaluator_rag_content if (self.rag_mode == "writer") else ""
                        if isinstance(final_question, list):
                            # 三条题目的情况
                            educator_prompt = build_educator_user_prompt_binary(final_question, "", exercise_config, reference_examples=ref_examples)
                        else:
                            # 单个题目的情况（重写时）
                            educator_prompt = build_educator_user_prompt_binary(final_question, "", exercise_config, reference_examples=ref_examples)
                    else:
                        educator_prompt = build_educator_user_prompt(final_question, "", exercise_config)
                    # Educator 阶段重试机制
                    educator_response = await self._retry_agent_call(
                        agent_name="educator",
                        task=educator_prompt,
                        stage_name="Educator评分",
                        input_data=input_data,
                        replan_attempt=replan_attempt,
                        rewrite_attempt=rewrite_attempt
                    )
                    
                    # 提取educator内容
                    if hasattr(educator_response, 'messages') and educator_response.messages:
                        last_message = educator_response.messages[-1]
                        educator_content = last_message.content if hasattr(last_message, 'content') else str(last_message)
                    elif hasattr(educator_response, 'content'):
                        educator_content = educator_response.content
                    else:
                        educator_content = str(educator_response)
                    
                    # 提取token使用量
                    educator_token_usage = self._extract_token_usage(educator_response, stage="educator")
                    
                    # 保存educator workflow
                    self._save_workflow_step(
                        input_data['que_id'], 
                        f"educator_attempt_{replan_attempt + 1}_{rewrite_attempt + 1}", 
                        educator_prompt, 
                        educator_content,
                        educator_token_usage
                    )
                    
                    # 提取三条题目的评分和排名信息
                    three_questions_data = self._extract_three_questions_scores(solver_content, educator_content)
                    
                    # 获取选中的题目信息
                    selected_question = three_questions_data["selected_question"]
                    solver_score = selected_question["solver"]["score"]
                    educator_score = selected_question["educator"]["score"]
                    solver_pass = selected_question["solver"]["pass"]
                    educator_pass = selected_question["educator"]["pass"]
                    
                    logger.info(f"题目ID {input_data['que_id']} - 三条题目评分解析完成")
                    logger.info(f"题目ID {input_data['que_id']} - 选中题目{selected_question['question_num']}: Solver评分={solver_score}, Educator评分={educator_score}")
                    logger.info(f"题目ID {input_data['que_id']} - 选择原因: {three_questions_data['selection_reason']}")
                    
                    # 保存三条题目的完整评分信息到workflow
                    self._save_workflow_step(
                        input_data['que_id'], 
                        f"three_questions_analysis_{replan_attempt + 1}_{rewrite_attempt + 1}", 
                        "三条题目评分分析", 
                        json.dumps(three_questions_data, ensure_ascii=False, indent=2),
                        {"input_tokens": 0, "output_tokens": 0, "total_cost": 0.0}
                    )
                    
                    # 检查是否满足条件：
                    # - binary：Solver/Educator 都需 PASS=1
                    # - score：沿用阈值（score_threshold_s / score_threshold_e）
                    if (self.evaluation_mode == "binary" and solver_pass == 1 and educator_pass == 1) or \
                       (self.evaluation_mode != "binary" and solver_score >= score_threshold_s and educator_score > score_threshold_e):
                        # 二值模式也执行教师质检（恢复checker并增强：先解题再检查）
                        try:
                            # 在binary模式下，使用被选中的单个题目进行Teacher检查
                            if self.evaluation_mode == "binary" and isinstance(final_question, list):
                                # 从选中的题目信息中获取题目内容
                                selected_question_num = selected_question["question_num"]
                                if selected_question_num <= len(final_question):
                                    selected_question_content = final_question[selected_question_num - 1]
                                    teacher_check_prompt = build_teacher_checker_prompt(selected_question_content, exercise_config)
                                    logger.info(f"题目ID {input_data['que_id']} - Teacher检查使用选中的题目{selected_question_num}")
                                else:
                                    # 如果索引超出范围，使用第一个题目
                                    teacher_check_prompt = build_teacher_checker_prompt(final_question[0], exercise_config)
                                    logger.warning(f"题目ID {input_data['que_id']} - 选中题目索引超出范围，使用第一个题目")
                            else:
                                # 非binary模式或单个题目，使用原有逻辑
                                teacher_check_prompt = build_teacher_checker_prompt(final_question, exercise_config)
                            
                            # Teacher 阶段重试机制
                            teacher_check_resp = await self._retry_agent_call(
                                agent_name="teacher",
                                task=teacher_check_prompt,
                                stage_name="Teacher质检",
                                input_data=input_data,
                                replan_attempt=replan_attempt,
                                rewrite_attempt=rewrite_attempt
                            )
                            # 提取文本
                            if hasattr(teacher_check_resp, 'messages') and teacher_check_resp.messages:
                                last_msg = teacher_check_resp.messages[-1]
                                teacher_check_content = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)
                            elif hasattr(teacher_check_resp, 'content'):
                                teacher_check_content = teacher_check_resp.content
                            else:
                                teacher_check_content = str(teacher_check_resp)
                            # 保存workflow
                            teacher_tokens = self._extract_token_usage(teacher_check_resp, stage="teacher_check")
                            self._save_workflow_step(
                                input_data['que_id'],
                                "teacher_check",
                                teacher_check_prompt,
                                teacher_check_content,
                                teacher_tokens
                            )
                            # 解析CHECK（正则避免误判），并把解析结果写入内容便于排查
                            import re as _re
                            m = _re.search(r"CHECK\s*=\s*(0|1)", teacher_check_content, flags=_re.IGNORECASE)
                            check_val = int(m.group(1)) if m else 1
                            # 解析是否仅答案错误与纠正答案
                            m_only = _re.search(r"ONLY_ANSWER_ERROR\s*=\s*(0|1)", teacher_check_content, flags=_re.IGNORECASE)
                            only_ans_err = int(m_only.group(1)) if m_only else 0
                            m_corr = _re.search(r"CORRECTED_ANSWER\s*=\s*(.+)", teacher_check_content, flags=_re.IGNORECASE)
                            corrected_from_teacher = m_corr.group(1).strip() if m_corr else ""
                            teacher_check_content = f"PARSED_CHECK={check_val}\nONLY_ANSWER_ERROR={only_ans_err}\nCORRECTED_ANSWER={corrected_from_teacher}\n" + str(teacher_check_content)
                            if check_val == 0:
                                # 在binary模式下，重写时使用被选中的单个题目
                                rewrite_question = final_question
                                if self.evaluation_mode == "binary" and isinstance(final_question, list):
                                    selected_question_num = selected_question["question_num"]
                                    if selected_question_num <= len(final_question):
                                        rewrite_question = final_question[selected_question_num - 1]
                                        logger.info(f"题目ID {input_data['que_id']} - 重写使用选中的题目{selected_question_num}")
                                # 若仅答案错误，则直接应用教师给出的更正答案并跳过重写
                                if only_ans_err == 1 and corrected_from_teacher:
                                    try:
                                        final_question = self._apply_answer_correction(rewrite_question, corrected_from_teacher)
                                        logger.info(f"题目ID {input_data['que_id']} - 仅答案错误，已直接替换为: {corrected_from_teacher}")
                                        # 直接跳过重写流程，继续后续保存
                                        pass
                                    except Exception:
                                        logger.warning(f"题目ID {input_data['que_id']} - 应用更正答案失败，改走轻量重写")
                                else:
                                    # 触发一次轻量重写：将问题点作为student_feedback传给Writer
                                    logger.info(f"题目ID {input_data['que_id']} - 教师质检未通过，触发一次轻量重写")
                                # 提取Teacher修改建议：仅保留“修改建议”部分，以减少噪声
                                try:
                                    import re as _re_s
                                    # teacher_check_content 前已加了 PARSED_CHECK= 标记，这里只抓取修改建议块
                                    sugg_match = _re_s.search(r"修改建议[:：][\s\S]*$", teacher_check_content)
                                    if sugg_match:
                                        teacher_suggestions_only = sugg_match.group(0).strip()
                                    else:
                                        teacher_suggestions_only = teacher_check_content.strip()
                                    # 采用checker_only通道提示
                                    teacher_student_feedback = f"###检查结果\nCHECK=0\n{teacher_suggestions_only}"
                                except Exception:
                                    teacher_student_feedback = teacher_check_content

                                light_rewrite_prompt = build_writer_user_prompt_for_rewrite(
                                    previous_exercise=rewrite_question,
                                    solver_score=solver_score,
                                    solver_feedback=solver_content,
                                    educator_score=educator_score,
                                    educator_feedback=educator_content,
                                    student_feedback=teacher_student_feedback,
                                    question_type=input_data.get("question_type", "选择题"),
                                    use_writer_rag=False,
                                    knowledge_retriever=None,
                                    exercise_config=exercise_config,
                                    rag_content="",
                                    rag_mode=self.rag_mode
                                )
                                # Writer 轻量级重试机制
                                light_rewrite_resp = await self._retry_agent_call(
                                    agent_name="writer",
                                    task=light_rewrite_prompt,
                                    stage_name="Writer轻量重写",
                                    input_data=input_data,
                                    replan_attempt=replan_attempt,
                                    rewrite_attempt=rewrite_attempt
                                )
                                if hasattr(light_rewrite_resp, 'messages') and light_rewrite_resp.messages:
                                    last_msg = light_rewrite_resp.messages[-1]
                                    question_content = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)
                                elif hasattr(light_rewrite_resp, 'content'):
                                    question_content = light_rewrite_resp.content
                                else:
                                    question_content = str(light_rewrite_resp)
                                
                                # 使用轻量级解析器处理重写输出
                                if self.use_lightweight_parsing:
                                    rewrite_parsed = self.lightweight_parser.parse_agent_output('rewrite', question_content)
                                    
                                    # 检查是否只是答案错误，题目内容没有改变
                                    if self._is_only_answer_correction(rewrite_parsed, solver_parsed, {}):
                                        # 如果只是答案错误，直接使用建议中的正确答案
                                        corrected_answer = self._extract_corrected_answer(solver_parsed, {})
                                        if corrected_answer:
                                            final_question = self._apply_answer_correction(final_question, corrected_answer)
                                        else:
                                            final_question = self.lightweight_parser.extract_rewrite_question_content(rewrite_parsed)
                                    else:
                                        final_question = self.lightweight_parser.extract_rewrite_question_content(rewrite_parsed)
                                else:
                                    final_question = self._keep_stem_and_options(question_content)
                                # 保存workflow
                                light_tokens = self._extract_token_usage(light_rewrite_resp, stage="writer")
                                self._save_workflow_step(
                                    input_data['que_id'],
                                    "writer_light_rewrite_after_teacher_check",
                                    light_rewrite_prompt,
                                    question_content,  # 保存原始输出
                                    light_tokens
                                )
                            # 质检后（或轻量重写后）的最终题目，实时落盘
                            try:
                                final_entry = {
                                    "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                                    "que_id": input_data['que_id'],
                                    "input": {
                                        "grade": exercise_config["grade"],
                                        "difficulty": exercise_config["difficulty"],
                                        "competence": input_data.get("competence", []),
                                        "knowledge": exercise_config["knowledge_point"],
                                        "question_type": exercise_config.get("exercise_type", "选择题")
                                    },
                                    "question": final_question,
                                    "stage": "teacher_passed"
                                }
                                self._upsert_generated_question(input_data['que_id'], final_entry)
                            except Exception as _e_rt_final:
                                logger.warning(f"实时输出最终题目失败（忽略继续）：{_e_rt_final}")
                        except Exception as e:
                            logger.error(f"Teacher质检阶段失败: {e}")
                            # 写入占位记录，保证realtime可见
                            self._save_workflow_step(
                                input_data['que_id'],
                                "teacher_check",
                                "Teacher质检异常",
                                f"错误: {str(e)}\nPARSED_CHECK=1",
                                {"input_tokens": 0, "output_tokens": 0, "total_cost": 0.0}
                            )
                        except Exception as _:
                            # 质检异常不阻断主流程
                            pass

                        logger.info(f"题目ID {input_data['que_id']} - 评分达标，输出题目")
                        self.stats["successful_questions"] += 1

                        # 计算该题目的总花费
                        question_token_stats = self.calculate_token_usage_per_question()
                        que_id = input_data["que_id"]
                        total_cost = question_token_stats.get(que_id, {}).get('total_cost', 0.0)
                        total_tokens = question_token_stats.get(que_id, {}).get('total_tokens', 0)
                        total_prompt_tokens = question_token_stats.get(que_id, {}).get('total_prompt_tokens', 0)
                        total_completion_tokens = question_token_stats.get(que_id, {}).get('total_completion_tokens', 0)

                        # 在二选一模式下，确保写入的题目为被选中的单题
                        if self.evaluation_mode == "binary" and isinstance(final_question, list):
                            try:
                                _sel_num = selected_question.get("question_num", 1)
                                if 1 <= _sel_num <= len(final_question):
                                    final_question = final_question[_sel_num - 1]
                                else:
                                    final_question = final_question[0]
                            except Exception:
                                final_question = final_question[0]

                        result = {
                            "que_id": input_data["que_id"],
                            "input": {
                                "grade": input_data["grade"],
                                "difficulty": input_data["difficulty"],
                                "competence": input_data["competence"],
                                "knowledge": input_data["knowledge"],
                                "question_type": input_data.get("question_type")
                            },
                            "question": final_question,
                            "solver_score": solver_content,
                            "educator_score": educator_content,
                            "solver_score_numeric": solver_score,
                            "educator_score_numeric": educator_score,
                            "rewrite_attempts": rewrite_attempt + 1,
                            "replan_attempts": replan_attempt + 1,
                            "status": "success",
                            "cost_info": {
                                "total_cost": total_cost,
                                "total_tokens": total_tokens,
                                "prompt_tokens": total_prompt_tokens,
                                "completion_tokens": total_completion_tokens
                            }
                        }
                        return result
                    
                    # 不满足条件，记录重写历史并准备重写反馈
                    # 在二选一模式下，记录重写历史时也确保是被选中的单题
                    _question_for_history = final_question
                    if self.evaluation_mode == "binary" and isinstance(final_question, list):
                        try:
                            _sel_num_h = selected_question.get("question_num", 1)
                            if 1 <= _sel_num_h <= len(final_question):
                                _question_for_history = final_question[_sel_num_h - 1]
                            else:
                                _question_for_history = final_question[0]
                        except Exception:
                            _question_for_history = final_question[0]

                    rewrite_attempt_data = {
                        "question": _question_for_history,
                        "solver_score": solver_content,
                        "educator_score": educator_content,
                        "solver_score_numeric": solver_score,
                        "educator_score_numeric": educator_score,
                        "attempt_number": rewrite_attempt + 1,
                        "planner_content": planner_content_clean
                    }
                    rewrite_history.append(rewrite_attempt_data)
                    
                    # 提取关键反馈信息用于重写
                    if self.use_lightweight_parsing:
                        solver_parsed = self.lightweight_parser.parse_agent_output('solver', solver_content)
                        educator_parsed = self.lightweight_parser.parse_agent_output('educator', educator_content)
                        
                        solver_feedback = self.lightweight_parser.extract_solver_feedback(solver_parsed)
                        educator_feedback = self.lightweight_parser.extract_educator_feedback(educator_parsed)
                    else:
                        from prompts import _extract_key_feedback
                        solver_feedback = _extract_key_feedback(solver_content)
                        educator_feedback = _extract_key_feedback(educator_content)
                    
                    if rewrite_attempt == max_rewrites - 1:
                        logger.warning(f"题目ID {input_data['que_id']} - 重写{max_rewrites}次仍未达标，准备重新规划")
                        break
                
                # 如果重写次数用完仍未达标，进行下一轮规划
                if replan_attempt == max_replans - 1:
                    logger.error(f"题目ID {input_data['que_id']} - 规划{max_replans}次仍未达标，跳过此题")
                    break
            
            # 所有尝试都失败
            self.stats["failed_questions"] += 1
            
            # 计算该题目的总花费（即使失败也有花费）
            question_token_stats = self.calculate_token_usage_per_question()
            que_id = input_data["que_id"]
            total_cost = question_token_stats.get(que_id, {}).get('total_cost', 0.0)
            total_tokens = question_token_stats.get(que_id, {}).get('total_tokens', 0)
            total_prompt_tokens = question_token_stats.get(que_id, {}).get('total_prompt_tokens', 0)
            total_completion_tokens = question_token_stats.get(que_id, {}).get('total_completion_tokens', 0)
            
            
            result = {
                "que_id": input_data["que_id"],
                "input": {
                    "grade": input_data["grade"],
                    "difficulty": input_data["difficulty"],
                    "competence": input_data["competence"],
                    "knowledge": input_data["knowledge"],
                    "question_type": input_data.get("question_type")
                },
                "question": f"生成失败：经过{max_replans}次规划和{max_rewrites}次重写仍未达标",
                "solver_score": "生成失败",
                "educator_score": "生成失败",
                "solver_score_numeric": 0.0,
                "educator_score_numeric": 0.0,
                "rewrite_attempts": max_rewrites,
                "replan_attempts": max_replans,
                "cost_info": {
                    "total_cost": total_cost,
                    "total_tokens": total_tokens,
                    "prompt_tokens": total_prompt_tokens,
                    "completion_tokens": total_completion_tokens
                }
            }
            
            logger.error(f"题目ID {input_data['que_id']} - 最终生成失败")
            return result
            
        except Exception as e:
            error_str = str(e)
            logger.error(f"为题目ID {input_data['que_id']} 生成题目失败: {e}")
            
            # 检查是否是 API Key 相关错误
            if any(keyword in error_str.lower() for keyword in ["api key", "authentication", "unauthorized", "forbidden"]):
                logger.warning(f"检测到 API Key 认证问题，尝试重新初始化客户端并重试...")
                
                # 重新初始化客户端
                try:
                    self.agent_manager.model_manager._initialize_clients()
                    logger.info("客户端重新初始化成功，准备重试...")
                    
                    # 重试生成题目（递归调用，但限制重试次数）
                    if not hasattr(self, '_api_retry_count'):
                        self._api_retry_count = 0
                    
                    if self._api_retry_count < 3:  # 最多重试3次
                        self._api_retry_count += 1
                        logger.info(f"第 {self._api_retry_count} 次 API Key 重试...")
                        
                        # 递归调用，但重置重试计数
                        original_retry_count = self._api_retry_count
                        result = await self.generate_single_question(input_data, fast_mode)
                        self._api_retry_count = original_retry_count  # 恢复原始计数
                        return result
                    else:
                        logger.error("API Key 重试次数已达上限，放弃重试")
                        self._api_retry_count = 0  # 重置计数
                        
                except Exception as retry_error:
                    logger.error(f"重试过程中发生错误: {retry_error}")
                    self._api_retry_count = 0  # 重置计数
            
            self.stats["failed_questions"] += 1
            
            # 保存错误workflow
            self._save_workflow_step(
                input_data['que_id'], 
                "error", 
                f"生成失败: {str(e)}", 
                f"错误详情: {str(e)}"
            )
            
            # 计算该题目的总花费（即使异常也有花费）
            question_token_stats = self.calculate_token_usage_per_question()
            que_id = input_data["que_id"]
            total_cost = question_token_stats.get(que_id, {}).get('total_cost', 0.0)
            total_tokens = question_token_stats.get(que_id, {}).get('total_tokens', 0)
            total_prompt_tokens = question_token_stats.get(que_id, {}).get('total_prompt_tokens', 0)
            total_completion_tokens = question_token_stats.get(que_id, {}).get('total_completion_tokens', 0)
            
            
            return {
                "que_id": input_data["que_id"],
                "input": {
                    "grade": input_data["grade"],
                    "difficulty": input_data["difficulty"],
                    "competence": input_data["competence"],
                    "knowledge": input_data["knowledge"],
                    "question_type": input_data.get("question_type")
                },
                "question": f"生成失败: {str(e)}",
                "solver_score": "生成失败",
                "educator_score": "生成失败",
                "solver_score_numeric": 0.0,
                "educator_score_numeric": 0.0,
                "rewrite_attempts": 0,
                "replan_attempts": 0,
                "cost_info": {
                    "total_cost": total_cost,
                    "total_tokens": total_tokens,
                    "prompt_tokens": total_prompt_tokens,
                    "completion_tokens": total_completion_tokens
                }
            }
    
    async def generate_batch_questions(self, questions_data: List[Dict[str, Any]], 
                                     batch_size: int = 5, 
                                     delay_between_batches: float = 2.0,
                                     output_file: str = None) -> List[Dict[str, Any]]:
        """批量生成题目"""
        logger.info(f"开始批量生成题目，总共 {len(questions_data)} 条数据")
        
        generated_questions = []
        total_batches = (len(questions_data) + batch_size - 1) // batch_size
        consecutive_failures = 0  # 连续失败计数
        max_consecutive_failures = 5  # 最大连续失败次数
        
        # 创建批次进度条
        batch_pbar = tqdm(
            total=total_batches,
            desc="批次处理",
            unit="批次",
            position=0,
            leave=True
        )
        
        # 分批处理，避免过载
        try:
            failed_records: List[Dict[str, Any]] = []
            for i in range(0, len(questions_data), batch_size):
                batch = questions_data[i:i + batch_size]
                batch_num = i // batch_size + 1
                
                # 更新批次进度条描述
                batch_pbar.set_description(f"批次 {batch_num}/{total_batches}")
                batch_pbar.set_postfix({
                    "当前批次题目数": len(batch),
                    "已完成题目数": len(generated_questions)
                })
                
                # 并行处理当前批次
                tasks = []
                for question_data in batch:
                    input_data = self.extract_input_fields(question_data)
                    # 教育目标完整性校验：任一缺失则跳过
                    if not self._is_education_goal_complete(input_data):
                        logger.warning(
                            f"题目ID {input_data.get('que_id','')} - 教育目标不完整，跳过。"
                            f"grade={input_data.get('grade')}, difficulty={input_data.get('difficulty')}, "
                            f"knowledge={input_data.get('knowledge')}, competence={input_data.get('competence')}"
                        )
                        # 记录到已生成列表以便追踪
                        generated_questions.append({
                            "que_id": input_data.get("que_id", ""),
                            "index": i,  # 跳过时记录其全局索引（批首）
                            "input": {
                                "grade": input_data.get("grade", ""),
                                "difficulty": input_data.get("difficulty", ""),
                                "competence": input_data.get("competence", []),
                                "knowledge": input_data.get("knowledge", ""),
                                "question_type": input_data.get("question_type")
                            },
                            "status": "skipped",
                            "reason": "教育目标不完整（年级/难度/知识点/核心素养任一缺失）"
                        })
                        continue
                    task = self.generate_single_question(input_data)
                    tasks.append(task)
                
                # 等待当前批次完成
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # 处理结果
                batch_success_count = 0
                for j, result in enumerate(batch_results):
                    global_index = i + j
                    if isinstance(result, Exception):
                        logger.error(f"批次 {batch_num} 中第 {j+1} 条数据生成失败: {result}")
                        # 添加失败记录
                        que_id = batch[j].get("que_id", f"unknown_{i+j}")
                        input_data = self.extract_input_fields(batch[j])
                        
                        # 计算该题目的总花费（即使异常也有花费）
                        question_token_stats = self.calculate_token_usage_per_question()
                        total_cost = question_token_stats.get(que_id, {}).get('total_cost', 0.0)
                        total_tokens = question_token_stats.get(que_id, {}).get('total_tokens', 0)
                        total_prompt_tokens = question_token_stats.get(que_id, {}).get('total_prompt_tokens', 0)
                        total_completion_tokens = question_token_stats.get(que_id, {}).get('total_completion_tokens', 0)
                        
                        
                        failed_question = {
                            "que_id": que_id,
                            "index": global_index,
                            "input": {
                                "grade": input_data["grade"],
                                "difficulty": input_data["difficulty"],
                                "competence": input_data["competence"],
                                "knowledge": input_data["knowledge"],
                                "question_type": input_data.get("question_type")
                            },
                            "question": f"生成失败: {str(result)}",
                            "cost_info": {
                                "total_cost": total_cost,
                                "total_tokens": total_tokens,
                                "prompt_tokens": total_prompt_tokens,
                                "completion_tokens": total_completion_tokens
                            }
                        }
                        generated_questions.append(failed_question)
                        # 记录到失败清单（精简）
                        failed_records.append({
                            "que_id": que_id,
                            "index": global_index,
                            "error": str(result)
                        })
                    else:
                        # 为成功结果也注入索引，便于回溯
                        try:
                            if isinstance(result, dict):
                                result.setdefault("index", global_index)
                                result.setdefault("que_id", batch[j].get("que_id", ""))
                        except Exception:
                            pass
                        generated_questions.append(result)
                        batch_success_count += 1
                
                # 检查连续失败情况
                if batch_success_count == 0:
                    consecutive_failures += 1
                    logger.warning(f"批次 {batch_num} 全部失败，连续失败次数: {consecutive_failures}")
                    
                    if consecutive_failures >= max_consecutive_failures:
                        logger.error(f"连续 {max_consecutive_failures} 个批次全部失败，程序中断")
                        batch_pbar.close()
                        # 保存已生成的结果
                        if output_file:
                            self._save_generated_questions(generated_questions, output_file)
                        return generated_questions
                else:
                    consecutive_failures = 0  # 重置连续失败计数
                
                # 实时保存已生成的结果
                if output_file:
                    self._save_generated_questions(generated_questions, output_file)
                
                # 更新批次进度条
                batch_pbar.update(1)
                batch_pbar.set_postfix({
                    "当前批次题目数": len(batch),
                    "已完成题目数": len(generated_questions),
                    "连续失败": consecutive_failures
                })
                
                # 批次间延迟，避免过载
                if i + batch_size < len(questions_data):
                    logger.info(f"等待 {delay_between_batches} 秒后处理下一批...")
                    await asyncio.sleep(delay_between_batches)
        
        except KeyboardInterrupt:
            logger.info("程序被用户中断")
            batch_pbar.close()
            # 保存已生成的结果
            if output_file:
                self._save_generated_questions(generated_questions, output_file)
            return generated_questions
        except Exception as e:
            logger.error(f"批处理过程中发生错误: {e}")
            batch_pbar.close()
            # 保存已生成的结果
            if output_file:
                self._save_generated_questions(generated_questions, output_file)
            return generated_questions
        
        # 关闭批次进度条
        batch_pbar.close()

        # 如有失败题目，单独保存失败清单到与输出同目录
        try:
            if failed_records and output_file:
                failed_dir = os.path.dirname(output_file)
                failed_name = f"failed_{os.path.splitext(os.path.basename(output_file))[0]}.json"
                failed_path = os.path.join(failed_dir, failed_name)
                with open(failed_path, "w", encoding="utf-8") as f:
                    json.dump(failed_records, f, ensure_ascii=False, indent=2)
                logger.info(f"失败题目清单已保存到: {failed_path}")
        except Exception as _e:
            logger.warning(f"保存失败清单出错: {_e}")
        
        self.generated_questions = generated_questions
        logger.info(f"批量生成完成，成功生成 {len(generated_questions)} 条题目")
        return generated_questions
    
    def print_statistics(self):
        """打印统计信息"""
        logger.info("=" * 80)
        logger.info("📊 题目生成统计报告")
        logger.info("=" * 80)
        logger.info(f"📝 总题目数: {self.stats['total_questions']}")
        logger.info(f"✅ 成功生成: {self.stats['successful_questions']}")
        logger.info(f"❌ 生成失败: {self.stats['failed_questions']}")
        logger.info(f"🔄 需要重写的题目: {self.stats['questions_requiring_rewrite']}")
        logger.info(f"🔄 需要重新规划的题目: {self.stats['questions_requiring_replan']}")
        logger.info(f"📈 总重写次数: {self.stats['rewrite_attempts']}")
        logger.info(f"📈 总重新规划次数: {self.stats['replan_attempts']}")
        
        if self.stats['total_questions'] > 0:
            success_rate = (self.stats['successful_questions'] / self.stats['total_questions']) * 100
            rewrite_rate = (self.stats['questions_requiring_rewrite'] / self.stats['total_questions']) * 100
            replan_rate = (self.stats['questions_requiring_replan'] / self.stats['total_questions']) * 100
            logger.info(f"📊 成功率: {success_rate:.1f}%")
            logger.info(f"📊 重写率: {rewrite_rate:.1f}%")
            logger.info(f"📊 重新规划率: {replan_rate:.1f}%")
        
        logger.info("=" * 80)
    
    def calculate_token_usage_per_question(self) -> Dict[str, Dict[str, Any]]:
        """计算每条题目的token使用量和成本"""
        token_stats = {}
        
        for workflow_step in self.workflow_data:
            que_id = workflow_step.get('que_id', 'unknown')
            stage = workflow_step.get('stage', 'unknown')
            token_usage = workflow_step.get('token_usage', {})
            
            if que_id not in token_stats:
                token_stats[que_id] = {
                    'total_prompt_tokens': 0,
                    'total_completion_tokens': 0,
                    'total_tokens': 0,
                    'total_cost': 0.0,
                    'stages': {}
                }
            
            # 累加token使用量和成本
            prompt_tokens = token_usage.get('prompt_tokens', 0)
            completion_tokens = token_usage.get('completion_tokens', 0)
            total_tokens = token_usage.get('total_tokens', 0)
            total_cost = token_usage.get('total_cost', 0.0)
            
            token_stats[que_id]['total_prompt_tokens'] += prompt_tokens
            token_stats[que_id]['total_completion_tokens'] += completion_tokens
            token_stats[que_id]['total_tokens'] += total_tokens
            token_stats[que_id]['total_cost'] += total_cost
            
            # 记录每个阶段的token使用量和成本
            token_stats[que_id]['stages'][stage] = {
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'total_tokens': total_tokens,
                'total_cost': total_cost
            }
        
        return token_stats
    
    def print_token_usage_summary(self):
        """打印token使用量和成本摘要"""
        token_stats = self.calculate_token_usage_per_question()
        
        if not token_stats:
            logger.info("📊 没有找到token使用量数据")
            return
        
        logger.info("=" * 80)
        logger.info("📊 Token使用量和成本统计报告")
        logger.info("=" * 80)
        
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tokens = 0
        total_cost = 0.0
        
        for que_id, stats in token_stats.items():
            total_prompt_tokens += stats['total_prompt_tokens']
            total_completion_tokens += stats['total_completion_tokens']
            total_tokens += stats['total_tokens']
            total_cost += stats['total_cost']
        
        logger.info(f"📝 总题目数: {len(token_stats)}")
        logger.info(f"🔤 总Prompt Tokens: {total_prompt_tokens:,}")
        logger.info(f"📝 总Completion Tokens: {total_completion_tokens:,}")
        logger.info(f"📊 总Tokens: {total_tokens:,}")
        logger.info(f"💰 总成本: ${total_cost:.4f}")
        logger.info(f"📈 平均每题Tokens: {total_tokens/len(token_stats):.0f}")
        logger.info(f"💰 平均每题成本: ${total_cost/len(token_stats):.4f}")
        logger.info("=" * 80)
        
        # 显示前5题的详细统计
        logger.info("📋 前5题详细Token使用量和成本:")
        for i, (que_id, stats) in enumerate(list(token_stats.items())[:5]):
            logger.info(f"题目 {que_id}:")
            logger.info(f"  📊 总Tokens: {stats['total_tokens']:,}")
            logger.info(f"  🔤 Prompt: {stats['total_prompt_tokens']:,}")
            logger.info(f"  📝 Completion: {stats['total_completion_tokens']:,}")
            logger.info(f"  💰 总成本: ${stats['total_cost']:.4f}")
            
            # 显示各阶段使用量和成本
            for stage, stage_stats in stats['stages'].items():
                if stage_stats['total_tokens'] > 0:
                    logger.info(f"    {stage}: {stage_stats['total_tokens']:,} tokens, ${stage_stats['total_cost']:.4f}")
            logger.info("")
    
    def save_token_usage_summary(self, output_filename: str = None, start_index: int = None, end_index: int = None) -> str:
        """保存token使用量和成本统计报告到文件"""
        token_stats = self.calculate_token_usage_per_question()
        
        if not token_stats:
            logger.warning("📊 没有找到token使用量数据，无法保存统计报告")
            return None
        
        if not output_filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if start_index is not None and end_index is not None:
                output_filename = f"token_usage_summary_{timestamp}_{start_index}-{end_index}.json"
            else:
                output_filename = f"token_usage_summary_{timestamp}.json"
        
        output_path = os.path.join(self.output_dir, output_filename)
        
        try:
            # 计算总体统计
            total_prompt_tokens = 0
            total_completion_tokens = 0
            total_tokens = 0
            total_cost = 0.0
            
            for que_id, stats in token_stats.items():
                total_prompt_tokens += stats['total_prompt_tokens']
                total_completion_tokens += stats['total_completion_tokens']
                total_tokens += stats['total_tokens']
                total_cost += stats['total_cost']
            
            # 构建统计报告数据
            summary_data = {
                "generation_time": datetime.now().isoformat(),
                "summary": {
                    "total_questions": len(token_stats),
                    "total_prompt_tokens": total_prompt_tokens,
                    "total_completion_tokens": total_completion_tokens,
                    "total_tokens": total_tokens,
                    "total_cost": round(total_cost, 4),
                    "average_tokens_per_question": round(total_tokens / len(token_stats), 0),
                    "average_cost_per_question": round(total_cost / len(token_stats), 4)
                },
                "detailed_stats": token_stats
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"📊 Token使用量统计报告已保存到: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"保存token使用量统计报告失败: {e}")
            return None
    
    def save_generated_questions(self, output_filename: str = None) -> str:
        """保存生成的题目到JSON文件"""
        if not output_filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"generated_questions_{timestamp}.json"
        
        output_path = os.path.join(self.output_dir, output_filename)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.generated_questions, f, ensure_ascii=False, indent=2)
            
            logger.info(f"生成的题目已保存到: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"保存生成的题目失败: {e}")
            raise
    
    def save_workflow(self, output_filename: str = None) -> str:
        """保存workflow数据到JSON文件"""
        if not output_filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"workflow_{timestamp}.json"
        
        output_path = os.path.join(self.output_dir, output_filename)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.workflow_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Workflow数据已保存到: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"保存workflow数据失败: {e}")
            raise
    
    async def _retry_agent_call(self, agent_name: str, task: str, stage_name: str, 
                               input_data: Dict[str, Any], replan_attempt: int, 
                               rewrite_attempt: int, max_retries: int = 10, request_timeout: float = 1200.0) -> Any:
        """通用的智能体调用重试机制"""
        retry_count = 0
        base_delay = 1.0   # 固定重试间隔1秒
        max_delay = 1.0    # 最大延迟也设为1秒
        
        while retry_count < max_retries:
            try:
                # 记录 prompt 长度用于调试
                prompt_length = len(task) if isinstance(task, str) else len(str(task))
                logger.info(f"题目ID {input_data['que_id']} - {stage_name}阶段 prompt 长度: {prompt_length} 字符")
                
                # 记录重试次数
                if retry_count > 0:
                    logger.warning(f"题目ID {input_data['que_id']} - {stage_name}阶段第 {retry_count} 次重试，prompt 长度: {prompt_length} 字符")
                
                # 记录 prompt 前100字符用于调试
                prompt_preview = (task[:100] + "...") if len(task) > 100 else task
                logger.debug(f"题目ID {input_data['que_id']} - {stage_name}阶段 prompt 预览: {prompt_preview}")
                
                # 检查 prompt 是否过长（超过5000字符可能是异常）
                if prompt_length > 5000:
                    logger.warning(f"题目ID {input_data['que_id']} - {stage_name}阶段 prompt 过长: {prompt_length} 字符，可能存在重复内容")
                    
                    # 检查是否有明显的重复内容
                    if isinstance(task, str):
                        # 检查是否有重复的段落
                        lines = task.split('\n')
                        unique_lines = set(lines)
                        if len(unique_lines) < len(lines) * 0.8:  # 如果重复率超过20%
                            logger.error(f"题目ID {input_data['que_id']} - {stage_name}阶段检测到大量重复内容！")
                            logger.error(f"总行数: {len(lines)}, 唯一行数: {len(unique_lines)}")
                        
                        # 截断过长的 prompt 以避免异常 token 消耗
                        if len(task) > 5000:
                            task = task[:5000] + "\n\n[提示：内容过长，已截断]"
                            logger.warning(f"已截断 prompt 到 5000 字符")
                
                # 每次调用前重建该智能体，确保无历史上下文
                try:
                    self.agent_manager.recreate_agent(agent_name)
                except Exception as _e_recreate_once:
                    logger.warning(f"重建智能体 {agent_name} 失败（忽略继续）：{_e_recreate_once}")

                # 尝试调用智能体
                response = await asyncio.wait_for(
                    self.agent_manager.get_agent(agent_name).run(task=task),
                    timeout=request_timeout
                )
                return response
                
            except Exception as e:
                error_str = str(e)
                
                lower_err = error_str.lower()
                # 需要重试的瞬时错误集合
                transient_keywords = self._get_transient_error_keywords()

                # 检查是否是 API Key 相关错误
                if any(keyword in lower_err for keyword in ["api key", "authentication", "unauthorized", "forbidden"]):
                    retry_count += 1
                    logger.warning(f"题目ID {input_data['que_id']} - {stage_name}阶段 API Key 错误，第 {retry_count} 次重试...")
                    
                    # 重新初始化客户端
                    try:
                        self.agent_manager.model_manager._initialize_clients()
                        logger.info(f"客户端重新初始化成功，继续 {stage_name} 阶段重试...")
                    except Exception as init_error:
                        logger.error(f"重新初始化客户端失败: {init_error}")
                        
                    # 如果还有重试次数，继续循环
                    if retry_count < max_retries:
                        # 计算延迟并等待
                        delay = self._calculate_retry_delay(retry_count, base_delay, max_delay, lower_err)
                        await asyncio.sleep(delay)
                        continue
                    else:
                        logger.error(f"{stage_name}阶段 API Key 重试次数已达上限 ({max_retries} 次)，放弃重试")
                        raise e
                # 检查是否是瞬时错误（包含超时、限流、5xx网关类）
                elif any(keyword in lower_err for keyword in transient_keywords):
                    retry_count += 1
                    logger.warning(f"题目ID {input_data['que_id']} - {stage_name}阶段瞬时错误（可能超时/限流/网关），第 {retry_count} 次重试: {error_str}")
                    # 计算延迟并等待
                    delay = self._calculate_retry_delay(retry_count, base_delay, max_delay, lower_err)
                    logger.warning(f"{stage_name}阶段将等待 {delay:.1f}s 后重试（错误：{error_str}）")
                    await asyncio.sleep(delay)
                    continue
                else:
                    # 非 API Key 错误，直接抛出
                    raise e
        
        # 如果所有重试都失败了
        raise Exception(f"{stage_name}阶段重试失败，已达到最大重试次数 {max_retries}")

    def _keep_stem_and_options(self, text: str) -> str:
        """仅保留题干与选项：
        - 去除以"解析：/解释：/教学目的：/学生解答：/答案：/参考答案："开头的段落及其后的内容
        - 保留题干与选项行（以"选项：""A.""B.""C.""D."等）
        - 去除多余标题如"###题目："
        - 去除RAG参考信息（题目参考样例等）
        - 去除"选择方向"部分，只保留题目部分
        """
        try:
            if not isinstance(text, str):
                return str(text)
            lines = text.splitlines()
            kept = []
            stop_prefixes = ("解析：", "解释：", "教学目的：", "学生解答：", "答案：", "参考答案：")
            stop_re = re.compile(r"^(解析|解释|教学目的|学生解答|答案|参考答案)[：:]")
            rag_stop_re = re.compile(r"^(=== 题目参考样例 ===|以下为.*等题的真实样例|题目参考样例|参考样例)")
            direction_re = re.compile(r"^### 选择方向：")
            in_direction_section = False
            
            for line in lines:
                s = line.strip()
                if not s:
                    if not in_direction_section:  # 只在非方向选择部分保留空行
                        kept.append(line)
                    continue
                
                # 检测到选择方向部分，跳过
                if direction_re.match(s):
                    in_direction_section = True
                    continue
                
                # 检测到题目部分开始，结束方向选择部分
                if s.startswith("### 题目：") and in_direction_section:
                    in_direction_section = False
                    s = s.replace("### 题目：", "").strip()
                    if s:  # 如果题目内容不为空
                        kept.append(s)
                    continue
                
                # 如果在方向选择部分，跳过
                if in_direction_section:
                    continue
                
                # 遇到RAG参考信息则停止后续保留
                if rag_stop_re.match(s):
                    break
                    
                # 遇到解释/解析等则停止后续保留
                if stop_re.match(s):
                    break
                    
                # 去掉标题
                if s.startswith("###题目："):
                    s = s.replace("###题目：", "").strip()
                kept.append(s)
            result = "\n".join(kept).strip()
            return result
        except Exception:
            return text if isinstance(text, str) else str(text)
    
    def _extract_correct_answer_from_solver(self, solver_content: str) -> str:
        """从Solver反馈中提取正确答案"""
        if not solver_content:
            return ""
        
        import re
        lines = solver_content.split('\n')
        in_suggestions_section = False
        
        for line in lines:
            line = line.strip()
            
            # 检测是否进入修改建议部分
            if '修改建议：' in line or '建议：' in line:
                in_suggestions_section = True
                continue
            
            # 如果在修改建议部分，查找答案
            if in_suggestions_section:
                # 匹配各种答案格式
                patterns = [
                    r'将答案更正为[：:]\s*([A-D])',  # 将答案更正为：A
                    r'将答案更正为[：:]\s*([^，,；;。\n]+)',  # 将答案更正为：具体答案
                    r'答案[为是]?[：:]\s*([A-D])',  # 答案：A
                    r'答案[为是]?[：:]\s*([^，,；;。\n]+)',  # 答案：具体答案
                    r'正确答案[为是]?[：:]\s*([A-D])',  # 正确答案：A
                    r'正确答案[为是]?[：:]\s*([^，,；;。\n]+)',  # 正确答案：具体答案
                    r'应为[：:]\s*([A-D])',  # 应为：A
                    r'应为[：:]\s*([^，,；;。\n]+)',  # 应为：具体答案
                    r'应该[是]?[：:]\s*([A-D])',  # 应该是：A
                    r'应该[是]?[：:]\s*([^，,；;。\n]+)',  # 应该是：具体答案
                    r'([A-D])\s*[选项]',  # A选项
                    r'选择\s*([A-D])',  # 选择A
                ]
                
                for pattern in patterns:
                    match = re.search(pattern, line)
                    if match:
                        return match.group(1)
        
        return ""
    
    def _get_transient_error_keywords(self) -> List[str]:
        """获取瞬时错误关键词列表"""
        return [
            # 超时/限流/网关
            "timeout", "timed out", "read timeout", "write timeout", "gateway timeout",
            "rate limit", "too many requests", "429", "try again", "temporarily unavailable",
            "service unavailable", "unavailable", "bad gateway", "502", "503", "504",
            # 连接类错误
            "connection error", "connect error", "connection reset", "connection aborted",
            "network error", "network is unreachable", "connection refused",
            # DNS/解析类错误
            "dns", "resolve", "name resolution", "name or service not known",
            "temporary failure in name resolution",
            # SSL/TLS短暂错误
            "ssl error", "tlsv1", "certificate verify failed",
            # 通用服务端错误
            "server error"
        ]

    def _log_attempt(self, que_id: str, replan_attempt: int, rewrite_attempt: int = None, stage: str = None):
        """统一的尝试日志输出"""
        if rewrite_attempt is None:
            logger.info(f"题目ID {que_id} - 第{replan_attempt + 1}次规划")
        elif rewrite_attempt == 0:
            logger.info(f"题目ID {que_id} - 第{replan_attempt + 1}次规划，第1次写题")
        else:
            logger.info(f"题目ID {que_id} - 第{replan_attempt + 1}次规划，第{rewrite_attempt + 1}次重写")

    def _calculate_retry_delay(self, retry_count: int, base_delay: float, max_delay: float, error_str: str) -> float:
        """计算重试延迟时间 - 固定1秒间隔"""
        # 所有错误类型都使用固定1秒延迟
        return 1.0

    def _replace_answer_in_question(self, question_content: str, correct_answer: str) -> str:
        """替换题目中的答案"""
        if not correct_answer or not question_content:
            return question_content
        
        lines = question_content.split('\n')
        result = []
        
        for line in lines:
            if line.strip().startswith('答案为：') or line.strip().startswith('答案：'):
                # 替换答案行
                result.append(f"答案为：{correct_answer}")
            else:
                result.append(line)
        
        return '\n'.join(result)


# ==================== 主系统类 ====================

class MultiAgentSystemV3:
    """多智能体系统V3主类"""
    
    def __init__(self, use_rag: bool = True, rag_mode: str = "planner", rag_config: dict = None, evaluation_mode: str = "score"):
        self.model_manager = None
        self.agent_manager = None
        self.batch_generator = None
        self.use_rag = use_rag  # RAG开关
        self.rag_mode = rag_mode  # RAG模式：planner 或 writer
        self.evaluation_mode = evaluation_mode
        self.rag_config = rag_config  # RAG配置
        
        # 生成程序开始时间戳
        self.start_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 设置输出目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.output_dir = os.path.join(current_dir, "outputs")
        self._ensure_output_dir()
    
    def _ensure_output_dir(self):
        """确保输出目录存在"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    async def initialize(self):
        """初始化系统"""
        try:
            logger.info("正在初始化多智能体系统V3...")
            
            # 验证配置
            if not validate_config():
                raise ValueError("配置验证失败")
            
            # 初始化模型管理器
            self.model_manager = ModelClientManager()
            
            # 初始化智能体管理器
            self.agent_manager = AgentManager(self.model_manager)
            
            # 初始化批量生成器
            self.batch_generator = BatchQuestionGenerator(self.agent_manager, use_rag=self.use_rag, rag_mode=self.rag_mode, evaluation_mode=self.evaluation_mode, start_timestamp=self.start_timestamp)
            
            logger.info("多智能体系统V3初始化完成！")
            
        except Exception as e:
            logger.error(f"系统初始化失败: {e}")
            raise
    
    async def generate_questions_from_data(self, jsonl_file_path: str, 
                                         batch_size: int = 20,  # 大幅增加批处理大小
                                         delay_between_batches: float = 0.5,  # 减少延迟
                                         limit: int = None,
                                         start_index: int = 0,
                                         end_index: int = None,
                                         indices: List[int] = None) -> Dict[str, str]:
        """从数据文件生成题目，并立即进行测评，返回生成与测评结果文件路径"""
        import time
        start_time = time.time()
        
        try:
            logger.info(f"🚀 开始生成题目，批处理大小: {batch_size}")
            # 加载数据
            questions_data = self.batch_generator.load_questions_data(
                jsonl_file_path, 
                limit=limit, 
                start_index=start_index, 
                end_index=end_index,
                indices=indices
            )
            
            # 生成输出文件路径（仅一份最终文件）
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            mode_tag = (self.rag_mode or "nomode").lower()
            range_tag = ""
            try:
                if start_index is not None and end_index is not None:
                    range_tag = f"_{start_index}-{end_index}"
            except Exception:
                range_tag = ""
            final_output_path = os.path.join(self.output_dir, f"generated_questions_{timestamp}_{mode_tag}{range_tag}.json")
            
            # 批量生成题目（实时保存到最终文件）
            generated_questions = await self.batch_generator.generate_batch_questions(
                questions_data, batch_size, delay_between_batches, final_output_path
            )
            
            # 打印统计信息
            self.batch_generator.print_statistics()
            
            # 打印token使用量统计
            self.batch_generator.print_token_usage_summary()
            
            # 保存token使用量统计报告
            self.batch_generator.save_token_usage_summary(start_index=start_index, end_index=end_index)
            
            # 已在生成时写入最终文件，这里不再重复保存
            output_path = final_output_path
            
            # 保存workflow（使用与结果相同的时间戳命名）
            workflow_path = self.batch_generator.save_workflow(
                output_filename=f"workflow_{timestamp}_{mode_tag}{range_tag}.json"
            )

            # 关闭评估：V3只生成题目，不做测评
            eval_output_path = ""
            
            # 计算总执行时间
            end_time = time.time()
            total_time = end_time - start_time
            hours = int(total_time // 3600)
            minutes = int((total_time % 3600) // 60)
            seconds = int(total_time % 60)
            
            logger.info("=" * 80)
            logger.info(f"⏱️  总执行时间: {hours:02d}:{minutes:02d}:{seconds:02d}")
            logger.info(f"📊 平均每题时间: {total_time/len(questions_data):.2f}秒")
            logger.info(f"🚀 批处理大小: {batch_size}")
            logger.info("=" * 80)
            
            return {"generated": output_path, "workflow": workflow_path, "evaluation": eval_output_path}
            
        except Exception as e:
            logger.error(f"生成题目失败: {e}")
            raise
    
    async def shutdown(self):
        """关闭系统"""
        try:
            if self.model_manager:
                await self.model_manager.close_all()
            logger.info("系统已关闭")
        except Exception as e:
            logger.error(f"系统关闭时出错: {e}")


# ==================== 主函数 ====================

async def main(use_rag: bool = True, rag_mode: str = "planner", evaluation_mode: str = "score"):
    """主函数"""
    logger.info("🎓 多智能体教育系统V3 - 批量题目生成")
    logger.info("="*80)
    if use_rag:
        if rag_mode == "planner":
            logger.info("RAG模式: Planner RAG模式（完整知识检索）")
        elif rag_mode == "planner_kg":
            logger.info("RAG模式: Planner-KG 模式（Planner仅用知识图谱+课标，无样例）")
        elif rag_mode == "writer":
            logger.info("RAG模式: Writer RAG模式（Writer获取题目样例）")
        elif rag_mode == "writer_only":
            logger.info("RAG模式: Writer Only模式（跳过Planner，Writer获取题目样例）")
        else:
            logger.info(f"RAG模式: {rag_mode}（未知模式）")
    else:
        logger.info("RAG模式: 禁用")
    
    # 评估模式输出
    if evaluation_mode == "binary":
        logger.info("评估模式: 二值评估（PASS=1才通过）")
    else:
        logger.info("评估模式: 数值评分（默认）")
    
    # 创建系统（可控制RAG开关和模式）
    system = MultiAgentSystemV3(use_rag=use_rag, rag_mode=rag_mode, evaluation_mode=evaluation_mode)
    
    try:
        # 初始化系统
        await system.initialize()
        
        # 设置数据文件路径
        jsonl_file_path = r"D:\CODE\three_0921\data\choice_unquie_500.jsonl"
        
        # 检查文件是否存在
        if not os.path.exists(jsonl_file_path):
            logger.error(f"数据文件不存在: {jsonl_file_path}")
            return
        
        logger.info(f"开始从 {jsonl_file_path} 生成题目...")
        
        # 生成题目（使用特定索引：0,1,3,5）
        paths = await system.generate_questions_from_data(
            jsonl_file_path=jsonl_file_path,
            batch_size=1,  # 每批处理1条，避免过载
            delay_between_batches=0.2,  # 批次间延迟0.2秒
            indices=None,  # 选择第0,1,3,5条题目
            start_index=450,
            end_index=490,
            limit=None
        )
        
        logger.info("="*80)
        logger.info("🎉 批量题目生成完成！")
        logger.info(f"📁 生成的题目已保存到: {paths['generated']}")
        logger.info(f"📋 Workflow数据已保存到: {paths['workflow']}")
        logger.info(f"📊 测评结果已保存到: {paths['evaluation']}")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"系统运行错误: {e}")
        
    finally:
        # 关闭系统
        await system.shutdown()


# ==================== 程序入口 ====================

if __name__ == "__main__":
    try:
        # 可以通过修改这里的参数来控制模式：
        # RAG模式：
        #   - use_rag=True, rag_mode="planner": Planner RAG模式（默认）
        #   - use_rag=True, rag_mode="writer": Writer RAG模式
        #   - use_rag=True, rag_mode="writer_only": Writer Only模式（跳过Planner）
        #   - use_rag=True, rag_mode="planner_kg": Planner-KG模式（Planner仅用知识图谱+课标，无样例）
        #   - use_rag=False: 禁用RAG，直接使用基础配置
        # 评估模式：
        #   - evaluation_mode="score": 数值评分（Solver/Educator输出分数）
        #   - evaluation_mode="binary": 二值评估（各维度0/1，需PASS=1）
        asyncio.run(main(use_rag=True, rag_mode="writer", evaluation_mode="binary"))
    except KeyboardInterrupt:
        logger.info("程序被用户中断")
    except Exception as e:
        logger.error(f"程序异常退出: {e}")
