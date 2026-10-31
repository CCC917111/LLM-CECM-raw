# llm_market_sim/agents/llm_agent.py
import logging
import json
import re
from typing import Dict, List, Any, Optional, Tuple

from .base_agent import BaseAgent
from .belief_module import BeliefModule
from .memory_module import MemoryModule
from .adf_module import AdaptiveDecisionFrequency
from llm_interface.llm_client import LLMClient
from llm_interface.prompts import format_decision_prompt

class LLMAgent(BaseAgent):
    """
    使用大型语言模型 (LLM) 进行决策的智能体。
    集成了信念更新和经验学习功能。
    """
    def __init__(self, agent_id: str, config: Dict, llm_client: LLMClient, competitor_ids: List[str],
                 initial_belief_params: Optional[Dict] = None, memory_capacity: int = 100,
                 role_description: Optional[str] = None, adf_config: Optional[Dict] = None, 
                 llm_config: Optional[Dict] = None):
        """
        初始化LLM智能体。

        Args:
            agent_id: 智能体ID。
            config: 智能体配置 (成本、容量等)。
            llm_client: 用于与LLM交互的客户端实例。
            competitor_ids: 其他竞争对手的ID列表。
            initial_belief_params: 传递给BeliefModule的初始信念参数。
            memory_capacity: 传递给MemoryModule的记忆容量。
            role_description: 角色描述。
            adf_config: ADF模块配置参数。
            llm_config: LLM配置参数，包含enable_adf开关。
        """
        super().__init__(agent_id, config)
        self.llm_client = llm_client
        self.belief_module = BeliefModule(agent_id, competitor_ids, initial_belief_params)
        self.memory_module = MemoryModule(agent_id, memory_capacity)
        
        # 检查是否启用ADF机制
        self.enable_adf = llm_config.get('enable_adf', True) if llm_config else True
        
        # 检查是否启用IBR-CR机制（迭代信念修正与反事实推理）
        self.enable_ibr_cr = llm_config.get('enable_ibr_cr', True) if llm_config else True
        
        if self.enable_adf:
            self.adf_module = AdaptiveDecisionFrequency(agent_id, adf_config)  # 初始化ADF模块
            logging.info(f"LLM智能体 {agent_id} 初始化完成（启用ADF模块）。")
        else:
            self.adf_module = None
            logging.info(f"LLM智能体 {agent_id} 初始化完成（禁用ADF模块）。")
            
        # 记录IBR-CR状态
        ibr_status = "启用" if self.enable_ibr_cr else "禁用"
        logging.info(f"LLM智能体 {agent_id} IBR-CR机制: {ibr_status}")
            
        self.last_bid = None # 存储上一轮成功生成的报价
        self.last_profit = None  # 存储上一轮利润，用于ADF判断
        self.role_description = role_description or "一家普通的发电企业" # 角色描述，如果未提供则使用默认值
        logging.info(f"  私有信息: 能源成本={self.private_info['marginal_cost_energy']}, 备用成本={self.private_info['marginal_cost_reserve']}, 容量={self.private_info['max_capacity']}")
        # 记录上一轮出清量（用于爬坡约束）
        self.last_cleared_energy = 0.0
        self.last_cleared_reserve = 0.0

    def decide_bid(self, round_number: int, market_info: Dict) -> Dict:
        """
        使用LLM生成报价决策。

        Args:
            round_number: 当前轮次。
            market_info: 市场公开信息。

        Returns:
            报价字典 {'energy_bid': {...}, 'reserve_bid': {...}}。
            如果LLM调用失败，可能返回默认报价或上一次的报价。
        """
        logging.info(f"智能体 {self.agent_id} 开始决策 (轮次 {round_number})...")

        # 1. 收集决策所需信息
        belief_summary = self.belief_module.get_belief_summary()
        # 触发经验总结的生成/更新
        experience_summary = self.memory_module.get_experience_summary(round_number)
        current_belief_state = self.belief_module.get_estimated_competitor_costs() # 获取当前信念数值

        # 2. 格式化Prompt
        prompt = format_decision_prompt(
            round_number=round_number,
            agent_id=self.agent_id,
            private_info=self.private_info,
            market_info=market_info,
            belief_summary=belief_summary,
            experience_summary=experience_summary,
            role_description=self.role_description
        )
        logging.debug(f"智能体 {self.agent_id} 生成的Prompt (部分): \n{prompt[:500]}...") # 记录部分Prompt用于调试

        # 3. 调用LLM（追踪token使用）
        llm_response = None
        token_info = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
        try:
            llm_response, token_info = self.llm_client.chat(prompt, return_json=True, track_tokens=True)
            logging.debug(f"智能体 {self.agent_id} 收到LLM响应: {llm_response}")
            logging.info(f"智能体 {self.agent_id} Token使用: {token_info['total_tokens']} tokens (prompt: {token_info['prompt_tokens']}, completion: {token_info['completion_tokens']})")
            if token_info.get('completion_tokens', 0) == 0:
                logging.debug("提示: API未返回completion token计数（或为0），通常不影响功能，仅影响统计显示。")
        except Exception as e:
            logging.error(f"智能体 {self.agent_id} 调用LLM时发生严重错误: {e}", exc_info=True)
            llm_response = {"error": f"LLM call exception: {e}", "default_response": True} # 标记为错误

        # 4. 解析LLM响应并验证
        bid = None
        if llm_response and not llm_response.get("default_response", False) and isinstance(llm_response, dict):
            try:
                # 基本结构验证
                if 'energy_bid' in llm_response and 'reserve_bid' in llm_response and \
                   'price' in llm_response['energy_bid'] and 'quantity' in llm_response['energy_bid'] and \
                   'price' in llm_response['reserve_bid'] and 'quantity' in llm_response['reserve_bid']:

                    energy_price = float(llm_response['energy_bid']['price'])
                    energy_quantity = float(llm_response['energy_bid']['quantity'])
                    reserve_price = float(llm_response['reserve_bid']['price'])
                    reserve_quantity = float(llm_response['reserve_bid']['quantity'])

                    # 数值范围和约束验证
                    if energy_price < 0 or reserve_price < 0 or energy_quantity < 0 or reserve_quantity < 0:
                         raise ValueError("价格或数量不能为负数。")

                    total_quantity = energy_quantity + reserve_quantity
                    if total_quantity > self.private_info['max_capacity'] * 1.01: # 允许微小超额以防LLM精度问题
                         logging.warning(f"智能体 {self.agent_id} LLM生成的总报价数量 {total_quantity:.2f} 超出容量 {self.private_info['max_capacity']:.2f}，将按比例缩减。")
                         scale_factor = self.private_info['max_capacity'] / total_quantity
                         energy_quantity *= scale_factor
                         reserve_quantity *= scale_factor
                    elif total_quantity < 0.01 and self.private_info['max_capacity'] > 0: # 如果LLM报空，但有容量，可能需要默认行为
                         logging.warning(f"智能体 {self.agent_id} LLM生成的总报价数量接近零，将使用默认保守报价。")
                         raise ValueError("总报价量过低。")


                    bid = {
                        "energy_bid": {"price": energy_price, "quantity": energy_quantity},
                        "reserve_bid": {"price": reserve_price, "quantity": reserve_quantity}
                    }
                    reasoning = llm_response.get('reasoning', '无')
                    logging.info(f"智能体 {self.agent_id} 成功生成报价: 能源({energy_price:.2f},{energy_quantity:.2f}), 备用({reserve_price:.2f},{reserve_quantity:.2f}). 理由: {reasoning[:50]}...")
                    self.last_bid = bid # 存储成功的报价

                else:
                    raise ValueError("LLM响应缺少必要的报价字段或结构不正确。")

            except (ValueError, TypeError, KeyError) as e:
                logging.error(f"智能体 {self.agent_id} 解析或验证LLM响应失败: {e}. LLM原始响应: {llm_response}")
                bid = None # 解析失败，不使用此响应

        else:
            # LLM调用失败或返回错误标志
            error_msg = llm_response.get("error", "未知LLM错误") if isinstance(llm_response, dict) else "LLM响应格式错误"
            logging.error(f"智能体 {self.agent_id} LLM调用失败或返回默认响应: {error_msg}")
            bid = None

        # 5. 失败时的后备策略
        if bid is None:
            logging.warning(f"智能体 {self.agent_id} 未能从LLM获取有效报价，将使用后备策略。")
            if self.last_bid:
                 logging.warning(f"  使用上一轮成功报价: {self.last_bid}")
                 bid = self.last_bid # 使用上一次成功的报价
            else:
                 # 简单的基于成本的保守报价作为最终后备
                 logging.warning("  使用基于成本的保守报价。")
                 bid = self._get_fallback_bid(market_info)

        # 在提交报价前，再次确保总量不超过容量 (可能在后备策略中也需要)
        if bid:
            # 若本轮无备用市场，则强制将备用报价置零（包括沿用last_bid的情形）
            reserve_requirement = market_info.get('reserve_requirement', 0)
            if reserve_requirement <= 0:
                bid['reserve_bid']['price'] = 0
                bid['reserve_bid']['quantity'] = 0
            total_q = bid['energy_bid']['quantity'] + bid['reserve_bid']['quantity']
            if total_q > self.private_info['max_capacity']:
                 logging.warning(f"  后备/最终报价总量 {total_q:.2f} 仍超容量，再次缩减。")
                 scale = self.private_info['max_capacity'] / total_q
                 bid['energy_bid']['quantity'] *= scale
                 bid['reserve_bid']['quantity'] *= scale

            # 应用最小出力与爬坡约束（在容量缩减之后）
            try:
                e_q = float(bid['energy_bid'].get('quantity', 0))
                r_q = float(bid['reserve_bid'].get('quantity', 0))
                e_q, r_q = self._apply_operational_constraints(e_q, r_q)
                bid['energy_bid']['quantity'] = e_q
                bid['reserve_bid']['quantity'] = r_q
            except Exception as e:
                logging.debug(f"约束应用失败（decide_bid）：{e}")

        # 记录决策时的信念状态和token使用信息，用于后续记忆存储
        bid['belief_at_decision'] = current_belief_state
        bid['token_usage'] = token_info

        return bid


    def _get_fallback_bid(self, market_info: Optional[Dict] = None) -> Dict:
        """生成一个简单、保守的后备报价；若无备用需求则不报备用。"""
        # 例如，报边际成本加上一个小的利润边际
        energy_price = self.private_info['marginal_cost_energy'] * 1.1
        total_capacity = self.private_info['max_capacity']
        reserve_requirement = (market_info or {}).get('reserve_requirement', 0)

        if reserve_requirement <= 0:
            # 无备用市场：全部容量用于能源，备用置零
            energy_quantity = total_capacity
            reserve_price = 0
            reserve_quantity = 0
        else:
            reserve_price = max(self.private_info['marginal_cost_reserve'] * 1.1, energy_price * 0.1)  # 备用价不低于能源价的10%
            # 简单分配容量，例如 80% 能源，20% 备用
            energy_quantity = total_capacity * 0.8
            reserve_quantity = total_capacity * 0.2
        # 应用最小出力与爬坡约束
        try:
            energy_quantity, reserve_quantity = self._apply_operational_constraints(energy_quantity, reserve_quantity)
        except Exception as e:
            logging.debug(f"约束应用失败（fallback）：{e}")

        return {
            "energy_bid": {"price": energy_price, "quantity": energy_quantity},
            "reserve_bid": {"price": reserve_price, "quantity": reserve_quantity},
            "belief_at_decision": self.belief_module.get_estimated_competitor_costs()  # Fallback时也记录信念
        }


    def update_state(self, round_number: int, market_results: Dict, my_bid_result: Dict):
        """
        更新信念、记忆和ADF模块。

        Args:
            round_number: 当前轮次。
            market_results: 市场总体结果 (价格等)。
            my_bid_result: 当前智能体的具体结果 (中标量、利润等)。
        """
        logging.debug(f"智能体 {self.agent_id} 开始更新状态 (轮次 {round_number})...")

        # 提取当前轮次利润用于ADF模块（兼容不同键名）
        current_profit = my_bid_result.get('profit', my_bid_result.get('total_profit', 0))
        self.last_profit = current_profit

        # 从结果中拿到提交的报价，供信念更新与记忆模块使用
        submitted_bid = my_bid_result.get('submitted_bid') or {}

        # 1. 更新信念模块（BeliefModule 需要 energy_bid 在顶层，构造最小所需结构）
        try:
            belief_input = {
                'energy_bid': submitted_bid.get('energy_bid', {}),
                'reserve_bid': submitted_bid.get('reserve_bid', {}),
                'cleared_energy': my_bid_result.get('cleared_energy', 0),
                'cleared_reserve': my_bid_result.get('cleared_reserve', 0),
                'profit': current_profit,
            }
            self.belief_module.update_beliefs(round_number, market_results, belief_input)
        except Exception as e:
            logging.error(f"智能体 {self.agent_id} 更新belief时出错: {e}", exc_info=True)

        # 2. 更新记忆模块
        try:
            # submitted_bid 已在上方提取
            if submitted_bid:
                belief_at_decision = submitted_bid.pop('belief_at_decision', {})  # 提取并移除信念状态
                # 将市场价格合并到 my_bid_result 中以便存储
                my_bid_result_for_memory = {**my_bid_result, **market_results}
                self.memory_module.add_record(round_number, submitted_bid, my_bid_result_for_memory, belief_at_decision)
            else:
                logging.warning(f"智能体 {self.agent_id} 无法记录记忆：缺少提交的报价信息。")

        except Exception as e:
            logging.error(f"智能体 {self.agent_id} 更新记忆时出错: {e}", exc_info=True)
        
        # 3. 更新ADF模块性能数据
        try:
            if self.adf_module is not None:
                self.adf_module.update_performance_data(round_number, current_profit, market_results)
            else:
                logging.debug(f"智能体 {self.agent_id} ADF已禁用，跳过性能数据更新。")
        except Exception as e:
            logging.error(f"智能体 {self.agent_id} 更新ADF模块时出错: {e}", exc_info=True)

    def decide_coupled_bid(self, round_number: int, market_info: Dict, history: Optional[Dict] = None, market_events_info: Optional[str] = None) -> Dict:
        """
        联合决策：同时生成电力市场和碳市场的报价。
        集成ADF（自适应决策频率）架构：
        - 根据触发条件决定使用LLM深度决策还是启发式快速决策
        - 平衡仿真真实性与计算成本
        返回: {'electricity': {...}, 'carbon': {...}}
        """
        logging.info(f"智能体 {self.agent_id} 开始ADF联合决策 (轮次 {round_number})...")
        
        # 如果禁用ADF，直接使用LLM深度决策
        if not self.enable_adf:
            logging.info(f"智能体 {self.agent_id} ADF已禁用，使用LLM深度决策")
            # 根据IBR-CR开关决定使用哪种决策方法
            if self.enable_ibr_cr:
                return self._decide_with_ibr_cr(round_number, market_info, history, market_events_info)
            else:
                return self._decide_coupled_bid_original(round_number, market_info, history, market_events_info)
        
        # 检查是否为初始化阶段，强制使用LLM建立基线
        if round_number < self.adf_module.config.get('min_llm_rounds', 3):
            logging.info(f"智能体 {self.agent_id} 初始化阶段第{round_number}轮，强制使用LLM建立基线")
            decision_type = 'llm_deep'
            trigger_reason = f'初始化阶段({round_number}/{self.adf_module.config.get("min_llm_rounds", 3)}轮)'
        else:
            # 使用ADF模块判断决策类型
            should_trigger, trigger_reason = self.adf_module.should_trigger_llm_decision(
                round_number, market_info, self.last_profit, market_events_info
            )
            decision_type = 'llm_deep' if should_trigger else 'heuristic'
        
        if decision_type == 'llm_deep':
            logging.info(f"智能体 {self.agent_id} 触发LLM深度决策 - 原因: {trigger_reason}")
            # 根据IBR-CR开关决定使用哪种LLM决策方法
            if self.enable_ibr_cr:
                bid_result = self._decide_with_ibr_cr(round_number, market_info, history, market_events_info)
            else:
                bid_result = self._decide_coupled_bid_original(round_number, market_info, history, market_events_info)
            
            # 更新ADF模块的LLM策略基线
            if bid_result and isinstance(bid_result, dict):
                elec_bid = bid_result.get('electricity')
                if isinstance(elec_bid, dict) and 'energy_bid' in elec_bid and 'reserve_bid' in elec_bid:
                    self.adf_module.update_llm_strategy(bid_result, self.private_info)
                    try:
                        e = elec_bid.get('energy_bid', {})
                        r = elec_bid.get('reserve_bid', {})
                        logging.debug(
                            f"ADF基线已更新: energy(price={e.get('price', 0):.4f}, qty={e.get('quantity', 0):.4f}), "
                            f"reserve(price={r.get('price', 0):.4f}, qty={r.get('quantity', 0):.4f})"
                        )
                    except Exception as e:
                        logging.debug(f"记录ADF基线更新详情时出错: {e}")
                else:
                    logging.warning(
                        f"LLM深度决策返回结构不含预期的'electricity/energy_bid/reserve_bid'，ADF基线未更新。返回键: {list(bid_result.keys())}"
                    )
                
        else:
            logging.info(f"智能体 {self.agent_id} 使用启发式决策 - 基于上次LLM策略")
            # 使用启发式决策
            heuristic_decision = self.adf_module.generate_heuristic_bid(
                round_number, market_info, self.private_info
            )
            
            # 添加空的token使用信息（启发式决策不消耗token）
            heuristic_decision['token_usage'] = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
            # 应用运行约束到电力报价
            try:
                elec_bid = heuristic_decision.get('electricity', {})
                if isinstance(elec_bid, dict) and 'energy_bid' in elec_bid and 'reserve_bid' in elec_bid:
                    e_q = float(elec_bid['energy_bid'].get('quantity', 0))
                    r_q = float(elec_bid['reserve_bid'].get('quantity', 0))
                    e_q, r_q = self._apply_operational_constraints(e_q, r_q)
                    elec_bid['energy_bid']['quantity'] = e_q
                    elec_bid['reserve_bid']['quantity'] = r_q
            except Exception as e:
                logging.debug(f"约束应用失败（启发式）：{e}")
            
            return heuristic_decision
            
        # 深度决策路径：在返回前应用运行约束
        try:
            if isinstance(bid_result, dict):
                elec_bid = bid_result.get('electricity', {})
                if isinstance(elec_bid, dict) and 'energy_bid' in elec_bid and 'reserve_bid' in elec_bid:
                    e_q = float(elec_bid['energy_bid'].get('quantity', 0))
                    r_q = float(elec_bid['reserve_bid'].get('quantity', 0))
                    e_q, r_q = self._apply_operational_constraints(e_q, r_q)
                    elec_bid['energy_bid']['quantity'] = e_q
                    elec_bid['reserve_bid']['quantity'] = r_q
        except Exception as e:
            logging.debug(f"约束应用失败（深度）：{e}")

        return bid_result

    def _apply_operational_constraints(self, energy_quantity: float, reserve_quantity: float) -> Tuple[float, float]:
        """将最小稳定出力与爬坡约束应用到(能源, 备用)报价数量上。
        规则：
        - 能源报价量不得低于 min_output
        - (能源+备用) 的总量相对上一轮总出清，不能超过 ramp_up_mw 上升或 ramp_down_mw 下降
        - 不得超过 max_capacity
        优先满足最小出力，其次压缩备用以满足爬坡/容量约束。
        """
        cap = float(self.private_info.get('max_capacity', 0) or 0)
        min_output = float(self.private_info.get('min_output', 0) or 0)
        ramp_up = float(self.private_info.get('ramp_up_mw', 0) or 0)
        ramp_down = float(self.private_info.get('ramp_down_mw', 0) or 0)

        # 1) 最小出力（仅对能源）
        e = max(float(energy_quantity), min_output)
        r = max(float(reserve_quantity), 0.0)

        # 2) 总量爬坡
        prev_total = max(0.0, float(self.last_cleared_energy) + float(self.last_cleared_reserve))
        max_total = cap if ramp_up <= 0 else min(cap, prev_total + ramp_up)
        min_total = 0.0 if ramp_down <= 0 else max(0.0, prev_total - ramp_down)

        total = e + r
        if total > max_total + 1e-9:
            # 优先削减备用
            over = total - max_total
            r = max(0.0, r - over)
            total = e + r
            if total > max_total + 1e-9:
                # 仍然超出，削减能源但不低于最小出力
                e = max(min_output, e - (total - max_total))
        elif total < min_total - 1e-9:
            # 低于可行最小总量，尽量提升能源（不超过容量）
            shortage = min_total - total
            e = min(cap, e + shortage)
            total = e + r

        # 3) 容量上限
        if e + r > cap and (e + r) > 0:
            scale = cap / (e + r)
            e *= scale
            r *= scale

        # 清理数值
        e = max(0.0, float(e))
        r = max(0.0, float(r))
        return e, r
    
    def _decide_with_ibr_cr(self, round_number: int, market_info: Dict, history: Optional[Dict] = None, market_events_info: Optional[str] = None, k: int = 2) -> Dict:
        """
        IBR-CR机制：迭代信念修正与反事实推理
        
        Args:
            round_number: 轮次
            market_info: 市场信息
            history: 历史信息
            market_events_info: 市场事件信息
            k: 迭代深度，默认为2
            
        Returns:
            最终优化的报价决策
        """
        total_token_info = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
        
        try:
            # 第0步：生成初始报价 bid_i^(0)
            logging.info(f"智能体 {self.agent_id} IBR-CR 第0步：生成初始意向报价")
            initial_bid, step0_tokens = self._generate_initial_bid(round_number, market_info, history, market_events_info)
            self._accumulate_tokens(total_token_info, step0_tokens)
            
            if k == 0:
                # 如果k=0，直接返回初始报价
                initial_bid['token_usage'] = total_token_info
                return initial_bid
            
            # 第1步：信念修正/换位思考 - 模拟竞争对手反应
            logging.info(f"智能体 {self.agent_id} IBR-CR 第1步：模拟竞争对手对初始报价的反应")
            competitor_response, step1_tokens = self._simulate_competitor_response(initial_bid, round_number, market_info, history, market_events_info)
            self._accumulate_tokens(total_token_info, step1_tokens)
            
            if k == 1:
                # 如果k=1，返回初始报价但记录竞争对手预期
                initial_bid['token_usage'] = total_token_info
                initial_bid['competitor_expectation'] = competitor_response
                return initial_bid
            
            # 第2步：反事实推理/策略优化 - 基于竞争对手预期优化报价
            logging.info(f"智能体 {self.agent_id} IBR-CR 第2步：基于竞争对手预期优化最终报价")
            final_bid, step2_tokens = self._optimize_bid_with_competitor_expectation(
                initial_bid, competitor_response, round_number, market_info, history, market_events_info
            )
            self._accumulate_tokens(total_token_info, step2_tokens)
            
            final_bid['token_usage'] = total_token_info
            final_bid['ibr_cr_steps'] = {
                'initial_bid': initial_bid,
                'competitor_response': competitor_response,
                'final_bid': final_bid
            }
            
            logging.info(f"智能体 {self.agent_id} IBR-CR 完成，总Token使用: {total_token_info['total_tokens']}")
            return final_bid
            
        except Exception as e:
            logging.error(f"智能体 {self.agent_id} IBR-CR机制执行失败: {e}")
            # 回退到原始方法
            return self._decide_coupled_bid_original(round_number, market_info, history, market_events_info)
    
    def _decide_coupled_bid_original(self, round_number: int, market_info: Dict, history: Optional[Dict] = None, market_events_info: Optional[str] = None) -> Dict:
        """
        原始的联合决策方法（作为IBR-CR的fallback）
        """
        prompt = self.build_coupled_prompt(round_number, market_info, history=history, market_events_info=market_events_info)

        # 添加LLM调用的错误处理
        try:
            llm_response, token_info = self.llm_client.chat(prompt, return_json=True, track_tokens=True)
            logging.info(f"智能体 {self.agent_id} Token使用: {token_info['total_tokens']} tokens (prompt: {token_info['prompt_tokens']}, completion: {token_info['completion_tokens']})")
            if token_info.get('completion_tokens', 0) == 0:
                logging.debug("提示: API未返回completion token计数（或为0），通常不影响功能，仅影响统计显示。")
        except Exception as e:
            logging.warning(f"智能体 {self.agent_id} LLM调用失败: {e}")
            logging.info(f"智能体 {self.agent_id} 使用fallback策略生成报价")
            return self._get_fallback_coupled_bid(market_info)
        
        try:
            bid = None

            # 如果LLM返回的是原始响应格式
            if isinstance(llm_response, dict) and 'candidates' in llm_response:
                text = llm_response['candidates'][0]['content']['parts'][0]['text']
                text = re.sub(r"^```json|^```|```$", "", text.strip(), flags=re.IGNORECASE)
                bid = json.loads(text)
            # 如果LLM返回的是包含text的格式
            elif isinstance(llm_response, dict) and 'text' in llm_response:
                text = llm_response['text']
                text = re.sub(r"^```json|^```|```$", "", text.strip(), flags=re.IGNORECASE)
                bid = json.loads(text)
            # 如果LLM返回的是已解析的JSON（新的格式）
            elif isinstance(llm_response, dict) and ('reasoning' in llm_response or 'electricity_bid' in llm_response):
                bid = llm_response

            if bid is not None:
                # 验证和处理电力市场的报价
                elec_bid = bid.get('electricity_bid', {})
                if elec_bid:
                    energy_bid = elec_bid.get('energy_bid', {'quantity': 0})
                    reserve_bid = elec_bid.get('reserve_bid', {'quantity': 0})

                    energy_quantity = float(energy_bid.get('quantity', 0))
                    reserve_quantity = float(reserve_bid.get('quantity', 0))

                    total_quantity = energy_quantity + reserve_quantity
                    max_capacity = self.private_info['max_capacity']

                    # 如果总报价量超过最大容量，则按比例缩减
                    if total_quantity > max_capacity:
                        logging.warning(
                            f"智能体 {self.agent_id} 联合决策生成的总报价数量 "
                            f"{total_quantity:.2f} 超出容量 {max_capacity:.2f}，将按比例缩减。"
                        )
                        if total_quantity > 0:
                            scale_factor = max_capacity / total_quantity
                            if 'quantity' in energy_bid:
                                energy_bid['quantity'] *= scale_factor
                            if 'quantity' in reserve_bid:
                                reserve_bid['quantity'] *= scale_factor

                # 若无备用市场，强制将解析出的备用报价置零
                if market_info.get('reserve_requirement', 0) <= 0 and elec_bid:
                    if 'reserve_bid' in elec_bid:
                        elec_bid['reserve_bid']['price'] = 0
                        elec_bid['reserve_bid']['quantity'] = 0

                return {
                    'electricity': elec_bid,
                    'carbon': bid.get('carbon_bid', {}),
                    'token_usage': token_info # 添加token使用信息
                }
        except (json.JSONDecodeError, ValueError, TypeError, KeyError) as e:
            logging.error(f"智能体 {self.agent_id} 联合决策LLM响应解析或验证失败: {e}, 原始: {llm_response}")
            return self._get_fallback_coupled_bid(market_info)

        # 如果LLM调用失败或返回空文本
        logging.error(f"智能体 {self.agent_id} 联合决策LLM调用失败或返回空响应, 原始: {llm_response}")
        # 确保token_info已定义
        if not 'token_info' in locals():
            token_info = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
        return self._get_fallback_coupled_bid(market_info)

    def build_coupled_prompt(self, round_number, market_info, history=None, market_events_info=None):
        history_str = ""
        if history and (history.get('market_prices') or history.get('awards')):
            prices = history.get('market_prices', [])
            awards = history.get('awards', [])
            history_str = (
                "\n### 历史参考信息\n"
                f"- 最近几轮电力市场出清价: {prices[-5:] if prices else '无'}\n"
                f"- 你在最近几轮的电力市场中标情况 (True=中标): {awards[-5:] if awards else '无'}\n"
            )
        
        # 获取碳市场信息
        carbon_market = market_info.get('carbon_market', {})
        current_carbon_price = carbon_market.get('current_price', 50.0)
        current_quota = carbon_market.get('current_quota', 80.0)
        
        # 检查备用市场需求
        reserve_requirement = market_info.get('reserve_requirement', 0)
        has_reserve_market = reserve_requirement > 0

        # 获取市场事件信息
        market_events_str = ""
        if market_events_info:
            market_events_str = (
                "\n### 🚨 重要市场事件公告\n"
                f"{market_events_info}\n\n"
                "⚠️ 市场事件响应指南:\n"
                "- 分析事件对你的成本结构的直接影响，特别是当事件涉及你使用的燃料类型\n"
                "- 对于价格变化公告：考虑在实际变化前反映未来成本，同时评估市场接受度\n"
                "- 评估市场价格趋势变化，在价格实际变动前适度调整报价策略\n"
                "- 考虑优化容量分配，根据预期的市场变化调整能源和备用的比例\n"
                "- 预判其他市场参与者的可能反应，尤其是使用相同或替代燃料类型的竞争对手\n"
            )
        
        # 添加技术类型特定指导
        tech_type = self.private_info.get('technology_type', '').lower()
        tech_guidance = ""
        
        # 根据技术类型提供特定指导，但保持通用性
        if any(fuel in tech_type for fuel in ['gas', '燃气', '天然气']):
            tech_guidance = (
                "\n### 📊 燃气机组特定策略指导\n"
                "- 你是燃气发电机组，对燃料价格变化较为敏感\n"
                "- 关注燃气市场动态，及时调整报价策略\n"
                "- 根据市场情况灵活分配能源和备用容量\n"
                "- 密切关注市场出清价格，根据中标情况调整策略\n"
            )
        elif any(fuel in tech_type for fuel in ['coal', '煤', '燃煤']):
            tech_guidance = (
                "\n### 📊 煤电机组特定策略指导\n"
                "- 你是煤电机组，对燃料价格变化有一定敏感度\n"
                "- 关注煤炭市场动态，及时调整报价策略\n"
                "- 煤电机组通常作为基荷运行，可以保持较高的能源报价比例\n"
            )
        elif any(fuel in tech_type for fuel in ['hydro', '水电', '水力']):
            tech_guidance = (
                "\n### 📊 水电机组特定策略指导\n"
                "- 你是水电机组，边际成本较低，对燃料价格变化不敏感\n"
                "- 关注市场整体价格趋势，适时调整报价策略\n"
                "- 水电机组灵活性高，可以在备用市场获得较好收益\n"
            )
        elif any(fuel in tech_type for fuel in ['wind', '风电', '风力']):
            tech_guidance = (
                "\n### 📊 风电机组特定策略指导\n"
                "- 你是风电机组，边际成本极低，对燃料价格变化不敏感\n"
                "- 关注市场整体价格趋势，适时调整报价策略\n"
                "- 风电出力不稳定，需要合理安排能源和备用容量\n"
            )
        elif any(fuel in tech_type for fuel in ['solar', '光伏', '太阳能']):
            tech_guidance = (
                "\n### 📊 光伏机组特定策略指导\n"
                "- 你是光伏机组，边际成本极低，对燃料价格变化不敏感\n"
                "- 关注市场整体价格趋势，适时调整报价策略\n"
                "- 光伏出力受天气影响，需要合理安排能源和备用容量\n"
            )
        
        private_info_str = json.dumps(self.private_info, ensure_ascii=False, indent=2)
        market_info_str = json.dumps(market_info, ensure_ascii=False, indent=2)

        prompt = (
            f"你是一个名为 {self.agent_id} 的发电企业智能体，目标是在电力和碳市场中最大化长期利润。你是一个理性的市场参与者。\n\n"
            f"### 你的内部参数:\n{private_info_str}\n\n"
            f"### 当前市场信息 (轮次 {round_number}):\n{market_info_str}\n"
            f"{market_events_str}"
            f"{tech_guidance}"
            f"{history_str}\n"
            f"### 🌱 碳市场交易机制\n"
            f"- 当前碳价格: {current_carbon_price:.2f} 元/吨CO2\n"
            f"- 有效报价范围: [{current_carbon_price*0.9:.2f} - {current_carbon_price*1.1:.2f}] 元/吨\n"
            f"- 你当前持有配额: {current_quota:.1f} 吨CO2\n"
            f"- 建议交易量: 5-20 吨 (即使小额交易也有流动性奖励)\n"
            f"- 🔥 强烈建议每轮都参与碳市场交易以获得额外收益！\n\n"
            f"### 决策指导原则\n"
            f"**利润最大化策略：**\n"
            f"1. **成本效益分析**: 你的边际成本是制定报价的基准。能源成本: {self.private_info['marginal_cost_energy']:.2f} 元/MWh，备用成本: {self.private_info['marginal_cost_reserve']:.2f} 元/MWh\n"
            f"2. **预期行为策略**: 当市场公告未来变化时，应及时调整报价以反映预期影响\n"
            f"3. **竞争定价**: 考虑历史价格趋势。报价应在成本基础上加合理利润，但要保持竞争力\n"
            f"4. **容量优化**: 总容量 {self.private_info['max_capacity']:.2f} MW，合理分配能源和备用报价以最大化收入\n"
            f"5. **市场分析**: 观察市场价格趋势，根据市场情况调整报价策略\n"
            f"6. **碳配额管理**: 预估发电导致的碳排放，合理买卖配额\n\n"
            f"**重要约束:**\n"
            f"1. **容量约束**: 能源 + 备用报价总量 ≤ {self.private_info['max_capacity']:.2f} MW\n"
            f"2. **成本约束**: 报价应 ≥ 边际成本 + 合理利润边际\n"
            f"3. **预期影响**: 如有市场事件公告，应考虑其对成本和市场的潜在影响\n"
            f"4. **碳排放**: 考虑排放因子 {self.private_info.get('emission_factor', 0.8):.2f}，预估配额需求\n"
            f"5. **碳价格约束**: 碳报价必须在 [{current_carbon_price*0.9:.2f}, {current_carbon_price*1.1:.2f}] 范围内\n"
            f"6. **备用市场状态**: 本轮备用需求为 {reserve_requirement:.2f} MW {'(有备用市场)' if has_reserve_market else '(无备用市场)'}\n\n"
            f"**报价建议：**\n"
            f"- 能源报价: 在边际成本基础上加合理利润\n"
            + (f"- 备用报价: 考虑备用机会成本，通常低于能源报价\n" if has_reserve_market else "- 备用报价: 本轮无备用需求，设置为0或不报价\n") +
            f"- 数量分配: 根据市场情况合理分配能源和备用容量\n"
            f"- 碳交易策略: 根据预期排放和持有配额决定买入或卖出\n\n"
            f"请基于以上分析，制定理性的报价策略。输出严格的JSON格式，不要添加其他文字。\n\n"
            "```json\n"
            "{\n"
            "  \"reasoning\": \"在这里简要描述你的决策原因，比如你对市场价格的判断、你的成本、以及你的报价策略是如何结合这些因素的。\",\n"
            "  \"electricity_bid\": {\n"
            "    \"energy_bid\": {\"price\": <你的能源报价价格>, \"quantity\": <你的能源报价数量>}" +
            (",\n    \"reserve_bid\": {\"price\": <你的备用报价价格>, \"quantity\": <你的备用报价数量>}" if has_reserve_market else ",\n    \"reserve_bid\": {\"price\": 0, \"quantity\": 0}") +
            "\n  },\n"
            "  \"carbon_bid\": {\n"
            "    \"buy_quantity\": <你希望购买的碳配额数量(吨CO2)>,\n"
            "    \"buy_price\": <你的碳配额买入价格(元/吨)>,\n"
            "    \"sell_quantity\": <你希望卖出的碳配额数量(吨CO2)>,\n"
            "    \"sell_price\": <你的碳配额卖出价格(元/吨)>\n"
            "  }\n"
            "}\n"
            "```\n"
        )
        return prompt

    def _get_fallback_coupled_bid(self, market_info: Optional[Dict] = None):
        fb = self._get_fallback_bid(market_info)
        fallback_energy = fb['energy_bid']
        fallback_reserve = fb['reserve_bid']
        max_capacity = self.private_info.get('max_capacity', 100)
        min_output = self.private_info.get('min_output', 0)
        
        # 自动修正超容量
        e_q = fallback_energy.get('quantity', 0)
        r_q = fallback_reserve.get('quantity', 0)
        total = max(0, e_q) + max(0, r_q)
        if total > max_capacity:
            # 按比例缩放
            scale = max_capacity / total if total > 0 else 0
            fallback_energy['quantity'] = max(0, e_q) * scale
            fallback_reserve['quantity'] = max(0, r_q) * scale
        else:
            fallback_energy['quantity'] = max(0, e_q)
            fallback_reserve['quantity'] = max(0, r_q)
        
        # 生成有意义的碳市场Fallback报价
        base_carbon_price = 50.0  # 默认碳价格
        fallback_carbon = {
            'buy_quantity': 8,  # 适中的买入量
            'buy_price': base_carbon_price * 0.95,  # 略低于市价买入
            'sell_quantity': 5,  # 较少的卖出量
            'sell_price': base_carbon_price * 1.05  # 略高于市价卖出
        }
        
        # 添加空的token使用信息
        empty_token_info = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
        
        # 若无备用市场，确保备用置零
        if (market_info or {}).get('reserve_requirement', 0) <= 0:
            fallback_reserve['price'] = 0
            fallback_reserve['quantity'] = 0

        return {
            'electricity': {
                'energy_bid': fallback_energy,
                'reserve_bid': fallback_reserve,
                'max_capacity': max_capacity,
                'min_output': min_output
            },
            'carbon': fallback_carbon,
            'token_usage': empty_token_info  # 添加空的token使用信息
        }
    
    def _generate_initial_bid(self, round_number: int, market_info: Dict, history: Optional[Dict], market_events_info: Optional[str]) -> Tuple[Dict, Dict]:
        """
        IBR-CR 第0步：生成初始意向报价 bid_i^(0)
        
        Returns:
            Tuple[初始报价, token使用信息]
        """
        prompt = self.build_coupled_prompt(round_number, market_info, history=history, market_events_info=market_events_info)
        prompt += "\n\n### IBR-CR 第0步：初始意向报价\n请生成你的初始报价意向，这将作为后续策略优化的基础。"
        
        try:
            llm_response, token_info = self.llm_client.chat(prompt, return_json=True, track_tokens=True)
            bid = self._parse_bid_response(llm_response)
            if bid:
                return bid, token_info
        except Exception as e:
            logging.warning(f"智能体 {self.agent_id} 初始报价生成失败: {e}")
        
        # 回退到fallback
        return self._get_fallback_coupled_bid(market_info), {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
    
    def _simulate_competitor_response(self, initial_bid: Dict, round_number: int, market_info: Dict, history: Optional[Dict], market_events_info: Optional[str]) -> Tuple[Dict, Dict]:
        """
        IBR-CR 第1步：信念修正/换位思考 - 模拟理性竞争对手对初始报价的最优反应
        
        Args:
            initial_bid: 自己的初始报价 bid_i^(0)
            
        Returns:
            Tuple[预期的竞争对手报价 bid_comp^(0), token使用信息]
        """
        # 检查备用市场需求
        reserve_requirement = market_info.get('reserve_requirement', 0)
        has_reserve_market = reserve_requirement > 0
        
        # 构建换位思考的提示词
        initial_bid_str = json.dumps({
            'electricity': initial_bid.get('electricity', {}),
            'carbon': initial_bid.get('carbon', {})
        }, ensure_ascii=False, indent=2)
        
        market_info_str = json.dumps(market_info, ensure_ascii=False, indent=2)
        
        prompt = (
            f"你现在需要模拟一个理性的竞争对手的视角。假设你是市场中的一个典型竞争对手，"
            f"观察到智能体 {self.agent_id} 提交了以下初始报价：\n\n"
            f"### 观察到的初始报价:\n{initial_bid_str}\n\n"
            f"### 当前市场信息 (轮次 {round_number}):\n{market_info_str}\n\n"
            f"### 备用市场状态: 本轮备用需求为 {reserve_requirement:.2f} MW {'(有备用市场)' if has_reserve_market else '(无备用市场)'}\n\n"
            f"### 竞争对手分析任务\n"
            f"作为一个理性的竞争对手，请分析这个初始报价，并预测其他竞争对手（包括你自己）会如何反应。\n"
            f"考虑以下因素：\n"
            f"1. 这个报价的竞争力如何？价格是否过高或过低？\n"
            f"2. 如果你是竞争对手，你会如何调整自己的报价来应对？\n"
            f"3. 市场中其他参与者可能的反应是什么？\n"
            f"4. 这种报价策略可能导致什么样的市场结果？\n\n"
            f"请输出一个典型竞争对手的预期报价策略：\n\n"
            "```json\n"
            "{\n"
            "  \"analysis\": \"对初始报价的分析和竞争对手可能反应的推理\",\n"
            "  \"expected_competitor_strategy\": {\n"
            "    \"electricity_bid\": {\n"
            "      \"energy_bid\": {\"price\": <预期竞争对手能源报价>, \"quantity\": <预期数量>}" +
            (",\n      \"reserve_bid\": {\"price\": <预期竞争对手备用报价>, \"quantity\": <预期数量>}" if has_reserve_market else ",\n      \"reserve_bid\": {\"price\": 0, \"quantity\": 0}") +
            "\n    },\n"
            "    \"carbon_bid\": {\n"
            "      \"buy_quantity\": <预期竞争对手碳配额买入量>,\n"
            "      \"buy_price\": <预期买入价格>,\n"
            "      \"sell_quantity\": <预期竞争对手碳配额卖出量>,\n"
            "      \"sell_price\": <预期卖出价格>\n"
            "    }\n"
            "  },\n"
            "  \"market_impact\": \"预期这种竞争格局对市场价格和清算结果的影响\"\n"
            "}\n"
            "```\n"
        )
        
        try:
            llm_response, token_info = self.llm_client.chat(prompt, return_json=True, track_tokens=True)
            competitor_response = self._parse_competitor_response(llm_response)
            if competitor_response:
                return competitor_response, token_info
        except Exception as e:
            logging.warning(f"智能体 {self.agent_id} 竞争对手反应模拟失败: {e}")
        
        # 回退到默认竞争对手反应
        reserve_requirement = market_info.get('reserve_requirement', 0)
        has_reserve_market = reserve_requirement > 0
        
        default_response = {
            'analysis': '竞争对手分析失败，使用默认预期',
            'expected_competitor_strategy': {
                'electricity_bid': {
                    'energy_bid': {'price': 30.0, 'quantity': 50.0},
                    'reserve_bid': {'price': 8.0, 'quantity': 20.0} if has_reserve_market else {'price': 0, 'quantity': 0}
                },
                'carbon_bid': {
                    'buy_quantity': 10.0, 'buy_price': 48.0,
                    'sell_quantity': 5.0, 'sell_price': 52.0
                }
            },
            'market_impact': '预期市场竞争激烈'
        }
        return default_response, {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
    
    def _optimize_bid_with_competitor_expectation(self, initial_bid: Dict, competitor_response: Dict, round_number: int, market_info: Dict, history: Optional[Dict], market_events_info: Optional[str]) -> Tuple[Dict, Dict]:
        """
        IBR-CR 第2步：反事实推理/策略优化 - 基于竞争对手预期重新优化报价
        
        Args:
            initial_bid: 初始报价 bid_i^(0)
            competitor_response: 预期的竞争对手反应 bid_comp^(0)
            
        Returns:
            Tuple[最终优化报价 bid_i^(1), token使用信息]
        """
        initial_bid_str = json.dumps({
            'electricity': initial_bid.get('electricity', {}),
            'carbon': initial_bid.get('carbon', {})
        }, ensure_ascii=False, indent=2)
        
        competitor_str = json.dumps(competitor_response.get('expected_competitor_strategy', {}), ensure_ascii=False, indent=2)
        market_info_str = json.dumps(market_info, ensure_ascii=False, indent=2)
        
        prompt = (
            f"你是智能体 {self.agent_id}，现在需要基于对竞争对手反应的预期来优化你的最终报价策略。\n\n"
            f"### 你的初始报价意向:\n{initial_bid_str}\n\n"
            f"### 预期的竞争对手反应:\n{competitor_str}\n\n"
            f"### 竞争分析:\n{competitor_response.get('analysis', '无分析')}\n\n"
            f"### 市场影响预期:\n{competitor_response.get('market_impact', '无预期')}\n\n"
            f"### 当前市场信息 (轮次 {round_number}):\n{market_info_str}\n\n"
            f"### 策略优化任务\n"
            f"基于以上信息，请重新优化你的报价策略。考虑：\n"
            f"1. **反事实推理**: 如果竞争对手按预期反应，你的初始报价是否仍然最优？\n"
            f"2. **策略调整**: 是否需要调整价格或数量来应对预期的竞争？\n"
            f"3. **市场定位**: 在预期的竞争格局中，如何定位自己的报价？\n"
            f"4. **风险管理**: 如何平衡收益最大化和中标概率？\n"
            f"5. **动态适应**: 如何利用对竞争对手的预期来获得竞争优势？\n\n"
            f"### 你的私有信息提醒\n"
            f"- 边际成本: 能源 {self.private_info['marginal_cost_energy']:.2f} 元/MWh, 备用 {self.private_info['marginal_cost_reserve']:.2f} 元/MWh\n"
            f"- 最大容量: {self.private_info['max_capacity']:.2f} MW\n"
            f"- 技术类型: {self.private_info.get('technology_type', '未指定')}\n\n"
            f"请输出你的最终优化报价：\n\n"
            "```json\n"
            "{\n"
            "  \"optimization_reasoning\": \"基于竞争对手预期的策略优化推理过程\",\n"
            "  \"strategy_changes\": \"相比初始报价的主要调整和原因\",\n"
            "  \"electricity_bid\": {\n"
            "    \"energy_bid\": {\"price\": <最终能源报价>, \"quantity\": <最终能源数量>},\n"
            "    \"reserve_bid\": {\"price\": <最终备用报价>, \"quantity\": <最终备用数量>}\n"
            "  },\n"
            "  \"carbon_bid\": {\n"
            "    \"buy_quantity\": <最终碳配额买入量>,\n"
            "    \"buy_price\": <最终买入价格>,\n"
            "    \"sell_quantity\": <最终碳配额卖出量>,\n"
            "    \"sell_price\": <最终卖出价格>\n"
            "  },\n"
            "  \"expected_outcome\": \"对这个优化策略预期效果的评估\"\n"
            "}\n"
            "```\n"
        )
        
        try:
            llm_response, token_info = self.llm_client.chat(prompt, return_json=True, track_tokens=True)
            final_bid = self._parse_bid_response(llm_response)
            if final_bid:
                return final_bid, token_info
        except Exception as e:
            logging.warning(f"智能体 {self.agent_id} 最终报价优化失败: {e}")
        
        # 回退到初始报价
        return initial_bid, {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
    
    def _parse_bid_response(self, llm_response: Any) -> Optional[Dict]:
        """
        解析LLM返回的报价响应
        """
        try:
            bid = None
            
            # 如果LLM返回的是原始响应格式
            if isinstance(llm_response, dict) and 'candidates' in llm_response:
                text = llm_response['candidates'][0]['content']['parts'][0]['text']
                text = re.sub(r"^```json|^```|```$", "", text.strip(), flags=re.IGNORECASE)
                bid = json.loads(text)
            # 如果LLM返回的是包含text的格式
            elif isinstance(llm_response, dict) and 'text' in llm_response:
                text = llm_response['text']
                text = re.sub(r"^```json|^```|```$", "", text.strip(), flags=re.IGNORECASE)
                bid = json.loads(text)
            # 如果LLM返回的是已解析的JSON
            elif isinstance(llm_response, dict) and ('reasoning' in llm_response or 'electricity_bid' in llm_response):
                bid = llm_response
            
            if bid:
                # 验证和处理电力市场的报价
                elec_bid = bid.get('electricity_bid', {})
                if elec_bid:
                    energy_bid = elec_bid.get('energy_bid', {'quantity': 0})
                    reserve_bid = elec_bid.get('reserve_bid', {'quantity': 0})
                    
                    energy_quantity = float(energy_bid.get('quantity', 0))
                    reserve_quantity = float(reserve_bid.get('quantity', 0))
                    
                    total_quantity = energy_quantity + reserve_quantity
                    max_capacity = self.private_info['max_capacity']
                    
                    # 如果总报价量超过最大容量，则按比例缩减
                    if total_quantity > max_capacity:
                        logging.warning(
                            f"智能体 {self.agent_id} 报价总量 {total_quantity:.2f} 超出容量 {max_capacity:.2f}，按比例缩减"
                        )
                        if total_quantity > 0:
                            scale_factor = max_capacity / total_quantity
                            energy_bid['quantity'] = energy_quantity * scale_factor
                            reserve_bid['quantity'] = reserve_quantity * scale_factor
                
                return {
                    'electricity': elec_bid,
                    'carbon': bid.get('carbon_bid', {})
                }
            
        except (json.JSONDecodeError, ValueError, TypeError, KeyError) as e:
            logging.error(f"智能体 {self.agent_id} 报价响应解析失败: {e}")
        
        return None
    
    def _parse_competitor_response(self, llm_response: Any) -> Optional[Dict]:
        """
        解析竞争对手反应的LLM响应
        """
        try:
            response = None
            
            # 如果LLM返回的是原始响应格式
            if isinstance(llm_response, dict) and 'candidates' in llm_response:
                text = llm_response['candidates'][0]['content']['parts'][0]['text']
                text = re.sub(r"^```json|^```|```$", "", text.strip(), flags=re.IGNORECASE)
                response = json.loads(text)
            # 如果LLM返回的是包含text的格式
            elif isinstance(llm_response, dict) and 'text' in llm_response:
                text = llm_response['text']
                text = re.sub(r"^```json|^```|```$", "", text.strip(), flags=re.IGNORECASE)
                response = json.loads(text)
            # 如果LLM返回的是已解析的JSON
            elif isinstance(llm_response, dict):
                response = llm_response
            
            return response
            
        except (json.JSONDecodeError, ValueError, TypeError, KeyError) as e:
            logging.error(f"智能体 {self.agent_id} 竞争对手响应解析失败: {e}")
        
        return None
    
    def _accumulate_tokens(self, total_tokens: Dict, step_tokens: Dict) -> None:
        """
        累积token使用统计
        """
        total_tokens['prompt_tokens'] += step_tokens.get('prompt_tokens', 0)
        total_tokens['completion_tokens'] += step_tokens.get('completion_tokens', 0)
        total_tokens['total_tokens'] += step_tokens.get('total_tokens', 0)