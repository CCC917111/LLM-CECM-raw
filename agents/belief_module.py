# llm_market_sim/agents/belief_module.py
import random
import logging
from typing import Dict, List, Any, Optional

# 假设我们有一个LLMClient实例可以用于辅助更新 (可选)
# from ..llm_interface.llm_client import LLMClient
# from ..llm_interface.prompts import format_belief_update_prompt

class BeliefModule:
    """
    管理智能体对竞争对手信念的模块。
    简化实现：主要关注对竞争对手平均边际成本的估计。
    """
    def __init__(self, agent_id: str, competitor_ids: List[str], initial_belief_params: Optional[Dict] = None):
        """
        初始化信念模块。

        Args:
            agent_id: 当前智能体的ID。
            competitor_ids: 竞争对手的ID列表。
            initial_belief_params: 初始信念参数，例如:
                {
                    "avg_cost_estimation_mean": 40.0, # 初始估计竞争对手平均成本的均值
                    "avg_cost_estimation_std": 15.0,  # 初始估计竞争对手平均成本的标准差
                    "update_aggressiveness": 0.1 # 信念更新的学习率/幅度
                }
        """
        self.agent_id = agent_id
        self.competitor_ids = competitor_ids
        self.params = initial_belief_params or {}
        self.learning_rate = self.params.get("update_aggressiveness", 0.1)

        # 核心信念：对每个竞争对手的平均边际成本的估计 (均值和标准差)
        # 简化：先只维护一个对所有竞争对手的总体估计
        self.estimated_competitor_avg_cost_mean = self.params.get("avg_cost_estimation_mean", 40.0)
        self.estimated_competitor_avg_cost_std = self.params.get("avg_cost_estimation_std", 15.0)

        # 可以扩展为对每个竞争对手分别维护信念
        # self.competitor_beliefs = {
        #     cid: {"cost_mean": ..., "cost_std": ...} for cid in competitor_ids
        # }

        # 信念的文本摘要，可以由LLM生成或基于规则生成
        self.belief_summary_text = f"初始估计：竞争对手平均成本可能在 {self.estimated_competitor_avg_cost_mean - self.estimated_competitor_avg_cost_std:.1f} 到 {self.estimated_competitor_avg_cost_mean + self.estimated_competitor_avg_cost_std:.1f} 范围内。"

        # (可选) LLM客户端用于辅助更新
        # self.llm_client = LLMClient(...) # 需要传递或创建实例

        logging.info(f"智能体 {agent_id} 的信念模块初始化完成。初始信念: {self.belief_summary_text}")


    def get_belief_summary(self) -> str:
        """获取当前信念的文本摘要"""
        # 可以增加逻辑，如果数值变化不大，摘要不变
        return self.belief_summary_text


    def update_beliefs(self, round_number: int, market_results: Dict, my_bid_result: Dict):
        """
        根据市场结果更新信念。
        这是一个简化的启发式更新逻辑。可以替换为更复杂的贝叶斯更新或LLM辅助更新。

        Args:
            round_number: 当前轮次。
            market_results: 包含市场出清价格等的字典。
                {'energy_price': float, 'reserve_price': float, ...}
            my_bid_result: 当前智能体在本轮的报价和中标结果。
                {'energy_bid': {'price': P_e, 'quantity': Q_e}, 'reserve_bid': {'price': P_r, 'quantity': Q_r},
                 'cleared_energy': float, 'cleared_reserve': float, 'profit': float}
        """
        energy_price = market_results.get('energy_price')
        reserve_price = market_results.get('reserve_price')
        my_energy_bid = my_bid_result.get('energy_bid', {})
        my_energy_bid_price = my_energy_bid.get('price')
        my_cleared_energy = my_bid_result.get('cleared_energy', 0)
        my_energy_bid_quantity = my_energy_bid.get('quantity', 0)

        if None in [energy_price, my_energy_bid_price]:
            logging.warning(f"智能体 {self.agent_id} 信念更新跳过：缺少必要的市场价格或自身报价信息。")
            return

        logging.debug(f"智能体 {self.agent_id} 开始更新信念 (轮次 {round_number})...")
        logging.debug(f"  市场能源价格: {energy_price:.2f}, 我的能源报价: {my_energy_bid_price:.2f}, 中标量: {my_cleared_energy:.2f}/{my_energy_bid_quantity:.2f}")

        # --- 启发式更新逻辑 ---
        # 如果市场价格远高于我的报价，并且我中标了，可能意味着竞争对手成本较高或报价保守
        # 如果市场价格低于我的报价，或者我没中标，可能意味着竞争对手成本较低或报价激进

        cost_adjustment_factor = 0.0 # 成本估计调整因子

        # 场景1: 市场价格显著高于我的报价，且我中标了
        if my_cleared_energy > 0 and energy_price > my_energy_bid_price * 1.1: # 价格高出10%
             cost_adjustment_factor = 1.0 # 倾向于认为对手成本高
             logging.debug("  信念更新信号：市场价格较高且我中标，倾向于认为对手成本较高。")
        # 场景2: 市场价格低于我的报价，导致未中标或部分中标 (假设报价是理性的，高于成本)
        elif my_cleared_energy < my_energy_bid_quantity * 0.9 and energy_price < my_energy_bid_price: # 中标率低于90%且价格低于报价
             cost_adjustment_factor = -1.0 # 倾向于认为对手成本低
             logging.debug("  信念更新信号：市场价格较低或我未中标，倾向于认为对手成本较低。")
        # 场景3: 市场价格略高于我的报价，且我中标
        elif my_cleared_energy > 0 and my_energy_bid_price <= energy_price <= my_energy_bid_price * 1.1:
             cost_adjustment_factor = 0.2 # 轻微倾向于认为对手成本较高
             logging.debug("  信念更新信号：市场价格略高于报价且中标，轻微倾向对手成本较高。")
        # 场景4: 其他情况 (例如价格等于报价，或中标情况不明确)
        else:
             cost_adjustment_factor = 0.0 # 保持不变或微调
             logging.debug("  信念更新信号：市场结果不明确，信念调整幅度小。")

        # --- 应用更新 ---
        # 调整均值估计
        old_mean = self.estimated_competitor_avg_cost_mean
        self.estimated_competitor_avg_cost_mean += cost_adjustment_factor * self.estimated_competitor_avg_cost_std * self.learning_rate
        # 防止成本估计变为负数
        self.estimated_competitor_avg_cost_mean = max(5.0, self.estimated_competitor_avg_cost_mean) # 假设成本至少为5

        # 调整标准差估计 (例如，如果观察到的价格波动大，可以增加std)
        # 简化：暂时不调整标准差
        # price_volatility_signal = abs(energy_price - old_mean) / old_mean if old_mean > 0 else 0
        # if price_volatility_signal > 0.3: # 如果价格偏离均值较大
        #    self.estimated_competitor_avg_cost_std *= (1 + 0.05 * self.learning_rate) # 略微增加不确定性
        # else:
        #    self.estimated_competitor_avg_cost_std *= (1 - 0.05 * self.learning_rate) # 略微减少不确定性
        # self.estimated_competitor_avg_cost_std = max(5.0, self.estimated_competitor_avg_cost_std) # 保持最小标准差

        logging.info(f"智能体 {self.agent_id} 更新后信念：估计竞争对手平均成本均值从 {old_mean:.2f} 调整为 {self.estimated_competitor_avg_cost_mean:.2f}")

        # --- 更新文本摘要 ---
        # 这里可以用更复杂的方式生成摘要，甚至调用LLM辅助
        self.belief_summary_text = f"基于第{round_number}轮结果(能源价:{energy_price:.1f})，更新估计：竞争对手平均成本可能在 {max(0, self.estimated_competitor_avg_cost_mean - self.estimated_competitor_avg_cost_std):.1f} 到 {self.estimated_competitor_avg_cost_mean + self.estimated_competitor_avg_cost_std:.1f} 范围内。"

        # --- (可选) 调用LLM辅助更新信念 ---
        # if self.llm_client:
        #     try:
        #         prompt = format_belief_update_prompt(...)
        #         llm_summary = self.llm_client.chat(prompt, return_json=False, model='gemini-2.0-flash') # 或适合的模型
        #         if not llm_summary.startswith("Error:"):
        #             self.belief_summary_text = llm_summary # 使用LLM生成的摘要
        #             logging.info(f"智能体 {self.agent_id} 使用LLM更新了信念摘要: {llm_summary}")
        #         else:
        #             logging.warning(f"智能体 {self.agent_id} LLM辅助信念更新失败: {llm_summary}")
        #     except Exception as e:
        #         logging.error(f"智能体 {self.agent_id} 调用LLM进行信念更新时出错: {e}", exc_info=True)


    def get_estimated_competitor_costs(self) -> Dict:
        """返回当前对竞争对手成本的数值估计"""
        return {
            "mean": self.estimated_competitor_avg_cost_mean,
            "std": self.estimated_competitor_avg_cost_std
        }