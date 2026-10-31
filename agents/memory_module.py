# llm_market_sim/agents/memory_module.py
import logging
from collections import deque
from typing import List, Dict, Any, Optional
import pandas as pd # 使用Pandas方便数据处理和检索

# (可选) LLM客户端用于辅助经验总结
# from ..llm_interface.llm_client import LLMClient
# from ..llm_interface.prompts import format_experience_summary_prompt

class MemoryModule:
    """
    管理智能体历史经验（记忆）的模块。
    存储过去的报价、市场结果、利润等，并提供检索和总结功能。
    """
    def __init__(self, agent_id: str, capacity: int = 100):
        """
        初始化记忆模块。

        Args:
            agent_id: 当前智能体的ID。
            capacity: 记忆容量（存储多少轮的数据）。
        """
        self.agent_id = agent_id
        self.capacity = capacity
        # 使用deque或列表存储历史记录字典
        # self.history = deque(maxlen=capacity)
        # 使用Pandas DataFrame存储历史记录，方便查询和分析
        self.history_df = pd.DataFrame(columns=[
            'round', 'energy_bid_price', 'energy_bid_quantity',
            'reserve_bid_price', 'reserve_bid_quantity',
            'cleared_energy', 'cleared_reserve',
            'market_energy_price', 'market_reserve_price', 'profit',
            'belief_mean_at_decision', 'belief_std_at_decision' # 记录决策时的信念状态
        ])
        self.last_summary_round = -1 # 记录上次生成摘要的轮次
        self.experience_summary_text = "尚未有足够的历史经验可供总结。"

        # (可选) LLM客户端用于辅助总结
        # self.llm_client = LLMClient(...)

        logging.info(f"智能体 {agent_id} 的记忆模块初始化完成，容量为 {capacity} 轮。")

    def add_record(self, round_number: int, bid: Dict, result: Dict, belief_state: Dict):
        """
        向记忆中添加一条新的记录。

        Args:
            round_number: 当前轮次。
            bid: 智能体提交的报价 {'energy_bid': {...}, 'reserve_bid': {...}}。
            result: 市场出清后该智能体的结果 {'cleared_energy': ..., 'cleared_reserve': ..., 'profit': ...}
                     以及市场价格 {'market_energy_price': ..., 'market_reserve_price': ...}。
            belief_state: 做出该报价决策时的信念状态 {'mean': ..., 'std': ...}。
        """
        if len(self.history_df) >= self.capacity:
             # 如果达到容量上限，移除最早的记录 (Pandas会自动处理索引，但我们手动删除第一行更明确)
             self.history_df = self.history_df.iloc[1:].reset_index(drop=True)

        new_record = {
            'round': round_number,
            'energy_bid_price': bid.get('energy_bid', {}).get('price'),
            'energy_bid_quantity': bid.get('energy_bid', {}).get('quantity'),
            'reserve_bid_price': bid.get('reserve_bid', {}).get('price'),
            'reserve_bid_quantity': bid.get('reserve_bid', {}).get('quantity'),
            'cleared_energy': result.get('cleared_energy'),
            'cleared_reserve': result.get('cleared_reserve'),
            'market_energy_price': result.get('market_energy_price'),
            'market_reserve_price': result.get('market_reserve_price'),
            'profit': result.get('profit'),
            'belief_mean_at_decision': belief_state.get('mean'),
            'belief_std_at_decision': belief_state.get('std')
        }
        # 使用 concat 添加新行 (注意 ignore_index=True)
        new_record_df = pd.DataFrame([new_record])
        self.history_df = pd.concat([self.history_df, new_record_df], ignore_index=True)

        logging.debug(f"智能体 {self.agent_id} 记忆模块：添加了第 {round_number} 轮的记录。当前记录数: {len(self.history_df)}")

    def get_recent_history(self, n: int = 5) -> List[Dict]:
        """获取最近 n 条历史记录 (字典列表形式)"""
        if self.history_df.empty:
            return []
        return self.history_df.tail(n).to_dict('records')

    def get_relevant_history_snippet(self, n_recent: int = 5, n_extreme: int = 2) -> str:
        """
        获取用于Prompt的相关历史经验片段（文本格式）。
        包括最近的几轮和利润最高/最低的几轮。
        """
        if self.history_df.empty:
            return "无历史记录。"

        snippet = "最近经验:\n"
        recent_df = self.history_df.tail(n_recent)
        if not recent_df.empty:
             snippet += recent_df.to_string(index=False, float_format="%.2f") + "\n"
        else:
             snippet += "无最近记录。\n"


        if len(self.history_df) > n_recent: # 只有当历史记录足够多时才找极值
            snippet += "\n特殊经验 (高/低利润轮次):\n"
            sorted_df = self.history_df.sort_values(by='profit', ascending=False)
            # 选择最高和最低利润的记录，避免重复显示最近的记录
            top_df = sorted_df.head(n_extreme)
            bottom_df = sorted_df.tail(n_extreme)
            # 过滤掉可能已包含在 recent_df 中的记录
            extreme_indices = list(top_df.index) + list(bottom_df.index)
            recent_indices = list(recent_df.index)
            unique_extreme_indices = [idx for idx in extreme_indices if idx not in recent_indices]

            if unique_extreme_indices:
                 extreme_df = self.history_df.loc[unique_extreme_indices]
                 # 按轮次排序，使得显示更有条理
                 extreme_df_sorted = extreme_df.sort_values(by='round')
                 snippet += extreme_df_sorted.to_string(index=False, header=False, float_format="%.2f") + "\n" # 不显示表头
            else:
                 snippet += "无额外的特殊记录。\n"


        return snippet.strip()


    def generate_experience_summary(self, current_round: int, force_update: bool = False) -> str:
        """
        生成或更新经验总结文本。
        可以基于规则生成，或调用LLM辅助生成。
        避免频繁生成以节省资源。
        """
        # 每隔一定轮次或强制更新时才生成总结
        rounds_since_last_summary = current_round - self.last_summary_round
        if not force_update and rounds_since_last_summary < 10 and self.last_summary_round != -1: # 每10轮更新一次
             return self.experience_summary_text

        if len(self.history_df) < 5: # 需要至少几轮数据才有意义
             return "历史数据不足，无法生成有效总结。"

        logging.debug(f"智能体 {self.agent_id} 正在生成经验总结 (轮次 {current_round})...")

        # --- 基于规则的简单总结 (示例) ---
        try:
            recent_history = self.history_df.tail(10) # 分析最近10轮
            avg_profit = recent_history['profit'].mean()
            max_profit_round = recent_history.loc[recent_history['profit'].idxmax()] if not recent_history.empty else None
            min_profit_round = recent_history.loc[recent_history['profit'].idxmin()] if not recent_history.empty else None

            summary = f"过去{len(recent_history)}轮经验总结 (截至第{current_round}轮): "
            summary += f"平均利润约为 {avg_profit:.2f}。"
            if max_profit_round is not None:
                summary += f" 利润最高在第 {int(max_profit_round['round'])} 轮({max_profit_round['profit']:.2f})，当时报价(能/备): {max_profit_round['energy_bid_price']:.1f}/{max_profit_round['reserve_bid_price']:.1f}。"
            if min_profit_round is not None and min_profit_round['profit'] < avg_profit:
                 summary += f" 利润最低在第 {int(min_profit_round['round'])} 轮({min_profit_round['profit']:.2f})，可能原因需分析报价与市场价关系。"

            # 简单策略建议
            high_price_rounds = recent_history[recent_history['market_energy_price'] > recent_history['energy_bid_price'] * 1.1]
            if not high_price_rounds.empty and high_price_rounds['cleared_energy'].mean() > 0.5 * high_price_rounds['energy_bid_quantity'].mean(): # 价格高且中标率尚可
                summary += " 观察到在高市场价时提高报价仍可能盈利。"
            low_bid_win_rounds = recent_history[recent_history['cleared_energy'] > 0.9 * recent_history['energy_bid_quantity']]
            if not low_bid_win_rounds.empty and low_bid_win_rounds['profit'].mean() < avg_profit * 0.8: # 中标但利润不高
                summary += " 注意避免报价过低导致利润微薄。"

            self.experience_summary_text = summary
            self.last_summary_round = current_round
            logging.info(f"智能体 {self.agent_id} 更新了基于规则的经验总结。")

        except Exception as e:
            logging.error(f"智能体 {self.agent_id} 生成规则化经验总结时出错: {e}", exc_info=True)
            # 保留旧的总结
            return self.experience_summary_text


        # --- (可选) 调用LLM辅助生成总结 ---
        # if self.llm_client:
        #     try:
        #         historical_snippet = self.get_relevant_history_snippet(n_recent=10, n_extreme=3)
        #         if "无历史记录" in historical_snippet or len(self.history_df) < 5:
        #              return self.experience_summary_text # 数据不足

        #         prompt = format_experience_summary_prompt(
        #             agent_id=self.agent_id,
        #             historical_data_snippet=historical_snippet,
        #             num_rounds_to_summarize=len(self.history_df) # 或者只总结最近N轮
        #         )
        #         llm_summary = self.llm_client.chat(prompt, return_json=False, model='gemini-2.0-flash') # 或适合的模型

        #         if not llm_summary.startswith("Error:"):
        #             self.experience_summary_text = llm_summary
        #             self.last_summary_round = current_round
        #             logging.info(f"智能体 {self.agent_id} 使用LLM更新了经验总结: {llm_summary[:100]}...")
        #         else:
        #             logging.warning(f"智能体 {self.agent_id} LLM辅助经验总结失败: {llm_summary}")
        #     except Exception as e:
        #         logging.error(f"智能体 {self.agent_id} 调用LLM进行经验总结时出错: {e}", exc_info=True)


        return self.experience_summary_text


    def get_experience_summary(self, current_round: int) -> str:
        """获取最新的经验总结 (可能会触发生成)"""
        # 调用生成函数，它内部会判断是否需要更新
        return self.generate_experience_summary(current_round) 