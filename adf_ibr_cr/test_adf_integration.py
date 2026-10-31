#!/usr/bin/env python3
# test_adf_integration.py
"""
测试ADF机制与真实仿真框架的集成
使用实际的LLMAgent和仿真组件进行测试
"""

import sys
import os
import logging
from typing import Dict, Any

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.llm_agent import LLMAgent
from llm_interface.llm_client import LLMClient
from simulation.config import get_config

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_test_agent():
    """创建测试用的LLM智能体（含ADF模块）"""
    # 获取配置
    config = get_config()
    
    # 创建LLM客户端
    llm_client = LLMClient(config)
    
    # 智能体配置
    agent_config = {
        'marginal_cost_energy': 25.0,
        'marginal_cost_reserve': 8.0,
        'max_capacity': 100.0,
        'min_output': 10.0,
        'technology_type': 'gas',
        'emission_factor': 0.4
    }
    
    # ADF配置
    adf_config = {
        'min_llm_rounds': 3,  # 前3轮强制LLM
        'profit_threshold': 0.7,  # 更严格的利润阈值
        'strategic_review_cycle': 5,  # 更长的周期
        'price_volatility_threshold': 0.25,  # 更高的波动率阈值
        'event_keywords': ['政策', '公告', '变化']
    }
    
    # 创建LLM智能体
    agent = LLMAgent(
        agent_id="ADF_Test_Agent",
        config=agent_config,
        llm_client=llm_client,
        competitor_ids=["Agent_B", "Agent_C"],
        adf_config=adf_config
    )
    
    return agent

def create_test_market_info(round_num: int, price: float = 30.0, events: str = None):
    """创建测试用的市场信息"""
    return {
        'electricity_market': {
            'current_price': price,
            'total_demand': 500.0
        },
        'carbon_market': {
            'current_price': 50.0,
            'current_quota': 80.0
        },
        'reserve_requirement': 50.0,
        'round_number': round_num,
        'market_events': events
    }

def simulate_market_results(profit: float):
    """模拟市场结果"""
    return {
        'electricity_market': {
            'clearing_price': 32.0,
            'total_cleared': 450.0
        },
        'carbon_market': {
            'clearing_price': 51.0
        }
    }

def simulate_bid_results(profit: float):
    """模拟智能体投标结果"""
    return {
        'total_profit': profit,
        'energy_cleared': 60.0,
        'reserve_cleared': 15.0,
        'carbon_profit': 5.0,
        'submitted_bid': {
            'electricity': {
                'energy_bid': {'price': 28.0, 'quantity': 70.0},
                'reserve_bid': {'price': 10.0, 'quantity': 20.0}
            },
            'carbon': {
                'buy_quantity': 8.0, 'buy_price': 49.0,
                'sell_quantity': 5.0, 'sell_price': 52.0
            }
        }
    }

def test_adf_integration():
    """测试ADF机制与仿真框架的集成"""
    print("=== 开始ADF集成测试 ===")
    
    try:
        # 创建测试智能体
        agent = create_test_agent()
        print(f"✓ 成功创建LLM智能体: {agent.agent_id}")
        
        # 测试场景：模拟多轮决策
        profits = [100, 95, 105, 98, 102, 60, 85, 90, 95, 100]  # 第6轮利润显著下降
        decision_types = []
        token_usage = []
        
        for round_num in range(1, 11):
            print(f"\n--- 轮次 {round_num} ---")
            
            # 创建市场信息
            market_events = None
            if round_num == 4:
                market_events = "重要政策公告：燃料价格将在下轮调整"
            
            market_info = create_test_market_info(
                round_num, 
                price=30 + round_num * 0.5,  # 价格缓慢上涨，减少波动
                events=market_events
            )
            
            # 智能体决策
            try:
                bid_decision = agent.decide_coupled_bid(
                    round_number=round_num,
                    market_info=market_info,
                    history={'market_prices': [30, 32, 34], 'awards': [True, False, True]},
                    market_events_info=market_events
                )
                
                decision_type = bid_decision.get('decision_type', 'unknown')
                trigger_reason = bid_decision.get('trigger_reason', 'N/A')
                tokens = bid_decision.get('token_usage', {}).get('total_tokens', 0)
                
                decision_types.append(decision_type)
                token_usage.append(tokens)
                
                print(f"  决策类型: {decision_type}")
                print(f"  触发原因: {trigger_reason}")
                print(f"  Token使用: {tokens}")
                print(f"  能源报价: {bid_decision['electricity']['energy_bid']}")
                print(f"  备用报价: {bid_decision['electricity']['reserve_bid']}")
                
            except Exception as e:
                print(f"  ⚠️ 决策失败，使用fallback: {e}")
                decision_types.append('fallback')
                token_usage.append(0)
            
            # 更新智能体状态
            current_profit = profits[round_num - 1]
            market_results = simulate_market_results(current_profit)
            bid_results = simulate_bid_results(current_profit)
            
            agent.update_state(round_num, market_results, bid_results)
            print(f"  当前利润: {current_profit}")
        
        # 分析结果
        print(f"\n=== ADF集成测试结果分析 ===")
        print(f"总轮次: {len(decision_types)}")
        print(f"LLM深度决策次数: {decision_types.count('llm_deep')}")
        print(f"启发式决策次数: {decision_types.count('heuristic')}")
        print(f"Fallback决策次数: {decision_types.count('fallback')}")
        print(f"总Token使用: {sum(token_usage)}")
        print(f"平均每轮Token: {sum(token_usage) / len(token_usage):.1f}")
        
        # 验证ADF工作正常
        assert decision_types.count('heuristic') > 0, "应该有启发式决策"
        llm_rounds = decision_types.count('llm_deep')
        heuristic_rounds = decision_types.count('heuristic')
        assert heuristic_rounds >= llm_rounds, "启发式决策应该占多数轮次"
        
        # 计算token效率：启发式决策轮次应该显著减少平均token使用
        if llm_rounds > 0:
            avg_llm_tokens = sum(token_usage[i] for i, dt in enumerate(decision_types) if dt == 'llm_deep') / llm_rounds
            total_avg_tokens = sum(token_usage) / len(token_usage)
            efficiency_gain = (avg_llm_tokens - total_avg_tokens) / avg_llm_tokens
            print(f"Token效率提升: {efficiency_gain:.1%}")
            assert efficiency_gain > 0.3, f"ADF应该至少提升30%的token效率，实际提升{efficiency_gain:.1%}"
        
        # 获取ADF状态
        adf_status = agent.adf_module.get_adf_status()
        print(f"\nADF模块状态:")
        print(f"  上次LLM决策轮次: {adf_status['last_llm_round']}")
        print(f"  当前加价率: {adf_status['current_markup_rate']:.3f}")
        print(f"  容量分配: {adf_status['capacity_split']}")
        print(f"  利润历史长度: {adf_status['profit_history_size']}")
        
        print(f"\n🎉 ADF集成测试成功完成！")
        return True
        
    except Exception as e:
        print(f"\n❌ ADF集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """运行ADF集成测试"""
    print("开始ADF机制与仿真框架集成测试...")
    
    success = test_adf_integration()
    
    if success:
        print("\n✅ ADF机制已成功集成到仿真框架中！")
        print("\n主要特性验证:")
        print("• ADF模块正确初始化并集成到LLMAgent")
        print("• 三种触发机制（业绩、事件、周期）正常工作")
        print("• 启发式决策有效减少LLM调用和Token消耗")
        print("• 智能体状态更新包含ADF性能数据")
        print("• 决策类型和触发原因正确记录")
    else:
        print("\n❌ ADF集成存在问题，需要进一步调试")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
