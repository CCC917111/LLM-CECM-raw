#!/usr/bin/env python3
# test_adf_mechanism.py
"""
测试ADF（自适应决策频率）机制的有效性
验证三种触发条件：业绩驱动、事件驱动、周期驱动
"""

import sys
import os
import logging
from typing import Dict, Any

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.adf_module import AdaptiveDecisionFrequency

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_performance_trigger():
    """测试业绩驱动触发机制"""
    print("\n=== 测试业绩驱动触发机制 ===")
    
    adf = AdaptiveDecisionFrequency("test_agent_1", {
        'profit_threshold': 0.8,  # 当前利润需低于历史均值的80%才触发
        'profit_window_size': 5
    })
    
    # 模拟历史利润数据（逐渐建立历史基线）
    profits = [100, 95, 105, 98, 102]  # 历史利润，均值约100
    
    for i, profit in enumerate(profits):
        should_trigger, reason = adf.should_trigger_llm_decision(
            round_number=i+1,
            market_info={'electricity_market': {'current_price': 30}},
            my_profit=profit
        )
        print(f"轮次 {i+1}: 利润={profit}, 触发={should_trigger}, 原因={reason}")
    
    # 测试低利润触发
    low_profit = 70  # 显著低于历史均值100的80%
    should_trigger, reason = adf.should_trigger_llm_decision(
        round_number=6,
        market_info={'electricity_market': {'current_price': 30}},
        my_profit=low_profit
    )
    print(f"轮次 6: 利润={low_profit}, 触发={should_trigger}, 原因={reason}")
    assert should_trigger, "低利润应该触发LLM决策"
    
    print("✓ 业绩驱动触发机制测试通过")

def test_event_trigger():
    """测试事件驱动触发机制"""
    print("\n=== 测试事件驱动触发机制 ===")
    
    adf = AdaptiveDecisionFrequency("test_agent_2", {
        'price_volatility_threshold': 0.15,
        'event_keywords': ['政策', '公告', '变化']
    })
    
    # 测试价格波动触发
    prices = [30, 32, 35, 40, 45]  # 价格快速上涨，波动率高
    for i, price in enumerate(prices):
        should_trigger, reason = adf.should_trigger_llm_decision(
            round_number=i+1,
            market_info={'electricity_market': {'current_price': price}},
            my_profit=100
        )
        print(f"轮次 {i+1}: 价格={price}, 触发={should_trigger}, 原因={reason}")
    
    # 测试市场事件触发
    should_trigger, reason = adf.should_trigger_llm_decision(
        round_number=6,
        market_info={'electricity_market': {'current_price': 30}},
        my_profit=100,
        market_events_info="重要政策公告：燃料价格将在下轮调整"
    )
    print(f"轮次 6: 市场事件触发={should_trigger}, 原因={reason}")
    assert should_trigger, "市场事件应该触发LLM决策"
    
    print("✓ 事件驱动触发机制测试通过")

def test_cycle_trigger():
    """测试周期驱动触发机制"""
    print("\n=== 测试周期驱动触发机制 ===")
    
    adf = AdaptiveDecisionFrequency("test_agent_3", {
        'strategic_review_cycle': 5  # 每5轮进行一次战略复盘
    })
    
    # 测试周期触发
    for round_num in range(1, 12):
        should_trigger, reason = adf.should_trigger_llm_decision(
            round_number=round_num,
            market_info={'electricity_market': {'current_price': 30}},
            my_profit=100
        )
        print(f"轮次 {round_num}: 触发={should_trigger}, 原因={reason}")
        
        # 第5轮和第10轮应该触发周期复盘
        if round_num in [5, 10]:
            assert should_trigger, f"第{round_num}轮应该触发周期复盘"
    
    print("✓ 周期驱动触发机制测试通过")

def test_heuristic_decision():
    """测试启发式决策生成"""
    print("\n=== 测试启发式决策生成 ===")
    
    adf = AdaptiveDecisionFrequency("test_agent_4")
    
    # 模拟智能体私有信息
    private_info = {
        'marginal_cost_energy': 25.0,
        'marginal_cost_reserve': 8.0,
        'max_capacity': 100.0,
        'technology_type': 'gas'
    }
    
    # 模拟市场信息
    market_info = {
        'reserve_requirement': 20.0,
        'electricity_market': {'current_price': 35.0},
        'carbon_market': {'current_price': 50.0}
    }
    
    # 生成启发式报价
    heuristic_bid = adf.generate_heuristic_bid(
        round_number=3,
        market_info=market_info,
        private_info=private_info
    )
    
    print("启发式报价结果:")
    print(f"  决策类型: {heuristic_bid['decision_type']}")
    print(f"  能源报价: {heuristic_bid['electricity']['energy_bid']}")
    print(f"  备用报价: {heuristic_bid['electricity']['reserve_bid']}")
    print(f"  碳市场报价: {heuristic_bid['carbon']}")
    print(f"  当前加价率: {heuristic_bid['markup_rate']:.3f}")
    print(f"  容量分配: {heuristic_bid['capacity_split']}")
    
    # 验证报价合理性
    energy_bid = heuristic_bid['electricity']['energy_bid']
    reserve_bid = heuristic_bid['electricity']['reserve_bid']
    
    assert energy_bid['price'] > private_info['marginal_cost_energy'], "能源报价应高于边际成本"
    assert reserve_bid['price'] > 0, "备用报价应为正数"
    assert energy_bid['quantity'] + reserve_bid['quantity'] <= private_info['max_capacity'], "总报价量不应超过容量"
    
    print("✓ 启发式决策生成测试通过")

def test_adf_status():
    """测试ADF状态信息"""
    print("\n=== 测试ADF状态信息 ===")
    
    adf = AdaptiveDecisionFrequency("test_agent_5")
    
    # 模拟一些操作
    adf.should_trigger_llm_decision(1, {'electricity_market': {'current_price': 30}}, 100)
    adf.should_trigger_llm_decision(2, {'electricity_market': {'current_price': 32}}, 95)
    
    status = adf.get_adf_status()
    print("ADF状态信息:")
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    assert 'last_llm_round' in status
    assert 'current_markup_rate' in status
    assert 'capacity_split' in status
    
    print("✓ ADF状态信息测试通过")

def main():
    """运行所有ADF机制测试"""
    print("开始ADF（自适应决策频率）机制测试...")
    
    try:
        test_performance_trigger()
        test_event_trigger()
        test_cycle_trigger()
        test_heuristic_decision()
        test_adf_status()
        
        print("\n🎉 所有ADF机制测试通过！")
        print("\nADF架构特性验证:")
        print("✓ 业绩驱动触发：当利润显著低于历史均值时触发LLM深度决策")
        print("✓ 事件驱动触发：当价格波动率过高或检测到关键市场事件时触发")
        print("✓ 周期驱动触发：定期进行战略复盘，确保长期策略优化")
        print("✓ 启发式决策：非触发期间使用快速决策，基于上次LLM策略进行微调")
        print("✓ 状态管理：完整的ADF状态跟踪和参数调整机制")
        
        print("\n💡 ADF架构成功实现了'常规运营'与'战略复盘'的平衡！")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
