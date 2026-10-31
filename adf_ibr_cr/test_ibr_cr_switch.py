#!/usr/bin/env python3
"""
IBR-CR机制开关测试脚本
测试enable_ibr_cr开关是否正常工作
"""

import sys
import logging
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent))

from simulation.config import get_config
from agents.llm_agent import LLMAgent
from llm_interface.llm_client import LLMClient

def test_ibr_cr_switch():
    """测试IBR-CR开关功能"""
    
    # 设置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("="*60)
    print("IBR-CR机制开关测试")
    print("="*60)
    
    # 获取基础配置
    config = get_config()
    llm_client = LLMClient(config.get("llm_client", {}))
    
    # 测试数据
    test_market_info = {
        "demand": 350,
        "reserve_requirement": 35,
        "current_round": 1,
        "carbon_market": {
            "current_price": 50.0,
            "current_quota": 80.0
        }
    }
    
    # 测试1: 启用IBR-CR机制
    print("\n🔬 测试1: 启用IBR-CR机制")
    print("-" * 40)
    
    llm_config_with_ibr = {
        "model": "gemma-3-27b-it",
        "enable_adf": False,  # 禁用ADF以便直接测试IBR-CR
        "enable_ibr_cr": True  # 启用IBR-CR
    }
    
    agent_config = {
        "marginal_cost_energy": 25.0,
        "marginal_cost_reserve": 5.0,
        "max_capacity": 100.0,
        "min_output": 0.0,
        "emission_factor": 0.8
    }
    
    agent_with_ibr = LLMAgent(
        agent_id="TestAgent_IBR",
        config=agent_config,
        llm_client=llm_client,
        competitor_ids=["Competitor1", "Competitor2"],
        llm_config=llm_config_with_ibr
    )
    
    print(f"✅ 智能体创建成功")
    print(f"   - ADF状态: {'启用' if agent_with_ibr.enable_adf else '禁用'}")
    print(f"   - IBR-CR状态: {'启用' if agent_with_ibr.enable_ibr_cr else '禁用'}")
    
    # 测试2: 禁用IBR-CR机制
    print("\n🔬 测试2: 禁用IBR-CR机制")
    print("-" * 40)
    
    llm_config_without_ibr = {
        "model": "gemma-3-27b-it",
        "enable_adf": False,  # 禁用ADF以便直接测试IBR-CR
        "enable_ibr_cr": False  # 禁用IBR-CR
    }
    
    agent_without_ibr = LLMAgent(
        agent_id="TestAgent_NoIBR",
        config=agent_config,
        llm_client=llm_client,
        competitor_ids=["Competitor1", "Competitor2"],
        llm_config=llm_config_without_ibr
    )
    
    print(f"✅ 智能体创建成功")
    print(f"   - ADF状态: {'启用' if agent_without_ibr.enable_adf else '禁用'}")
    print(f"   - IBR-CR状态: {'启用' if agent_without_ibr.enable_ibr_cr else '禁用'}")
    
    # 测试3: 默认配置（应该启用IBR-CR）
    print("\n🔬 测试3: 默认配置")
    print("-" * 40)
    
    agent_default = LLMAgent(
        agent_id="TestAgent_Default",
        config=agent_config,
        llm_client=llm_client,
        competitor_ids=["Competitor1", "Competitor2"],
        llm_config=None  # 使用默认配置
    )
    
    print(f"✅ 智能体创建成功")
    print(f"   - ADF状态: {'启用' if agent_default.enable_adf else '禁用'}")
    print(f"   - IBR-CR状态: {'启用' if agent_default.enable_ibr_cr else '禁用'}")
    
    # 验证结果
    print("\n📊 测试结果验证")
    print("-" * 40)
    
    # 验证IBR-CR开关是否正确设置
    assert agent_with_ibr.enable_ibr_cr == True, "启用IBR-CR的智能体应该enable_ibr_cr=True"
    assert agent_without_ibr.enable_ibr_cr == False, "禁用IBR-CR的智能体应该enable_ibr_cr=False"
    assert agent_default.enable_ibr_cr == True, "默认配置应该启用IBR-CR"
    
    print("✅ 所有测试通过！")
    
    # 展示配置示例
    print("\n📝 配置示例")
    print("-" * 40)
    print("要禁用IBR-CR机制，在配置文件中设置：")
    print("""
    "llm_config": {
        "model": "gemma-3-27b-it",
        "enable_adf": True,      # ADF开关（可选）
        "enable_ibr_cr": False   # 禁用IBR-CR，直接让LLM生成报价
    }
    """)
    
    print("\n🎯 功能说明")
    print("-" * 40)
    print("• enable_ibr_cr=True:  使用IBR-CR机制（我-他-我的三步思考过程）")
    print("• enable_ibr_cr=False: 跳过IBR-CR，直接让LLM生成报价（原始方法）")
    print("• 默认值: True（保持现有行为）")
    
    return True

if __name__ == "__main__":
    try:
        success = test_ibr_cr_switch()
        if success:
            print("\n🎉 IBR-CR开关测试完成！")
            sys.exit(0)
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
