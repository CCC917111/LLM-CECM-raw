#!/usr/bin/env python3
"""
测试ADF开关功能
验证智能体可以通过配置启用或禁用ADF机制
"""

import logging
from agents.llm_agent import LLMAgent
from llm_interface.llm_client import LLMClient

def test_adf_switch():
    """测试ADF开关功能"""
    
    # 设置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 初始化LLM客户端
    llm_client = LLMClient()
    
    # 测试配置
    agent_config = {
        "marginal_cost_energy": 25.0,
        "marginal_cost_reserve": 5.0,
        "max_capacity": 100.0,
        "min_output": 0.0,
        "emission_factor": 0.8
    }
    
    competitor_ids = ["Agent_B", "Agent_C"]
    
    print("=== ADF开关功能测试 ===\n")
    
    # 测试1：启用ADF
    print("🔧 测试1: 启用ADF机制")
    llm_config_enabled = {"model": "gemma-3-27b-it", "enable_adf": True}
    
    agent_adf_enabled = LLMAgent(
        agent_id="Test_Agent_ADF_ON",
        config=agent_config,
        llm_client=llm_client,
        competitor_ids=competitor_ids,
        llm_config=llm_config_enabled
    )
    
    print(f"  智能体 {agent_adf_enabled.agent_id}:")
    print(f"  - ADF启用状态: {agent_adf_enabled.enable_adf}")
    print(f"  - ADF模块存在: {agent_adf_enabled.adf_module is not None}")
    
    # 测试2：禁用ADF
    print("\n🔧 测试2: 禁用ADF机制")
    llm_config_disabled = {"model": "gemma-3-27b-it", "enable_adf": False}
    
    agent_adf_disabled = LLMAgent(
        agent_id="Test_Agent_ADF_OFF",
        config=agent_config,
        llm_client=llm_client,
        competitor_ids=competitor_ids,
        llm_config=llm_config_disabled
    )
    
    print(f"  智能体 {agent_adf_disabled.agent_id}:")
    print(f"  - ADF启用状态: {agent_adf_disabled.enable_adf}")
    print(f"  - ADF模块存在: {agent_adf_disabled.adf_module is not None}")
    
    # 测试3：默认配置（应该启用ADF）
    print("\n🔧 测试3: 默认配置（无explicit设置）")
    llm_config_default = {"model": "gemma-3-27b-it"}
    
    agent_default = LLMAgent(
        agent_id="Test_Agent_DEFAULT",
        config=agent_config,
        llm_client=llm_client,
        competitor_ids=competitor_ids,
        llm_config=llm_config_default
    )
    
    print(f"  智能体 {agent_default.agent_id}:")
    print(f"  - ADF启用状态: {agent_default.enable_adf}")
    print(f"  - ADF模块存在: {agent_default.adf_module is not None}")
    
    # 测试4：无llm_config参数（应该启用ADF）
    print("\n🔧 测试4: 无llm_config参数")
    
    agent_no_config = LLMAgent(
        agent_id="Test_Agent_NO_CONFIG",
        config=agent_config,
        llm_client=llm_client,
        competitor_ids=competitor_ids
    )
    
    print(f"  智能体 {agent_no_config.agent_id}:")
    print(f"  - ADF启用状态: {agent_no_config.enable_adf}")
    print(f"  - ADF模块存在: {agent_no_config.adf_module is not None}")
    
    # 验证结果
    print("\n=== 测试结果验证 ===")
    
    # 验证启用ADF的智能体
    assert agent_adf_enabled.enable_adf == True, "ADF应该被启用"
    assert agent_adf_enabled.adf_module is not None, "ADF模块应该存在"
    
    # 验证禁用ADF的智能体
    assert agent_adf_disabled.enable_adf == False, "ADF应该被禁用"
    assert agent_adf_disabled.adf_module is None, "ADF模块应该不存在"
    
    # 验证默认配置
    assert agent_default.enable_adf == True, "默认应该启用ADF"
    assert agent_default.adf_module is not None, "默认应该有ADF模块"
    
    # 验证无配置
    assert agent_no_config.enable_adf == True, "无配置时默认应该启用ADF"
    assert agent_no_config.adf_module is not None, "无配置时应该有ADF模块"
    
    print("✅ 所有测试通过！")
    print("\n📋 ADF开关功能总结:")
    print("  - enable_adf=True: 启用ADF机制，使用混合决策")
    print("  - enable_adf=False: 禁用ADF机制，始终使用LLM深度决策")
    print("  - 默认值: True (启用ADF)")
    print("  - 配置位置: llm_config.enable_adf")

if __name__ == "__main__":
    test_adf_switch()
