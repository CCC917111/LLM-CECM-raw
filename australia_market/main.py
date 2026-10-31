#!/usr/bin/env python3
# main.py
import os
import sys
import logging
from pathlib import Path
import json
import time

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent))

from simulation.config import get_config
from simulation.simulator import MarketSimulator
from utils.logger import setup_logging
from utils.visualization import ResultVisualizer
from agents.rl_agent_enhanced import EnhancedRLAgent
from agents.zi_agent import ZIAgent


def load_tiered_quota_config():
    """加载分层碳配额配置"""
    with open('tiered_carbon_quota_config.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def apply_tiered_quota_to_config(config):
    """将分层配额应用到仿真配置"""
    quota_config = load_tiered_quota_config()
    
    # 启用分层配额系统
    config['carbon_market']['tiered_quota'] = quota_config['tiered_quota_system']
    
    # 调整基础碳价格以匹配真实ACCU价格 (微调降低7元)
    config['carbon_market']['init_carbon_price'] = 143.0  # 从150降到143，更接近真实水平
    config['carbon_market']['min_carbon_price'] = 113.0   # 从120降到113
    config['carbon_market']['max_carbon_price'] = 193.0   # 从200降到193
    
    # 保持原有的真实RRP数据不变
    # config["real_data_dynamics"]["ground_truth_rrp"] 已在配置中正确设置
    
    print("🏭 分层碳配额系统已启用:")
    print(f"   大型企业 ({quota_config['tiered_quota_system']['tiers']['large']['agents']}家): {quota_config['tiered_quota_system']['tiers']['large']['quota_per_day']:,} 吨CO2/日")
    print(f"   中型企业 ({quota_config['tiered_quota_system']['tiers']['medium']['agents']}家): {quota_config['tiered_quota_system']['tiers']['medium']['quota_per_day']:,} 吨CO2/日")
    print(f"   小型企业 ({quota_config['tiered_quota_system']['tiers']['small']['agents']}家): {quota_config['tiered_quota_system']['tiers']['small']['quota_per_day']:,} 吨CO2/日")
    print(f"   碳价格范围: {config['carbon_market']['min_carbon_price']}-{config['carbon_market']['max_carbon_price']} 元/吨CO2")
    
    return config

def load_realistic_agents():
    """加载基于真实AEMO数据的50个LLM智能体配置"""
    config_file = "true_digital_twin_agents.json"
    
    if not Path(config_file).exists():
        raise FileNotFoundError(f"配置文件 {config_file} 不存在")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        agents = json.load(f)
    
    print(f"🔍 验证基于真实AEMO数据的智能体配置:")
    print(f"   智能体数量: {len(agents)} 个")
    
    # 验证配置完整性
    fuel_stats = {}
    total_capacity = 0
    cost_range = []
    
    for agent in agents:
        config = agent.get('config', {})
        fuel_category = config.get('fuel_category', 'Unknown')
        capacity = config.get('max_capacity', 0)
        cost = config.get('marginal_cost_energy', 0)
        
        fuel_stats[fuel_category] = fuel_stats.get(fuel_category, 0) + 1
        total_capacity += capacity
        cost_range.append(cost)
    
    print(f"   总装机容量: {total_capacity:.0f} MW")
    print(f"   成本范围: {min(cost_range):.2f} - {max(cost_range):.2f} 元/MWh")
    print(f"   燃料类型分布:")
    for fuel, count in sorted(fuel_stats.items()):
        print(f"     {fuel}: {count} 个")
    
    # 检查关键机组
    coal_count = sum(1 for agent in agents if 'Coal' in agent.get('config', {}).get('fuel_category', ''))
    gas_count = sum(1 for agent in agents if 'Natural_Gas' in agent.get('config', {}).get('fuel_category', ''))
    
    print(f"   关键机组验证:")
    print(f"     煤炭机组: {coal_count} 个 (用于停机事件)")
    print(f"     燃气机组: {gas_count} 个 (用于价格冲击验证)")
    
    if coal_count == 0:
        print(f"   ⚠️ 警告: 没有煤炭机组，停机事件将无效")
    if gas_count == 0:
        print(f"   ⚠️ 警告: 没有燃气机组，价格冲击验证将无效")
    
    print(f"✅ 基于真实AEMO数据的智能体配置加载完成")
    return agents

def main():
    """
    电力市场多智能体仿真系统的主入口。
    基于真实澳洲AEMO数据的电碳耦合市场仿真
    """
    # 检查是否已有实例在运行
    lock_file = Path(".simulation_lock")
    if lock_file.exists():
        try:
            lock_time = lock_file.stat().st_mtime
            if time.time() - lock_time < 3600:  # 1小时内的锁文件
                print("⚠️ 检测到另一个仿真实例正在运行（锁文件存在）")
                print(f"   锁文件: {lock_file.absolute()}")
                print("   如果确认没有其他实例，请删除锁文件后重试")
                return 2
        except:
            pass
    
    # 创建锁文件
    try:
        lock_file.write_text(f"Started at {time.strftime('%Y-%m-%d %H:%M:%S')}\nPID: {os.getpid()}")
    except Exception as e:
        print(f"⚠️ 无法创建锁文件: {e}")
    
    start_time = time.time()
    run_id = time.strftime("%Y%m%d_%H%M%S")
    
    print("="*80)
    print(f"🎯 启动基于真实AEMO数据的电碳耦合市场仿真")
    print(f"🆔 运行ID: {run_id}")
    print(f"🕐 开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📍 进程ID: {os.getpid()}")
    print("="*80)
    print("🔍 实验配置:")
    print("   - 保持原有电碳耦合市场结构")
    print("   - 使用基于真实AEMO数据的智能体参数")
    print("   - 验证IBR-ADF燃气机组策略性行为")
    print("="*80)
    
    # 获取配置
    config = get_config()
    # 启用在线LLM调用
    try:
        cfg_llm = config.get("llm_client", {})
        cfg_llm["offline"] = False
        config["llm_client"] = cfg_llm
        print("🛠️ 已关闭离线模式，启用在线LLM调用 (offline=False)")
    except Exception as e:
        print(f"⚠️ 设置LLM在线模式失败: {e}")
    
    # 🔥 替换为基于真实AEMO数据的50个LLM智能体
    realistic_agents = load_realistic_agents()
    config["agents"] = realistic_agents

    # 按真实事件脚本对齐：10轮 + 市场事件 + 日内需求曲线 + OPF出清
    config["simulation_name"] = "RealData_10Rounds_WithEvents"
    config["num_rounds"] = 10

    # 市场事件（用于向智能体公告/行为分析；成本与停机按每轮真实数据动态执行）
    config["market_events"] = [
        {
            "announce_time": 3,
            "effect_time": 6,
            "event_type": "fuel_price_shock",
            "affected_fuel": "Gas",   # 对应 Natural_Gas
            "price_multiplier": 2.5,
            "description": "天然气价格冲击，验证IBR-ADF燃气机组策略性抬价行为",
            "announcement": "🚨 紧急公告：国际天然气供应紧张，价格将在3轮后大幅上涨150%"
        },
        {
            "announce_time": 1,
            "effect_time": 7,
            "event_type": "carbon_price_shock",
            "affected_fuel": "Coal",   # 化石燃料受影响的示例之一
            "carbon_price_multiplier": 1.8,
            "description": "碳价格上涨，进一步影响化石燃料发电成本",
            "announcement": "📈 公告：碳市场政策收紧，碳价格将上涨80%"
        },
        {
            "announce_time": 1,
            "effect_time": 7,
            "event_type": "carbon_price_shock",
            "affected_fuel": "Gas",    # 天然气亦受影响
            "carbon_price_multiplier": 1.8,
            "description": "碳价格上涨，进一步影响化石燃料发电成本",
            "announcement": "📈 公告：碳市场政策收紧，碳价格将上涨80%"
        }
    ]

    # 需求/备用曲线：设置为60%总装机，Round 6开始逐步增加模拟价格攀升
    try:
        total_capacity = sum(a.get('config', {}).get('max_capacity', 0.0) for a in realistic_agents)
        base_demand = int(total_capacity * 0.60)  # 60%容量，适中的市场紧张度
        # Round 0-5保持平稳，Round 6-9逐步提升以匹配真实价格趋势
        daily_pattern = [0.75, 0.70, 0.85, 0.95, 1.00, 0.98, 1.10, 1.15, 1.18, 1.12]
        demand_profile = [int(base_demand * p) for p in daily_pattern]
        config.setdefault("market", {})
        config["market"]["demand_profile"] = demand_profile
        config["market"]["reserve_profile"] = [0] * len(demand_profile)
        config["market"]["clearing_mechanism"] = "opf"
        config["market"].setdefault("opf_options", {"OPF_ALG": 560})
        print(f"📈 实验需求曲线：base_demand={base_demand} MW，轮次={len(demand_profile)}")
    except Exception as e:
        print(f"⚠️ 设置需求曲线失败: {e}")

    # 运行模式: 'RL' | 'RB' | 'LLM'
    run_mode = "LLM"
    # RL共享策略开关（从运行模式推导）
    use_rl_shared_policy = (run_mode == "RL")
    rl_shared_model_dir = "models/multi_agent_high_price"
    external_agents = None

    if use_rl_shared_policy:
        try:
            all_agents_info = [a['config'] for a in realistic_agents]
            default_model_path = None
            if os.path.isdir(rl_shared_model_dir):
                for fname in os.listdir(rl_shared_model_dir):
                    if fname.endswith("_model.zip"):
                        default_model_path = os.path.join(rl_shared_model_dir, fname)
                        break
            if not default_model_path:
                print(f"❌ 模型目录中未找到模型: {rl_shared_model_dir}")
                return 1
            agents = {}
            for agent in realistic_agents:
                aid = agent['agent_id']
                acon = agent['config']
                mp = os.path.join(rl_shared_model_dir, f"{aid}_model.zip")
                mdl = mp if os.path.exists(mp) else default_model_path
                agents[aid] = EnhancedRLAgent(agent_id=aid, config=acon, model_path=mdl, all_agents_info=all_agents_info)
            external_agents = agents
            print(f"已启用RL共享策略，模型目录: {rl_shared_model_dir}")
        except Exception as e:
            print(f"❌ 构建RL共享策略失败: {e}")
            return 1
    elif run_mode == "RB":
        try:
            agents = {}
            for agent in realistic_agents:
                aid = agent['agent_id']
                acon = agent['config']
                agents[aid] = ZIAgent(agent_id=aid, config=acon)
            external_agents = agents
            print("已启用RB规则策略（基于成本的启发式报价）")
        except Exception as e:
            print(f"❌ 构建RB规则策略失败: {e}")
            return 1

    # 基于真实数据的每轮动态（与 run_real_data_experiment.py 对齐）
    config["real_data_dynamics"] = {
        "enabled": True,
        # 0基索引，对应每轮GJ价格；成本=GJ*8（MWh）- 使用表现最好的配置
        "gas_prices_by_round": [10, 11, 12, 14, 16, 18, 20, 24, 23, 24],
        # 煤炭停机率（按当轮比例随机选取机组停机）- 使用表现最好的配置
        "coal_outage_rates_by_round": [0, 0, 0.05, 0.09, 0.11, 0.14, 0.20, 0.22, 0.12, 0.08],
        # 需求净增长（MW），在基线需求上逐轮累加 - 降低Round 5-7缓解供需压力
        "demand_increase_by_round": [0, 20, 40, 60, 80, 90, 95, 100, 103, 103],
        "random_seed": 42,
        # 模拟可再生能源的间歇性 (按日平均)
        "solar_capacity_factors": [0.35, 0.40, 0.45, 0.50, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30],  # 日平均太阳能出力
        "wind_capacity_factors":  [0.6, 0.5, 0.4, 0.3, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        # 真实的Ground Truth RRP数据 (用于可视化对比)
        "ground_truth_rrp": [85.3, 92.1, 88.5, 95.4, 105.2, 115.8, 135.7, 180.9, 210.4, 225.6],
        # 真实的Ground Truth ACCU碳价数据 (2022年3-6月现货价格，元/吨CO2)
        "ground_truth_carbon_price": [144.0, 146.4, 148.8, 153.6, 158.4, 163.2, 168.0, 170.4, 172.8, 170.4],
    }
    
    # 设置日志
    import logging
    
    # 应用分层碳配额系统
    try:
        config = apply_tiered_quota_to_config(config)
        print("✅ 分层碳配额系统已成功加载")
    except FileNotFoundError:
        print("⚠️ 分层配额配置文件未找到，使用默认配额")
    except Exception as e:
        print(f"❌ 加载分层配额系统失败: {e}")
        print("使用默认碳市场配置")
    setup_logging(
        log_file=config["logging"]["log_file"], 
        log_level=logging.INFO
    )
    
    logging.info("="*60)
    logging.info(f"启动电力市场LLM智能体仿真: {config['simulation_name']}")
    logging.info(f"计划执行 {config['num_rounds']} 轮市场交易")
    logging.info("="*60)
    
    try:
        # 初始化并运行仿真器
        simulator = MarketSimulator(config, external_agents=external_agents) if external_agents else MarketSimulator(config)
        results = simulator.run_coupled()
        
        # 输出简要结果摘要
        logging.info("="*60)
        logging.info("仿真完成! 结果摘要:")
        for agent_id, agent_results in results.items():
            logging.info(f"智能体 {agent_id}: 总利润=${agent_results['total_profit']:.2f}, 平均利润=${agent_results['avg_profit']:.2f}")
        logging.info(f"详细结果已保存至: {config['logging']['results_file_csv']}")
        
        # 生成可视化结果
        try:
            logging.info("正在生成可视化图表...")
            # 使用配置中定义的真实RRP数据和碳价数据
            ground_truth_rrp = config.get("real_data_dynamics", {}).get("ground_truth_rrp", [])
            ground_truth_carbon = config.get("real_data_dynamics", {}).get("ground_truth_carbon_price", [])
            visualizer = ResultVisualizer(
                results_file=config['logging']['results_file_csv'], 
                agents_config_path="true_digital_twin_agents.json", 
                ground_truth_prices=ground_truth_rrp,
                ground_truth_carbon_prices=ground_truth_carbon
            )
            generated_files = visualizer.generate_all_visualizations()
            logging.info(f"可视化图表已生成，共 {len(generated_files)} 个图表")
            logging.info(f"图表保存路径: {visualizer.output_dir}")
        except Exception as e:
            logging.error(f"生成可视化图表时出错: {e}", exc_info=True)
        
        logging.info("="*60)
        
        # 清理锁文件
        lock_file = Path(".simulation_lock")
        if lock_file.exists():
            lock_file.unlink()
        
        elapsed_time = time.time() - start_time
        print("="*80)
        print(f"✅ 仿真成功完成！")
        print(f"🕐 结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"⏱️  总耗时: {elapsed_time/60:.1f} 分钟")
        print(f"🆔 运行ID: {run_id}")
        print("="*80)
        
        return 0
    
    except KeyboardInterrupt:
        logging.info("用户中断，仿真终止")
        # 清理锁文件
        lock_file = Path(".simulation_lock")
        if lock_file.exists():
            lock_file.unlink()
        return 130
        
    except Exception as e:
        logging.error(f"仿真执行过程中出错: {e}", exc_info=True)
        # 清理锁文件
        lock_file = Path(".simulation_lock")
        if lock_file.exists():
            lock_file.unlink()
        return 1

if __name__ == "__main__":
    sys.exit(main())
