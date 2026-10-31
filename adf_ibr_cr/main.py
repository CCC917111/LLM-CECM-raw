#!/usr/bin/env python3
# main.py
import os
import sys
import logging
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent))

from simulation.config import get_config
from simulation.simulator import MarketSimulator
from utils.logger import setup_logging
from utils.visualization import ResultVisualizer


def main():
    """
    电力市场多智能体仿真系统的主入口。
    """
    # 获取配置
    config = get_config()
    
    # 设置日志
    setup_logging(
        log_file=config["logging"]["log_file"], 
        log_level=config["log_level"]
    )
    
    logging.info("="*60)
    logging.info(f"启动电力市场LLM智能体仿真: {config['simulation_name']}")
    logging.info(f"计划执行 {config['num_rounds']} 轮市场交易")
    logging.info("="*60)
    
    try:
        # 初始化并运行仿真器
        simulator = MarketSimulator(config)
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
            visualizer = ResultVisualizer(config['logging']['results_file_csv'])
            generated_files = visualizer.generate_all_visualizations()
            logging.info(f"可视化图表已生成，共 {len(generated_files)} 个图表")
            logging.info(f"图表保存路径: {visualizer.output_dir}")
        except Exception as e:
            logging.error(f"生成可视化图表时出错: {e}", exc_info=True)
        
        logging.info("="*60)
        
        return 0
    
    except KeyboardInterrupt:
        logging.info("用户中断，仿真终止")
        return 130
        
    except Exception as e:
        logging.error(f"仿真执行过程中出错: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())