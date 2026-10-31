#!/usr/bin/env python3
"""
计算仿真结果的预测精度指标: MASE 和 MAPE
"""

import pandas as pd
import numpy as np

def calculate_mase(actual, predicted, seasonal_period=1):
    """
    计算 Mean Absolute Scaled Error (MASE)
    MASE = MAE / MAE_naive
    """
    # 计算预测误差的平均绝对误差
    mae = np.mean(np.abs(actual - predicted))
    
    # 计算朴素预测的平均绝对误差 (使用季节性朴素预测)
    if seasonal_period == 1:
        # 对于非季节性数据，使用前一期作为预测
        naive_errors = np.abs(np.diff(actual))
    else:
        # 对于季节性数据
        naive_errors = np.abs(actual[seasonal_period:] - actual[:-seasonal_period])
    
    mae_naive = np.mean(naive_errors)
    
    if mae_naive == 0:
        return np.inf if mae > 0 else 0
    
    return mae / mae_naive

def calculate_mape(actual, predicted):
    """
    计算 Mean Absolute Percentage Error (MAPE)
    MAPE = (1/n) * Σ|((actual - predicted) / actual)| * 100
    """
    # 避免除零错误
    mask = actual != 0
    if not mask.any():
        return np.inf
    
    mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
    return mape

def calculate_additional_metrics(actual, predicted):
    """计算其他预测精度指标"""
    # MAE - Mean Absolute Error
    mae = np.mean(np.abs(actual - predicted))
    
    # MSE - Mean Squared Error
    mse = np.mean((actual - predicted) ** 2)
    
    # RMSE - Root Mean Squared Error
    rmse = np.sqrt(mse)
    
    # R² - Coefficient of Determination
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R²': r2
    }

def main():
    print("📊 计算仿真预测精度指标: MASE & MAPE")
    print("="*60)
    
    # 读取仿真结果
    df = pd.read_csv('results/simulation_results.csv', on_bad_lines='skip')
    
    # 真实的Ground Truth数据
    ground_truth_rrp = [85.3, 92.1, 88.5, 95.4, 105.2, 115.8, 135.7, 180.9, 210.4, 225.6]
    # 真实ACCU现货价格 (2022年3-6月) - 基于Figure 1.2图表数据
    ground_truth_carbon = [144.0, 146.4, 148.8, 153.6, 158.4, 163.2, 168.0, 170.4, 172.8, 170.4]
    
    # 仿真预测结果
    sim_energy_prices = df['energy_price'].values
    sim_carbon_prices = df['carbon_market_price'].values
    
    print(f"📈 电力价格预测精度分析:")
    print("-" * 40)
    
    # 电力价格的MASE和MAPE
    energy_mase = calculate_mase(np.array(ground_truth_rrp), sim_energy_prices)
    energy_mape = calculate_mape(np.array(ground_truth_rrp), sim_energy_prices)
    energy_metrics = calculate_additional_metrics(np.array(ground_truth_rrp), sim_energy_prices)
    
    print(f"真实RRP平均: {np.mean(ground_truth_rrp):.2f} 元/MWh")
    print(f"仿真平均: {np.mean(sim_energy_prices):.2f} 元/MWh")
    print(f"")
    print(f"🎯 主要指标:")
    print(f"   MASE: {energy_mase:.4f}")
    print(f"   MAPE: {energy_mape:.2f}%")
    print(f"")
    print(f"📊 其他指标:")
    print(f"   MAE:  {energy_metrics['MAE']:.2f} 元/MWh")
    print(f"   RMSE: {energy_metrics['RMSE']:.2f} 元/MWh")
    print(f"   R²:   {energy_metrics['R²']:.4f}")
    
    # MASE解释
    if energy_mase < 1:
        mase_interpretation = "优秀 (比朴素预测更准确)"
    elif energy_mase < 1.5:
        mase_interpretation = "良好"
    elif energy_mase < 2:
        mase_interpretation = "一般"
    else:
        mase_interpretation = "需要改进"
    
    # MAPE解释
    if energy_mape < 10:
        mape_interpretation = "非常准确"
    elif energy_mape < 20:
        mape_interpretation = "准确"
    elif energy_mape < 50:
        mape_interpretation = "合理"
    else:
        mape_interpretation = "需要改进"
    
    print(f"")
    print(f"📝 电力价格预测评价:")
    print(f"   MASE {energy_mase:.4f}: {mase_interpretation}")
    print(f"   MAPE {energy_mape:.2f}%: {mape_interpretation}")
    
    print(f"\n🌱 碳价格预测精度分析:")
    print("-" * 40)
    
    # 碳价格的MASE和MAPE
    carbon_mase = calculate_mase(np.array(ground_truth_carbon), sim_carbon_prices)
    carbon_mape = calculate_mape(np.array(ground_truth_carbon), sim_carbon_prices)
    carbon_metrics = calculate_additional_metrics(np.array(ground_truth_carbon), sim_carbon_prices)
    
    print(f"真实ACCU平均: {np.mean(ground_truth_carbon):.2f} 元/吨CO2")
    print(f"仿真平均 (分层配额): {np.mean(sim_carbon_prices):.2f} 元/吨CO2")
    print(f"")
    print(f"🎯 主要指标:")
    print(f"   MASE: {carbon_mase:.4f}")
    print(f"   MAPE: {carbon_mape:.2f}%")
    print(f"")
    print(f"📊 其他指标:")
    print(f"   MAE:  {carbon_metrics['MAE']:.2f} 元/吨CO2")
    print(f"   RMSE: {carbon_metrics['RMSE']:.2f} 元/吨CO2")
    print(f"   R²:   {carbon_metrics['R²']:.4f}")
    
    # 碳价格MASE解释
    if carbon_mase < 1:
        carbon_mase_interpretation = "优秀 (比朴素预测更准确)"
    elif carbon_mase < 1.5:
        carbon_mase_interpretation = "良好"
    elif carbon_mase < 2:
        carbon_mase_interpretation = "一般"
    else:
        carbon_mase_interpretation = "需要改进"
    
    # 碳价格MAPE解释
    if carbon_mape < 10:
        carbon_mape_interpretation = "非常准确"
    elif carbon_mape < 20:
        carbon_mape_interpretation = "准确"
    elif carbon_mape < 50:
        carbon_mape_interpretation = "合理"
    else:
        carbon_mape_interpretation = "需要改进"
    
    print(f"")
    print(f"📝 碳价格预测评价:")
    print(f"   MASE {carbon_mase:.4f}: {carbon_mase_interpretation}")
    print(f"   MAPE {carbon_mape:.2f}%: {carbon_mape_interpretation}")
    
    # 详细对比表
    print(f"\n📋 逐轮详细对比:")
    print("轮次 | 真实RRP | 仿真RRP | RRP误差% | 真实碳价 | 仿真碳价 | 碳价误差%")
    print("-" * 75)
    
    for i in range(len(ground_truth_rrp)):
        real_rrp = ground_truth_rrp[i]
        sim_rrp = sim_energy_prices[i]
        real_carbon = ground_truth_carbon[i]
        sim_carbon = sim_carbon_prices[i]
        
        rrp_error = abs(real_rrp - sim_rrp) / real_rrp * 100
        carbon_error = abs(real_carbon - sim_carbon) / real_carbon * 100
        
        print(f"{i:4d} | {real_rrp:7.1f} | {sim_rrp:7.1f} | {rrp_error:7.1f}% | {real_carbon:7.1f} | {sim_carbon:7.1f} | {carbon_error:7.1f}%")
    
    # 总体评估
    print(f"\n🏆 总体预测精度评估:")
    print("="*60)
    
    overall_score = 0
    if energy_mase < 1.5 and energy_mape < 30:
        print("✅ 电力价格预测: 良好")
        overall_score += 1
    else:
        print("⚠️ 电力价格预测: 需要改进")
    
    if carbon_mase < 1.5 and carbon_mape < 20:
        print("✅ 碳价格预测: 良好")
        overall_score += 1
    else:
        print("⚠️ 碳价格预测: 需要改进")
    
    if overall_score == 2:
        print("\n🎉 仿真系统预测精度: 优秀")
        print("   您的数字孪生系统能够准确预测市场价格！")
    elif overall_score == 1:
        print("\n👍 仿真系统预测精度: 良好")
        print("   系统在某些方面表现出色，有改进空间。")
    else:
        print("\n🔧 仿真系统预测精度: 需要优化")
        print("   建议调整模型参数或增加更多真实数据。")

if __name__ == "__main__":
    main()
