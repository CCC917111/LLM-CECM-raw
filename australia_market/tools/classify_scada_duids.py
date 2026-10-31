#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import argparse
import json
import random
import statistics
from collections import defaultdict
from typing import Dict
try:
    import pandas as pd
except Exception:
    pd = None

# 1) 明确的 DUID -> 燃料类型 映射（来自用户提供的表 + 已知机组）
EXACT = {
    # 用户图片中的条目
    'NPS': 'Black_Coal',            # Northern Power Station（如与实际不符可再修正）
    'DDPS1': 'Natural_Gas',         # Darling Downs (CCGT)
    'MPP_1': 'Black_Coal',          # Millmerran Unit 1
    'MPP_2': 'Black_Coal',          # Millmerran Unit 2
    'TNPS1': 'Natural_Gas',         # Torrens Island Unit 1
    'TALWA1': 'Natural_Gas',        # Tallawarra (CCGT)

    # NSW 大型黑煤
    'ER01': 'Black_Coal', 'ER02': 'Black_Coal', 'ER03': 'Black_Coal', 'ER04': 'Black_Coal',
    'BW01': 'Black_Coal', 'BW02': 'Black_Coal', 'BW03': 'Black_Coal', 'BW04': 'Black_Coal',
    'VP5': 'Black_Coal', 'VP6': 'Black_Coal',

    # VIC 褐煤
    'LYA1': 'Brown_Coal', 'LYA2': 'Brown_Coal', 'LYA3': 'Brown_Coal', 'LYA4': 'Brown_Coal',
    'LOYYB1': 'Brown_Coal', 'LOYYB2': 'Brown_Coal',
    'YWPS1': 'Brown_Coal', 'YWPS2': 'Brown_Coal', 'YWPS3': 'Brown_Coal', 'YWPS4': 'Brown_Coal',

    # QLD 黑煤
    'CALL_B_1': 'Black_Coal', 'CALL_B_2': 'Black_Coal',
    'MP1': 'Black_Coal',        # Mt Piper 1（在样本中出现）
    'KPP_1': 'Black_Coal',      # Kogan Creek 1

    # NSW Liddell（用户图与SCADA大机组未识别，补充为黑煤）
    'LD01': 'Black_Coal', 'LD02': 'Black_Coal', 'LD03': 'Black_Coal', 'LD04': 'Black_Coal',

    # 其他样本中出现的
    'MACARTH1': 'Wind',         # Macarthur Wind Farm
    'VBBG1': 'Battery',         # Victorian Big Battery（非发电燃料，但单列便于查看）
}

# 2) 稳健的前缀/关键词映射（确定性规则，非凭空猜测）
PREFIX = [
    # 煤炭（黑煤/褐煤）
    ('STAN-', 'Black_Coal'), ('TARONG', 'Black_Coal'), ('GSTONE', 'Black_Coal'), ('CALL', 'Black_Coal'),
    ('MP', 'Black_Coal'), ('KPP', 'Black_Coal'),
    ('LYA', 'Brown_Coal'), ('LOYYB', 'Brown_Coal'), ('YWPS', 'Brown_Coal'),

    # 天然气
    ('BRAEMAR', 'Natural_Gas'), ('TORRENS', 'Natural_Gas'), ('TNPS', 'Natural_Gas'),
    ('PPCCGT', 'Natural_Gas'), ('OSBORNE', 'Natural_Gas'), ('ROMA', 'Natural_Gas'),
    ('OAKEY', 'Natural_Gas'), ('SWANBANK', 'Natural_Gas'), ('URANQ', 'Natural_Gas'),
    ('COLONGRA', 'Natural_Gas'), ('JEERALANG', 'Natural_Gas'), ('LAVERTON', 'Natural_Gas'),
    ('NEWPORT', 'Natural_Gas'), ('SOMERTON', 'Natural_Gas'), ('TALWA', 'Natural_Gas'),

    # 水电
    ('TUMUT', 'Hydro'), ('MURRAY', 'Hydro'), ('UPPTUMUT', 'Hydro'), ('SNOWYP', 'Hydro'),
    ('GORDON', 'Hydro'), ('BASTYAN', 'Hydro'), ('TREVALLYN', 'Hydro'),

    # 风电
    ('COOPGWF', 'Wind'), ('SAPHWF', 'Wind'), ('COLWF', 'Wind'), ('MACARTH', 'Wind'),
    ('WF', 'Wind'), ('WIND', 'Wind'),

    # 太阳能
    ('DARLSF', 'Solar'), ('LIMOSF', 'Solar'), ('SUNRSF', 'Solar'),
    ('SF', 'Solar'), ('SOLAR', 'Solar'), ('PV', 'Solar'),
]

# 可选：由官方机组信息表构建的映射（运行时加载）
EXCEL_MAP: Dict[str, str] = {}

# 风/光常见后缀
def classify_suffix(duid: str):
    if duid.endswith('WF1') or duid.endswith('WF2'): return 'Wind'
    if duid.endswith('SF1') or duid.endswith('SF2'): return 'Solar'
    return None


def classify_duid(duid: str) -> str:
    d = duid.upper()
    # 0) Excel 映射最高优先级
    if d in EXCEL_MAP:
        return EXCEL_MAP[d]
    if d in EXACT:
        return EXACT[d]
    for pref, cat in PREFIX:
        if d.startswith(pref) or (pref in d):
            return cat
    s = classify_suffix(d)
    if s:
        return s
    return 'Other'


def analyze_scada(scada_path: str, min_points: int = 5, min_cap: float = 1.0):
    max_by_duid = defaultdict(float)
    cnt_by_duid = defaultdict(int)
    prev_by_duid = {}

    # 为分位近似采样
    RES_SIZE_VALUE = 600
    RES_SIZE_DELTA = 800
    pos_seen = defaultdict(int)
    up_seen = defaultdict(int)
    down_seen = defaultdict(int)
    samples_pos = defaultdict(list)      # 正出力样本（估计Pmin分位）
    samples_up = defaultdict(list)       # 上爬坡样本（ΔP>0）
    samples_down = defaultdict(list)     # 下爬坡样本（ΔP<0，取正值）

    # 流式读取，内存友好
    with open(scada_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            if not line.startswith('D,DISPATCH,UNIT_SCADA'):
                continue
            parts = line.rstrip('\n').split(',')
            if len(parts) < 7:
                continue
            duid = parts[5].strip('"').upper()
            try:
                val = abs(float(parts[6])) if parts[6] else 0.0
            except Exception:
                continue
            cnt_by_duid[duid] += 1
            if val > max_by_duid[duid]:
                max_by_duid[duid] = val
            # 采样正出力
            if val > 0:
                pos_seen[duid] += 1
                k = pos_seen[duid]
                buf = samples_pos[duid]
                if len(buf) < RES_SIZE_VALUE:
                    buf.append(val)
                else:
                    j = random.randint(1, k)
                    if j <= RES_SIZE_VALUE:
                        buf[j-1] = val
            # 采样爬坡
            if duid in prev_by_duid:
                delta = val - prev_by_duid[duid]
                if delta > 0:
                    up_seen[duid] += 1
                    k = up_seen[duid]
                    buf = samples_up[duid]
                    if len(buf) < RES_SIZE_DELTA:
                        buf.append(delta)
                    else:
                        j = random.randint(1, k)
                        if j <= RES_SIZE_DELTA:
                            buf[j-1] = delta
                elif delta < 0:
                    down_seen[duid] += 1
                    k = down_seen[duid]
                    buf = samples_down[duid]
                    mag = -delta
                    if len(buf) < RES_SIZE_DELTA:
                        buf.append(mag)
                    else:
                        j = random.randint(1, k)
                        if j <= RES_SIZE_DELTA:
                            buf[j-1] = mag
            prev_by_duid[duid] = val
    
    # 统计分类
    fuel_cap = defaultdict(float)
    fuel_units = defaultdict(int)
    other_big = []
    # 用于动态比率（按燃料）
    ratio_pmin_lists = defaultdict(list)
    ratio_ramp_up_lists = defaultdict(list)
    ratio_ramp_down_lists = defaultdict(list)

    for duid, mx in max_by_duid.items():
        if mx < min_cap or cnt_by_duid[duid] < min_points:
            continue
        cat = classify_duid(duid)
        fuel_cap[cat] += mx
        fuel_units[cat] += 1
        if cat == 'Other' and mx > 300:
            other_big.append((duid, mx))
        # 计算该DUID的分位统计及比率
        vals = sorted(samples_pos.get(duid, []))
        def perc(a, q):
            if not a:
                return 0.0
            idx = int(max(0, min(len(a)-1, round(q*(len(a)-1)))))
            return float(a[idx])
        pmin_est = perc(vals, 0.10)
        up_list = sorted(samples_up.get(duid, []))
        down_list = sorted(samples_down.get(duid, []))
        ramp_up_p95 = perc(up_list, 0.95)
        ramp_down_p95 = perc(down_list, 0.95)
        if mx > 0:
            ratio_pmin_lists[cat].append(pmin_est / mx)
            ratio_ramp_up_lists[cat].append(ramp_up_p95 / mx)
            ratio_ramp_down_lists[cat].append(ramp_down_p95 / mx)

    total_cap = sum(fuel_cap.values())

    # 输出
    print('📊 重新分类后的燃料结构:')
    print('-' * 80)
    print('燃料类型       机组数  总容量(MW)  平均容量(MW)  占比(%)')
    print('-' * 80)
    for cat in sorted(fuel_cap.keys(), key=lambda c: fuel_cap[c], reverse=True):
        cap = fuel_cap[cat]
        units = fuel_units[cat]
        avg = (cap / units) if units else 0
        pct = (cap / total_cap * 100) if total_cap else 0
        print(f'{cat:<14} {units:>6}  {cap:>10.1f}  {avg:>12.1f}  {pct:>7.2f}')
    print('-' * 80)
    print(f'总计           {sum(fuel_units.values()):>6}  {total_cap:>10.1f}')

    # 计算各燃料的动态比率（中位数）
    fuel_dynamic_ratios = {}
    print('\n⚙️  各燃料动态比率（基于SCADA分位，按DUID中位聚合）:')
    print('-' * 80)
    print('燃料类型       pmin_ratio   ramp_up_ratio   ramp_down_ratio')
    print('-' * 80)
    for cat in sorted(fuel_cap.keys(), key=lambda c: fuel_cap[c], reverse=True):
        pmin_list = ratio_pmin_lists.get(cat, [])
        up_list = ratio_ramp_up_lists.get(cat, [])
        down_list = ratio_ramp_down_lists.get(cat, [])
        def med(lst):
            return float(statistics.median(lst)) if lst else 0.0
        r_pmin = med(pmin_list)
        r_up = med(up_list)
        r_down = med(down_list)
        fuel_dynamic_ratios[cat] = {
            'pmin_ratio': r_pmin,
            'ramp_up_ratio': r_up,
            'ramp_down_ratio': r_down,
        }
        print(f'{cat:<14} {r_pmin:>10.3f}   {r_up:>13.3f}   {r_down:>15.3f}')

    # 写出到 JSON，供构建器使用
    out_json = os.path.join(os.path.dirname(__file__), 'fuel_dynamic_ratios.json')
    try:
        with open(out_json, 'w', encoding='utf-8') as fjson:
            json.dump(fuel_dynamic_ratios, fjson, ensure_ascii=False, indent=2)
        print(f"\n💾 已写出动态比率: {out_json}")
    except Exception as e:
        print(f"⚠️ 写出动态比率失败: {e}")

    other_big.sort(key=lambda x: x[1], reverse=True)
    if other_big:
        print('\n🔍 仍需人工确认的 Other(>300MW) 机组 TOP15:')
        for duid, cap in other_big[:15]:
            print(f'  {duid:<12} {cap:>8.1f} MW')

    # 返回结构化结果供后续使用
    return {
        'fuel_cap': dict(fuel_cap),
        'fuel_units': dict(fuel_units),
        'total_cap': total_cap,
        'other_big': other_big,
    }


def _norm_fuel_category(raw: str) -> str:
    s = str(raw).lower()
    if 'black coal' in s or ('coal' in s and 'brown' not in s):
        return 'Black_Coal'
    if 'brown coal' in s:
        return 'Brown_Coal'
    if 'gas' in s:
        return 'Natural_Gas'
    if 'hydro' in s or 'water' in s:
        return 'Hydro'
    if 'wind' in s:
        return 'Wind'
    if 'solar' in s:
        return 'Solar'
    if 'battery' in s:
        return 'Battery'
    return 'Other'


def _load_excel_mapping(excel_path: str) -> int:
    global EXCEL_MAP
    if not excel_path:
        return 0
    if pd is None:
        print('⚠️ 未安装pandas，跳过Excel映射加载')
        return 0
    if not os.path.exists(excel_path):
        print(f'⚠️ Excel 文件不存在: {excel_path}，跳过Excel映射加载')
        return 0
    try:
        df = pd.read_excel(excel_path, sheet_name='ExistingGeneration&NewDevs', header=1)
    except Exception as e:
        print(f'⚠️ 读取Excel失败: {e}')
        return 0

    cols = {str(c).lower(): c for c in df.columns}

    def pick(substr: str):
        for k, v in cols.items():
            if substr in k:
                return v
        return None

    c_asset = pick('asset type')
    c_duid = pick('duid')
    c_fuel = pick('fuel type')
    c_status = pick('unit status')

    use = df
    if c_asset is not None:
        use = use[use[c_asset].astype(str).str.contains('Existing', case=False, na=False)]
    if c_status is not None:
        use = use[use[c_status].astype(str).str.contains('Service', case=False, na=False)]
    if c_duid is None or c_fuel is None:
        print('⚠️ Excel中未找到DUID或Fuel Type列，跳过Excel映射加载')
        return 0

    use = use[[c_duid, c_fuel]].dropna()
    use[c_duid] = use[c_duid].astype(str).str.upper()
    mapped = {duid: _norm_fuel_category(fuel) for duid, fuel in zip(use[c_duid], use[c_fuel])}

    # 合并到全局映射
    EXCEL_MAP.update(mapped)
    return len(mapped)


def main():
    parser = argparse.ArgumentParser(description='Classify DUIDs from AEMO SCADA by fuel type (Excel mapping + deterministic rules).')
    parser.add_argument('--scada', required=True, help='Path to PUBLIC_DVD_DISPATCH_UNIT_SCADA_*.CSV')
    parser.add_argument('--excel', default='', help='Path to NEM Generation Information Excel (to build DUID→Fuel map)')
    parser.add_argument('--min-points', type=int, default=5)
    parser.add_argument('--min-cap', type=float, default=1.0)
    args = parser.parse_args()

    if not os.path.exists(args.scada):
        print(f'❌ SCADA 文件不存在: {args.scada}')
        sys.exit(1)

    # 先加载Excel映射（若提供）
    if args.excel:
        n = _load_excel_mapping(args.excel)
        print(f'✅ 已从Excel加载映射: {n} 条 (DUID→FuelCategory)')

    analyze_scada(args.scada, min_points=args.min_points, min_cap=args.min_cap)


if __name__ == '__main__':
    main()
