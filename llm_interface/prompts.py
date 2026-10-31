"""
存储用于与LLM交互的提示模板。
"""

# ------------------------------------------
# 核心决策 Prompt (包含信念和经验学习)
# ------------------------------------------
DECISION_MAKING_PROMPT_TEMPLATE = """
作为一个理性的发电商智能体，你的目标是在当前的电力现货与备用联合市场中最大化你的利润。
你正在参与第 {round_number} 轮市场报价。

# 你的角色描述:
{role_description}

# 你的私有信息:
- 发电机组ID: {agent_id}
- 技术类型: {technology_type}
- 能源生产边际成本: {marginal_cost_energy:.2f} 元/MWh
- 备用提供边际成本: {marginal_cost_reserve:.2f} 元/MWh (通常低于能源成本，代表机会成本或准备成本)
- 最大总容量 (能源+备用): {max_capacity:.2f} MW
- 最小发电出力 (如有): {min_output:.2f} MW (简化模型中可能不使用)
- 爬坡限制等 (简化模型中可能不使用)

# 当前市场公开信息:
- 预测系统总负荷: {total_demand:.2f} MW
- 预测系统所需备用: {reserve_requirement:.2f} MW
- 竞争对手数量: {num_competitors}
- (可选) 上一轮市场出清价格 (如果适用):
    - 现货能源价格: {last_energy_price:.2f} 元/MWh
    - 备用价格: {last_reserve_price:.2f} 元/MWh

# 你对竞争对手的当前信念 (基于历史观察的估计):
{belief_summary}
(信念说明: 这里总结了你对其他竞争对手可能的成本或行为的看法。例如，估计他们的平均成本范围，或者他们是否倾向于激进报价等。)

# 你的历史经验总结 (最近几轮的表现和反思):
{experience_summary}
(经验说明: 回顾过去的报价、中标情况和利润。例如，"在第N轮，报价X，利润Y，原因是..."、"当市场价格高于某值时，某策略似乎更有效。")

# 市场事件和预期行为指南:
- 对市场公告的各类事件（如燃料价格变化、供应中断等）及时做出反应
- 当收到未来市场变化的公告时，应提前调整报价策略，而不是等到变化实际发生
- 预期行为是理性的市场策略，提前反映未来变化可以帮助你获得更好的利润
- 根据你的技术类型，采取相应的应对策略:
  * 燃料依赖型机组: 对相关燃料价格变化较为敏感，应及时调整报价
  * 可再生能源机组: 虽然对燃料价格不敏感，但可以根据市场整体变化调整策略

# 任务与要求:
根据以上所有信息，进行推理和博弈分析，为当前这一轮制定一个有利可图的战略性报价。
你需要同时报出现货能源的价格和数量，以及备用的价格和数量。
你的总报价数量 (能源 + 备用) 不能超过你的最大容量。
报价需要平衡争取中标概率和获得高利润之间的关系。考虑到不完全信息和竞争对手可能的反应。

## 战略思考指南:
1. **市场事件快速响应**: 对市场公告的各类事件立即做出反应。特别是当事件影响你的成本结构时，应提前调整策略。
2. **预期行动**: 如果收到未来市场变化的公告，应在变化实际发生前适当调整报价，反映预期影响。
3. **差异化策略**:
   - 如果你的成本结构将受到影响：提前调整报价，同时考虑容量分配策略
   - 如果竞争对手的成本结构将受到影响：准备调整你的策略以应对市场结构变化
4. **理性预期行为**:
   - 预期行为是理性的市场策略，可以帮助你在市场变化前获得更好的利润
   - 根据市场信息及时调整策略，避免错失利润机会

# 输出格式要求:
请严格按照以下JSON格式返回你的报价决策，不要包含任何额外的解释或说明文字。
```json
{{
  "energy_bid": {{
    "price": <你的能源报价价格 (元/MWh)>,
    "quantity": <你的能源报价数量 (MW)>
  }},
  "reserve_bid": {{
    "price": <你的备用报价价格 (元/MWh)>,
    "quantity": <你的备用报价数量 (MW)>
  }},
  "reasoning": "<对你报价策略的简短说明，例如基于什么考虑，预期市场情况等 (可选但推荐)>"
}}
```

请现在生成你的报价决策。
"""

# ------------------------------------------
# (可选) 信念更新辅助 Prompt
# ------------------------------------------
BELIEF_UPDATE_ASSIST_PROMPT_TEMPLATE = """
作为发电商智能体 {agent_id}，你刚刚观察到了第 {round_number} 轮的市场出清结果。
你的目标是根据这些新信息，更新你对竞争对手 ({competitor_ids}) 的信念（例如他们的成本或行为模式）。

# 你在本轮的报价和结果:
- 你的能源报价: 价格 {my_energy_price:.2f}, 数量 {my_energy_quantity:.2f}
- 你的备用报价: 价格 {my_reserve_price:.2f}, 数量 {my_reserve_quantity:.2f}
- 你中标的能源量: {my_cleared_energy:.2f} MW
- 你中标的备用量: {my_cleared_reserve:.2f} MW
- 市场出清价格: 能源 {market_energy_price:.2f} 元/MWh, 备用 {market_reserve_price:.2f} 元/MWh

# 你之前的信念:
{previous_belief_summary}

# 任务与要求:
分析市场出清价格和你自身的中标情况，推断竞争对手可能的报价行为。
例如：
- 市场价格远高于你的成本，意味着什么？可能是竞争对手成本高，或者需求高，或者有市场力行为？
- 你未能中标或部分中标，意味着什么？可能是你的报价太高，或者竞争对手报价非常低？
- 能源和备用价格的关系如何？这反映了备用的稀缺性或竞争程度吗？

基于你的分析，请以简明扼要的方式描述你更新后的信念。

# 输出格式要求:
请以简短的文本描述形式返回你更新后的信念摘要。例如：
"根据本轮较高的能源价格({market_energy_price:.2f})和备用价格({market_reserve_price:.2f})，以及我方({agent_id})能源全额中标但备用部分中标的情况，我推断至少部分竞争对手的能源成本可能高于预期，并且备用市场竞争激烈。我将略微调高对竞争对手平均成本的估计，并认为他们可能会在备用市场报出较低价格。"
或者，如果信息不足以显著改变信念，可以返回：
"本轮市场结果与预期大致相符，未提供足够强的信号来显著改变我对竞争对手的现有信念。"

请现在生成你更新后的信念摘要。
"""

# ------------------------------------------
# (可选) 经验总结辅助 Prompt
# ------------------------------------------
EXPERIENCE_SUMMARY_ASSIST_PROMPT_TEMPLATE = """
作为发电商智能体 {agent_id}，你需要总结过去 {num_rounds_to_summarize} 轮的市场经验，以便在未来做出更好的决策。

# 历史记录 (最近 {num_rounds_to_summarize} 轮):
{historical_data_snippet}
(格式说明: 每行代表一轮，包含: 轮次, 能源报价(价,量), 备用报价(价,量), 中标能源量, 中标备用量, 能源价格, 备用价格, 利润)

# 任务与要求:
分析上述历史数据，识别成功的模式、失败的教训，以及市场价格与你的报价/利润之间的关系。
形成一个简明扼要的经验总结，突出关键的发现和可以指导未来报价的策略调整建议。

# 输出格式要求:
请以简短的文本描述形式返回你的经验总结。例如：
"总结过去{num_rounds_to_summarize}轮经验：1) 当市场能源价格预测较高时，略微提高能源报价价格仍能中标且利润不错。2) 在备用需求高时，即使备用报价略高于边际成本也有机会中标。3) 避免能源和备用总报价量超过最大容量。4) 在第X轮报价过高导致零中标，应吸取教训。未来策略建议：更关注预测负荷和备用需求，动态调整报价。"

请现在生成你的经验总结。
"""

# --- Helper function to format prompts ---
def format_decision_prompt(round_number, agent_id, private_info, market_info, belief_summary, experience_summary, role_description=None):
    """填充决策Prompt模板"""
    return DECISION_MAKING_PROMPT_TEMPLATE.format(
        round_number=round_number,
        agent_id=agent_id,
        technology_type=private_info.get('technology_type', '未指定'),
        marginal_cost_energy=private_info.get('marginal_cost_energy', 0.0),
        marginal_cost_reserve=private_info.get('marginal_cost_reserve', 0.0),
        max_capacity=private_info.get('max_capacity', 100.0),
        min_output=private_info.get('min_output', 0.0),
        total_demand=market_info.get('total_demand', 500.0),
        reserve_requirement=market_info.get('reserve_requirement', 50.0),
        num_competitors=market_info.get('num_competitors', 3),
        last_energy_price=market_info.get('last_energy_price', 0.0),
        last_reserve_price=market_info.get('last_reserve_price', 0.0),
        belief_summary=belief_summary if belief_summary else "你目前对竞争对手的信息了解有限。",
        experience_summary=experience_summary if experience_summary else "你暂时还没有足够的历史经验可供参考。",
        role_description=role_description if role_description else "一家普通的发电企业"
    )

def format_belief_update_prompt(round_number, agent_id, my_bid_result, market_prices, previous_belief_summary, competitor_ids):
     """填充信念更新辅助Prompt模板"""
     # competitor_ids 应该是一个列表或元组
     competitor_id_str = ", ".join(map(str, competitor_ids))
     return BELIEF_UPDATE_ASSIST_PROMPT_TEMPLATE.format(
         round_number=round_number,
         agent_id=agent_id,
         my_energy_price=my_bid_result.get('energy_bid',{}).get('price', 0.0),
         my_energy_quantity=my_bid_result.get('energy_bid',{}).get('quantity', 0.0),
         my_reserve_price=my_bid_result.get('reserve_bid',{}).get('price', 0.0),
         my_reserve_quantity=my_bid_result.get('reserve_bid',{}).get('quantity', 0.0),
         my_cleared_energy=my_bid_result.get('cleared_energy', 0.0),
         my_cleared_reserve=my_bid_result.get('cleared_reserve', 0.0),
         market_energy_price=market_prices.get('energy_price', 0.0),
         market_reserve_price=market_prices.get('reserve_price', 0.0),
         previous_belief_summary=previous_belief_summary if previous_belief_summary else "初始信念。",
         competitor_ids=competitor_id_str
     )

def format_experience_summary_prompt(agent_id, historical_data_snippet, num_rounds_to_summarize):
    """填充经验总结辅助Prompt模板"""
    return EXPERIENCE_SUMMARY_ASSIST_PROMPT_TEMPLATE.format(
        agent_id=agent_id,
        historical_data_snippet=historical_data_snippet,
        num_rounds_to_summarize=num_rounds_to_summarize
    )