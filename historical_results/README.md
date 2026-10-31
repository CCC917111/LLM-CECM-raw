# Historical simulation results

Three recorded coupled electricity-carbon market runs, each with 50 LLM agents and 72 rounds.

| Run | Start time recorded in the log | ADF | IBR-CR |
|---|---|---|---|
| A | 2025-09-24 20:49:02 | Enabled | Enabled |
| B | 2025-09-25 12:39:08 | Enabled | Disabled |
| C | 2025-09-25 18:24:54 | Disabled | Enabled |

The CSV files were extracted from the recorded simulation logs. Each run directory contains its own agent population and results.

## Market prices

`market.csv` contains one row per round:

| Field | Description |
|---|---|
| `round` | Round index, from 0 to 71 |
| `energy_price` | Electricity market clearing price |
| `carbon_price` | Carbon market clearing price |

## Agent parameters and profits

`agents.csv` contains one row per agent:

| Field | Description |
|---|---|
| `idx` | Agent index within the run |
| `agent_id` | Agent identifier |
| `technology_type` | Technology description |
| `class_key` | Technology category |
| `energy_mc` | Marginal energy cost |
| `reserve_mc` | Marginal reserve cost |
| `max_capacity` | Maximum capacity in MW |
| `min_output` | Minimum output in MW |
| `belief_mean`, `belief_std` | Initial belief parameters |
| `total_profit` | Cumulative profit over 72 rounds |
| `avg_profit` | Average profit per round |
| `log_line` | Agent configuration line in the source log |
| `role` | Agent role description |
