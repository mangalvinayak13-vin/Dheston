**Following the jump model one trade at a time, test years (backtest, not advice)**

| Entry | Signal / holding | Cost per side | Trades | Wins per 100 | Blind, same mix | Rs per trade | 95% range (Rs per trade) |
|---|---|---|---|---|---|---|---|
| signal day close | forecast_edge / plain | 0% | 1,370 | 43.9 | 42.1 | +6.38 | [+1.35, +12.95] |
| signal day close | forecast_edge / plain | 1% | 1,370 | 43.0 | 41.0 | +4.30 | [-0.69, +10.85] |
| signal day close | forecast_edge / hedged | 0% | 1,370 | 43.5 | 41.4 | +5.78 | [+1.83, +10.03] |
| signal day close | forecast_edge / hedged | 1% | 1,370 | 41.1 | 38.5 | +3.69 | [-0.24, +7.93] |
| signal day close | rich_cheap_edge / plain | 0% | 4,168 | 44.9 | 43.4 | +2.69 | [+0.22, +5.45] |
| signal day close | rich_cheap_edge / plain | 1% | 4,168 | 43.7 | 42.2 | +0.71 | [-1.75, +3.44] |
| signal day close | rich_cheap_edge / hedged | 0% | 1,313 | 44.5 | 41.9 | +6.48 | [+2.33, +11.21] |
| signal day close | rich_cheap_edge / hedged | 1% | 1,313 | 42.0 | 39.0 | +4.34 | [+0.19, +8.98] |
| one day later | forecast_edge / plain | 0% | 845 | 44.6 | 42.1 | +6.04 | [-2.26, +14.34] |
| one day later | forecast_edge / plain | 1% | 845 | 43.6 | 41.2 | +3.82 | [-4.44, +12.04] |
| one day later | forecast_edge / hedged | 0% | 4,525 | 44.5 | 44.5 | +0.02 | [-1.25, +1.32] |
| one day later | forecast_edge / hedged | 1% | 4,525 | 41.2 | 41.1 | -2.04 | [-3.31, -0.76] |
| one day later | rich_cheap_edge / plain | 0% | 838 | 44.4 | 42.0 | +4.61 | [-4.02, +13.28] |
| one day later | rich_cheap_edge / plain | 1% | 838 | 43.3 | 41.1 | +2.37 | [-6.20, +10.95] |
| one day later | rich_cheap_edge / hedged | 0% | 4,525 | 45.0 | 44.9 | +0.05 | [-1.21, +1.38] |
| one day later | rich_cheap_edge / hedged | 1% | 4,525 | 41.6 | 41.6 | -2.02 | [-3.25, -0.70] |

*Ranges resample whole days. Closing prints are last trades, not quotes you can deal at.*
