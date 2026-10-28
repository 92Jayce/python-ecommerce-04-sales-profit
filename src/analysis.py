"""第04讲 销售额、退款、客单价与贡献利润
业务问题：卖得多是否就赚得多？
全部使用模拟教学数据。"""
from pathlib import Path
import sys
if "__file__" in globals():
    COURSE_ROOT = Path(__file__).resolve().parents[1]
else:
    COURSE_ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "course_common.py").exists())
if str(COURSE_ROOT) not in sys.path:
    sys.path.insert(0, str(COURSE_ROOT))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from course_common import (ROOT, DATA, AS_OF, BLUE, TEAL, ORANGE, load_tables,
    build_facts, positive_orders, rfm_table, safe_divide, outdir, save_table, save_figure, show)

# %% 付款口径指标：含后续全额退款的已付款订单
t=load_tables(); lines,orders=build_facts(t)
paid=orders.loc[orders.paid_total_cents.gt(0)].copy()
revenue=orders.net_sales_calc.sum(); refunds=orders.refund_total_cents.sum()
spend=t['marketing'].spend_cents.sum()
metrics=pd.DataFrame({'metric':['付款商品额','退款商品额','净商品收入','付款订单数','付款口径净客单价',
    '商品贡献额（扣营销前）','扣营销后贡献额','退款金额率'],
    'value':[paid.paid_total_cents.sum()/100,refunds/100,revenue/100,len(paid),
        safe_divide(revenue,len(paid))/100,orders.contribution_cents.sum()/100,
        (orders.contribution_cents.sum()-spend)/100,safe_divide(refunds,paid.paid_total_cents.sum())],
    'unit':['元','元','元','单','元/付款单','元','元','比例']})
show(metrics,20); save_table(metrics,4,'kpis')

# %% 类目贡献率应按金额加权，不能平均每笔订单利润率
category=lines.groupby('category',as_index=False).agg(net_sales_cents=('net_sales_cents','sum'),
    contribution_cents=('contribution_cents','sum'))
category['contribution_rate']=safe_divide(category.contribution_cents,category.net_sales_cents)
show(category); save_table(category,4,'category_profit')
fig,ax=plt.subplots(figsize=(7,4));ax.bar(category.category,category.contribution_cents/100,color=TEAL)
ax.set(title='Category contribution before marketing',ylabel='CNY');save_figure(fig,4,'category_contribution')
assert revenue==orders.paid_total_cents.sum()-refunds
sessions=len(t['sessions']); conversion=len(paid)/sessions; net_aov=revenue/len(paid)
assert np.isclose(sessions*conversion*net_aov,revenue)
print('经营拆解恒等式：净商品收入 = 会话数 × 会话支付转化率 × 付款口径净客单价（不是因果模型）。')
print('本例是付款口径管理报表；贡献额不等于财务净利润，不含税费、固定成本等。')

# %% 售后观察窗口：最近订单还没有同样长的退款机会
mature=paid.loc[paid.order_date.lt(AS_OF-pd.Timedelta(days=21))]
refund_window=pd.DataFrame([{'scope':'all_paid_as_observed','orders':len(paid),
    'refund_amount_rate':safe_divide(paid.refund_total_cents.sum(),paid.paid_total_cents.sum())},
    {'scope':'at_least_21_days_observed','orders':len(mature),
    'refund_amount_rate':safe_divide(mature.refund_total_cents.sum(),mature.paid_total_cents.sum())}])
show(refund_window);save_table(refund_window,4,'refund_observation_windows')
print('生成模型的退货延迟为5至21天；成熟组与全体的构成也不同，不能把比例差直接当作经营变化。')
