##41714058 滕岳  Project1
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

#输入
Header_i=["Date","pnl","$Long","$Short","$Total","TradeSh","TradeValue","#Long","#Short","IC"]
df=pd.read_table('C:/2020/Project1/5DR_Adjust_COMBO1500.txt',sep=" ",names=Header_i)
#将NaN置换为0
df.fillna(value=0)
#输出
Header_o=["from","to","$long","$short","Acpnl","ret","tvr","sharp","bpmg","AvgIC","dd%","ddsta","ddend","pwin","dall","ddup","mmup","#long","#short"]
output=pd.DataFrame(np.zeros((6,len(Header_o))),index=None,columns=Header_o)## initialize database
#返回最大回撤以及数列对应的位置
def maxdrawdown(arr):
	i = np.argmax(np.maximum.accumulate(arr) - arr) # end of the period
	j = np.argmax(arr[:i]) # start of period
	return (1-arr[i]/arr[j])
def maxdrawdown_sp(arr):
	i = np.argmax(np.maximum.accumulate(arr) - arr) # end of the period
	j = np.argmax(arr[:i]) # start of period
	return (j+1)
def maxdrawdown_ep(arr):
	i = np.argmax(np.maximum.accumulate(arr) - arr) # end of the period
	j = np.argmax(arr[:i]) # start of period
	return (i)

for i in range(6):
    #循环分别得到2012-01-01至2012-12-31...2017-01-01至2017-06-30的相应dataframe
    start_date=i*10000+20120101
    end_date=i*10000+20121231

    tf=df[(df["Date"]>=start_date) & (df["Date"]<=end_date)]

    output.loc[i,"from"]=tf.head(1).iloc[-1,0]
    output.loc[i,"to"]=tf.iloc[-1,0]
    output.loc[i,"$long"]=tf["$Long"].mean()
    output.loc[i,"$short"]=tf["$Short"].mean()*(-1)
    output.loc[i,"Acpnl"]=tf["pnl"].mean()

    daily_return=(tf["pnl"]/tf["$Total"]).mean()
    output.loc[i,"ret"]=daily_return*252
    output.loc[i,"tvr"]=(tf["TradeValue"]/tf["$Total"]).mean()
    output.loc[i,"sharp"]=(tf["pnl"]/tf["$Total"]).mean()/((tf["pnl"]/tf["$Total"]).std())
    output.loc[i,"bpmg"]=tf["pnl"].sum()*1e4/tf["TradeValue"].sum()
    output.loc[i,"AvgIC"]=tf["IC"].mean()
    output.loc[i, "#long"] = tf["#Long"].mean()
    output.loc[i, "#short"] = tf["#Short"].mean()
    #累加得到实际价格，以求得最大回撤

    cum_pnl= []
    b = 0
    for j in range(len(tf["pnl"])):
        b += tf["pnl"].iloc[j]
        cum_pnl.append(b)

    output.loc[i,"dd%"]=maxdrawdown(cum_pnl)
    output.loc[i,"ddsta"]=tf.head(maxdrawdown_sp(cum_pnl)+1).iloc[-1,0]
    output.loc[i,"ddend"]=tf.head(maxdrawdown_ep(cum_pnl)+1).iloc[-1,0]
    output.loc[i,"ddup"]=len(tf[tf["pnl"]>0])

    output.loc[i,"dall"] = len(tf["Date"])
    output.loc[i,"pwin"]=output.loc[i,"ddup"]/ output.loc[i,"dall"]

    count=0
#该循环为得到每阶段的pnl为正的月份个数
    for t in range(12):
        month_sdate=start_date+t*100
        month_edate=end_date-(11-t)*100
        ttf=df[(df["Date"]>=month_sdate) & (df["Date"]<=month_edate)]
        if (ttf["pnl"].sum()>0):
            count +=1

    output.loc[i,"mmup"] = count


##2012-2017时期的总指标
output.loc[6, "from"] = df.iloc[0, 0]
output.loc[6, "to"] = df.iloc[-1, 0]
output.loc[6, "$long"] = df["$Long"].mean()
output.loc[6, "$short"] = df["$Short"].mean() * (-1)
output.loc[6, "Acpnl"] = df["pnl"].mean()

daily_return = (df["pnl"] / df["$Total"]).mean()
output.loc[6, "ret"] = daily_return * 252
output.loc[6, "tvr"] = (df["TradeValue"] / df["$Total"]).mean()
output.loc[6, "sharp"] = (df["pnl"] / df["$Total"]).mean() / ((df["pnl"] / df["$Total"]).std())
output.loc[6, "bpmg"] = df["pnl"].sum() * 1e4 / df["TradeValue"].sum()
output.loc[6, "AvgIC"] = df["IC"].mean()
output.loc[6, "#long"] = df["#Long"].mean()
output.loc[6, "#short"] = df["#Short"].mean()
# 累加得到实际价格，以求得最大回撤

cum_pnl1 = []
c=0

for t in range(len(df["pnl"])):
    c += df["pnl"].iloc[t]
    cum_pnl1.append(c)

output.loc[6, "dd%"] = maxdrawdown(cum_pnl1)
output.loc[6, "ddsta"] = df.head(maxdrawdown_sp(cum_pnl1) + 1).iloc[-1, 0]
output.loc[6, "ddend"] = df.head(maxdrawdown_ep(cum_pnl1) + 1).iloc[-1, 0]
output.loc[6, "ddup"] = len(df[df["pnl"] > 0])

output.loc[6, "dall"] = len(df["Date"])
output.loc[6, "pwin"] = output.loc[6, "ddup"] / output.loc[6, "dall"]
output.loc[6, "mmup"] = output["mmup"].sum()

#显示所有列
pd.set_option('display.max_columns', 100)

print(output)

#画出 2012-2017 年化收益率 夏普比率 最大回撤率 各年的折线图
time = list(str(i)[0:4] for i in output["from"])[:-1]
ret = list(i for i in output["ret"])[:-1]
sharp = list(i for i in output["sharp"])[:-1]
md = list(i for i in output["dd%"])[:-1]


fig = plt.figure(figsize=(22.5,8))

ax1 = fig.add_subplot(3,1,1)
ax1.set_xlabel('time')
ax1.set_ylabel('annualized return')
plt.plot(time,ret,color = 'blue',marker = 'o')

ax2 = fig.add_subplot(3,1,2)
ax2.set_xlabel('time')
ax2.set_ylabel('sharpe ratio')
plt.plot(time,sharp,color = 'gold',marker = 'o')


ax3 = fig.add_subplot(3,1,3)
ax3.set_xlabel('time')
ax3.set_ylabel('maxdrawdown')
plt.plot(time,md,color = 'violet',marker = 'o')

plt.show()


