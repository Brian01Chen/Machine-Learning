import pandas as pd
from collections import Counter
from ggplot import *
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.preprocessing import LabelEncoder
from sklearn_pandas import DataFrameMapper


anaylse_columns = [1,3,6,9,16]
df = pd.read_csv(r'C:\Users\IBM_ADMIN\Desktop\Tools\shanshan\approve\foreign_invest2.csv',
                           usecols = anaylse_columns,
                           sep = '\t',
                           encoding = 'gbk')



for i in df.columns:
    if df[i].dtype == 'object':
        df[i].fillna('Missing',inplace=True)
    if i in ('STATUS','PF_STATUS'):
        df[i].fillna(10,inplace=True)
    else:
        df[i].fillna(0, inplace=True)

acount = df['ECONOMIC_TYPE'].value_counts()


consolas = font_manager.FontProperties(fname='C:\Windows\Fonts\consolas.ttf')
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
acount.plot()
plt.title('Column Analyse',color='#443456')
plt.show()

p = ggplot(df,aes(x = u'ECONOMIC_TYPE',y = 'PF_CL_NUMBER')) \
    + geom_bar() \
    + xlab("Etype") + ylab('PF_CL_NUMBER')
print (p)

mapper = DataFrameMapper([('ECONOMIC_TYPE',LabelEncoder()),
                          ('T_invest', LabelEncoder()),
                          ('PF_CL_NUMBER', LabelEncoder()),
                          ('STATUS', LabelEncoder()),
                          ('PF_STATUS', LabelEncoder())],
                         df_out=True)

df = mapper.fit_transform(df.copy()[:])
print (df)
#classCount = Counter(data[0] for data in df.values if data[2] != '')


