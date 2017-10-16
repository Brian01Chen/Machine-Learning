import pandas as pd
import math
import datetime

def deal_time(filename,sheetnum,cols):
    df = pd.read_excel(filename,sheetname=sheetnum,parse_cols=cols)

    for col in df.columns:
        #日期格式预处理
        df[col].replace(r'\s-','-',regex=True,inplace=True)
        df[col].replace(r'[0-9]{9}','000',regex=True,inplace=True)
        df[col].replace({'上午':'AM','下午':'PM'},regex=True,inplace=True)

    all_hours = []
    for rownum in range(len(df)):
        try:
            t1 = datetime.datetime.strptime(df.iloc[:,0][rownum],'%d-%m月-%y %I.%M.%S.%f %p')
            t2 = datetime.datetime.strptime(df.iloc[:,1][rownum],'%d-%m月-%y %I.%M.%S.%f %p')
            hours = abs(round((t2-t1).total_seconds()/3600,2))
            days = abs((t2-t1).days + 1)
            week,leftover = divmod(days,7)
            end_date_ofweek = t2.weekday()+1
            if week > 1:
                hours = hours - 48*week
            if week == 0 and end_date_ofweek < leftover:
                hours = hours - 48

        except AttributeError:
            hours = 0
        print(math.floor(hours))
        all_hours.append(math.floor(hours))
    return all_hours


if __name__ == '__main__':
    #读取文件的路径
    filename = r'C:\Users\IBM_ADMIN\Desktop\Tools\shanshan\approve\水运\水运办理时间效率.xlsx'
    #读取sheet页的number，从0开始计算，0表示第1页，1表示第2页
    sheetnum = 0
    #读取sheet页中的字段，从0开始计算，比如[2,3]表示第3列、第4列
    cols = [5,6]

    all_hours = deal_time(filename,sheetnum,cols)
    print (all_hours)