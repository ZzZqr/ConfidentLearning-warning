# 导入包
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

def readAllData():
    dataSet = pd.read_csv(r'./input_data/label_data.csv', encoding="GBK", low_memory=False)
    dataSetPart1 = dataSet.iloc[:, 2:5].reset_index(drop=True)
    dataSetPart2 = dataSet.iloc[:, 7:70].reset_index(drop=True)

    dfCategory = OneHot(dataSet['category'])
    dfKind = OneHot(dataSet['Kind'])
    label = dataSet['final_label'].reset_index(drop=True)
    # 在resolution上做
    dataSet0 = pd.concat([dfCategory, dfKind, dataSetPart1, dataSetPart2], axis=1).reset_index(drop=True)
    columns = dataSet0.columns
    scaler = StandardScaler(copy=False)
    dataSet0 = pd.DataFrame(scaler.fit_transform(dataSet0), columns=columns)
    dataSet0 = pd.concat([dataSet0, label], axis=1).reset_index(drop=True)
    dataSet0.to_csv("test.csv")
    print(dataSet0)


    return dataSet0

def OneHot(x):
    '''
        功能：one-hot 编码
        传入：需要编码的分类变量
        返回：返回编码后的结果，形式为 dataframe
    '''
    # 通过 LabelEncoder 将分类变量打上数值标签
    lb = LabelEncoder()  # 初始化
    x_pre = lb.fit_transform(x)  # 模型拟合
    x_dict = dict([[i, j] for i, j in zip(x, x_pre)])  # 生成编码字典--> {'收藏': 1, '点赞': 2, '关注': 0}
    x_num = [[x_dict[i]] for i in x]  # 通过 x_dict 将分类变量转为数值型

    # 进行one-hot编码
    enc = OneHotEncoder()  # 初始化
    enc.fit(x_num)  # 模型拟合
    array_data = enc.transform(x_num).toarray()  # one-hot 编码后的结果，二维数组形式

    # 转成 dataframe 形式
    df = pd.DataFrame(array_data)
    inverse_dict = dict([val, key] for key, val in x_dict.items())  # 反转 x_dict 的键、值
    df.columns = [inverse_dict[i] for i in df.columns]  # columns 重命名
    return df

def process(x):
    lb = LabelEncoder()  # 初始化
    x_pre = lb.fit_transform(x)  # 模型拟合
    x_dict = dict([[i, j] for i, j in zip(x, x_pre)])
    x_num = [x_dict[i] for i in x]

    df = pd.DataFrame(x_num)
    inverse_dict = dict([val, key] for key, val in x_dict.items())  # 反转 x_dict 的键、值
    df.columns = ['final_label']  # columns 重命名
    return df
