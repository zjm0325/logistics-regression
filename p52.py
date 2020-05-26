import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('/Users/zjm/Playground/logistics-regression/creditcard.csv')
count_classes = pd.value_counts(data['Class'], sort=True).sort_index()
count_classes.plot(kind='bar')
plt.title('Fraud class histogram')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.show()

from sklearn.preprocessing import StandardScaler

data['normAmount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1,1))  # 对差异较大的列进行标准化操作
data = data.drop(['Time', 'Amount'],axis=1)
data.head()

X = data.loc[:, data.columns != 'Class']
y = data.loc[:, data.columns == 'Class']

# 下采样
number_records_fraud = len(data[data['Class'] == 1])  # 计算异常样本的数目
fraud_indices = np.array(data[data.Class == 1].index)  # 取出异常样本所在位置的索引值，<class 'numpy.ndarray'>
normal_indices = data[data.Class == 0].index  # 取出正常样本所在位置的索引值#<class 'pandas.core.indexes.numeric.Int64Index'>

# 从正常样本中随机取出异常样本数目的样本
random_normal_indices = np.random.choice(normal_indices, number_records_fraud, replace=False)
random_normal_indices = np.array(random_normal_indices)  # 转换成为ndarray类型，虽然我也不知道为什么中间有个过度的类型

under_sample_indices = np.concatenate([fraud_indices, random_normal_indices])  # 正常样本与异常样本所在索引
under_sample_data = data.iloc[under_sample_indices, :]  # 通过索引取出训练样本
X_under_sample = under_sample_data.iloc[:, under_sample_data.columns != 'Class']  # train_data
Y_under_sample = under_sample_data.iloc[:, under_sample_data.columns == 'Class']  # train_target

# 构造训练集和测试集
from sklearn.model_selection import train_test_split

# 为了对比下采样与不处理构建模型的差异
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)  # 测试数据占30%，指定随机方式有利于复现
X_train_undersample, X_tes_undersamplet, y_train_undersample, y_test_undersample = train_test_split(X_under_sample,
                                                                                                    Y_under_sample,
                                                                                                    test_size=0.3,
                                                                                                    random_state=0)

# 使用逻辑回归进行建模操作
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import confusion_matrix, recall_score, classification_report

def print_kfold_scores(x_train_data, y_train_data):
    fold = KFold(5, shuffle=False)  # 数据数目，交叉折叠次数5次，不进行洗牌

    c_param_range = [0.01, 0.1, 1, 10, 100]  # 待选的模型参数
    # 新建一个dataFrame类型（csv，就像一个表格），列名是参数取值、平均召回率
    result_table = pd.DataFrame(index=range(len(c_param_range), 2), columns=['C_parameter', 'Mean recall score'])
    result_table['C_parameter'] = c_param_range

    j = 0
    for c_param in c_param_range:  # 将待选参数一个一个进行模型训练，并分别计算对应的召回率
        print('==========================')
        print('C parameter:', c_param)
        # print('C parameter:' + str(c_param))
        print('--------------------------')

        recall_accs = []
        for iteration, indices in enumerate(fold.split(x_train_data), start=1):  # 和交叉验证有关
            lr = LogisticRegression(C=c_param, penalty='l1', solver='liblinear')  # 实例化逻辑回归模型
            lr.fit(x_train_data.iloc[indices[0], :], y_train_data.iloc[indices[0], :].values.ravel())  # 将数据带入训练
            y_pred_undersample = lr.predict(x_train_data.iloc[indices[1], :].values)  # 预测结果
            recall_acc = recall_score(y_train_data.iloc[indices[1], :].values, y_pred_undersample)  # 召回率
            recall_accs.append(recall_acc)
            print('Iteration', iteration, ': recall score =', recall_acc)  # 交叉验证的折叠次数为5，有5次小结果

        result_table.loc[j, 'Mean recall score'] = np.mean(recall_accs)  # 计算该参数对应的平均召回率
        j += 1
        print('')
        print('Mean recall score', np.mean(recall_accs))
        print('')

    best_c = result_table.iloc[result_table['Mean recall score'].astype('float64').idxmax()]['C_parameter']

    print('**************************************')
    print('best model to choose from cross validation is with C parameter = ', best_c)
    print('**************************************')

    return best_c


best_c = print_kfold_scores(X_train_undersample, y_train_undersample)


