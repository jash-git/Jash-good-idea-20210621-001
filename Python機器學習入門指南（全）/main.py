# 这是一个简单的demo。使用iris植物的数据，训练iris分类模型，通过模型预测识别品种。
import pandas as pd

# 加载数据集 
data = load_iris()  # sklearn.datasets
df = pd.DataFrame(data.data, columns=data.feature_names)
df.head()


# 使用pandas_profiling库分析数据情况
import pandas_profiling

df.profile_report(title='iris')


# 划分标签y，特征x
y = df['class']
x = df.drop('class', axis=1)


#划分训练集，测试集
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(x, y)

# 模型训练
from xgboost import XGBClassifier

# 选择模型
xgb = XGBClassifier(max_depth=1, n_estimators=1)

xgb.fit(train_x, train_y)

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_curve, auc

def model_metrics(model, x, y, pos_label=2):
    """
    评估函数
    """
    yhat = model.predict(x)
    result = {'accuracy_score':accuracy_score(y, yhat),
              'f1_score_macro': f1_score(y, yhat, average = "macro"),
              'precision':precision_score(y, yhat,average="macro"),
              'recall':recall_score(y, yhat,average="macro")
             }
    return result


# 模型评估结果
print("TRAIN")
print(model_metrics(xgb, train_x, train_y))

print("TEST")
print(model_metrics(xgb, test_x, test_y))


# 模型预测
xgb.predict(test_x)