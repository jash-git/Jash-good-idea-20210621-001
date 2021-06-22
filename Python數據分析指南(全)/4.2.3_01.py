import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')


# 读取数据
df = pd.read_csv('tmall_order_report.csv')
df.head()