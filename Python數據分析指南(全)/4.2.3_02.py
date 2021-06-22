# 利用pandas_profiling一健生成数据情况（EDA）报告:数据描述、缺失、相关性等情况
import pandas_profiling as pp
report = pp.ProfileReport(df)
report