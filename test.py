import pandas as pd
from nlpinitiative.data_preparation import data_import, data_preparation


df1 = data_import.import_from_local_source("C:/Users/Daniel/Downloads/dataset.csv")
print(df1)

df2 = data_import.import_from_ext_source("https://github.com/albanyan/counterhate_reply/blob/main/Data/dataset.csv")
print(df2)
