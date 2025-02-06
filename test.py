import pandas as pd
from nlpinitiative.data_preparation import data_import, data_preparation


# df1 = data_import.import_from_local_source("C:/Users/Daniel/Downloads/dataset.csv")
# print(df1)

df2 = data_import.import_from_ext_source("https://github.com/intelligence-csd-auth-gr/Ethos-Hate-Speech-Dataset/blob/master/ethos/ethos_data/Ethos_Dataset_Multi_Label.csv")
print(df2)
