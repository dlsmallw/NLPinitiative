import pandas as pd
from nlpinitiative.data_preparation import data_import, data_preparation, dataset_normalizer
from nlpinitiative.config import (
    EXTERNAL_DATA_DIR,
    CONV_SCHEMA_DIR
)


# df1 = data_import.import_from_local_source("C:/Users/Daniel/Downloads/dataset.csv")
# print(df1)

srcs = [
    EXTERNAL_DATA_DIR / "intelligence-csd-auth-gr_Ethos-Hate-Speech-Dataset_Ethos_Dataset_Binary.csv",
    EXTERNAL_DATA_DIR / "intelligence-csd-auth-gr_Ethos-Hate-Speech-Dataset_Ethos_Dataset_Multi_Label.csv"
]
conv = CONV_SCHEMA_DIR / "ethos_schema_mapping.json"

# df2 = dataset_normalizer.convert_to_master_schema(srcs, conv, 'ETHOS_dataset_converted')
# print(df2)

dataset = data_preparation.get_dataset_from_file("ETHOS_dataset_converted.csv")
print(dataset)