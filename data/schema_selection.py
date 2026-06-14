from schema_selected import *
import pandas as pd
import numpy as np
import os

selected_schema = [index_cols, target_cols, binary, numeric, categorical]

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":

    raw_data = pd.read_parquet(os.path.join(PROJECT_DIR, 'masterdapt_merged.parquet'), engine='pyarrow')

    index_data = raw_data[index_cols]
    target_data = raw_data[target_cols]
    binary_data = raw_data[binary]
    numeric_data = raw_data[numeric]
    categorical_data = raw_data[categorical]

    final_df = pd.concat([index_data, target_data, binary_data, numeric_data, categorical_data], axis=1)

    final_df.to_parquet(os.path.join(PROJECT_DIR, 'selected_schema_data.parquet'), engine='pyarrow', index=False)

    print("Selected data shape:", final_df.shape)