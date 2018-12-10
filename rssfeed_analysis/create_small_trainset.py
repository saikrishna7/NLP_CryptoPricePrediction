import pandas as pd

df_train = pd.read_csv("data/clean_train.csv")
print(df_train.head())
subset_train = df_train[:500]
subset_train.to_csv("data/train_subset.csv", sep=',', encoding='utf-8', index=False)

df_test = pd.read_csv("data/clean_test.csv")
print(df_test.head())
subset_test = df_test[:100]
subset_test.to_csv("data/test_subset.csv", sep=',', encoding='utf-8', index=False)