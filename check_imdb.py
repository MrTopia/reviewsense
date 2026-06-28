import pandas as pd
df = pd.read_csv('dataset/IMDB_Dataset.csv')
print(df.shape)
print(df.head())
print(df.columns.tolist())
print(df['sentiment'].value_counts())