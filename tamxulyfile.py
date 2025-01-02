import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_excel('train_data_raw.xlsx')

sample_data , _ = train_test_split(df, test_size=0.9, random_state=None)

sample_data.to_excel("minitrain4.xlsx", index=False)

