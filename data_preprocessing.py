import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('./raw data/raw_data.csv', header=None)
df.drop(index=0, inplace=True)

pre_data = []

for i in range(df.shape[0]):
    intensity = df.iloc[i, 0]
    PI = df.iloc[i, 1]
    Do = df.iloc[i, 2]
    for j in range(4, df.shape[1]):
        exposure_time = j - 4
        cured_height = df.iloc[i, j]
        
        pre_data.append((PI, Do, intensity, exposure_time * 5, cured_height))

data = pd.DataFrame(pre_data, columns=['PI', 'Do', 'intensity', 'exposure_time', 'cured_height'])
data.to_csv('data_origin_scale.csv', index=False)

scaler = MinMaxScaler()
data_scaled = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
data_scaled.to_csv('data_minmax_scale.csv', index=False)

print(data.head())
print(data_scaled.head())

scalers = {}
for col in data.columns:
    scaler = MinMaxScaler()
    data[col] = scaler.fit_transform(data[[col]])
    scalers[col] = scaler

with open('./scalers.pkl', 'wb') as f:
    pickle.dump(scalers, f)

# data.to_csv('data_preprocessed.csv', index=False)