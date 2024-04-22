import pandas as pd

# 데이터 파일을 열 이름 없이 불러오기
df1 = pd.read_csv('data739359.3928.csv', header=None, names=None)
df2 = pd.read_csv('data739359.7362.csv', header=None, names=None)

# 데이터를 열 방향으로 합치기, 열 인덱스 리셋
combined_df = pd.concat([df1, df2], axis=0)

# 결과 확인
column_names  = ['intensity', 'PI', 'Do', 'M_o']
column_names += [f'{5 * i}ms' for i in range(0, 61)]
combined_df.columns = column_names
print(combined_df.head())

combined_df.to_csv('raw_data.csv', index=False)  # 새 파일로 저장