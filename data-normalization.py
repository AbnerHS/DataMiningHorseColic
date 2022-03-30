import pandas as pd
from names import cols, important
from sklearn.preprocessing import MinMaxScaler

def main():
  input_file = 'dataset/horse-colic-clear.data'
  output_file = 'dataset/horse-colic-normalized.data'
  target = important[16]
  features = important[:16]
  df = pd.read_csv(input_file, names=cols, usecols=important)
  
  #Normalização Min-Max
  x = df.loc[:, features].values
  x_minmax = MinMaxScaler().fit_transform(x)
  normalizedDf = pd.DataFrame(data=x_minmax, columns=features)
  normalizedDf = pd.concat([normalizedDf, df[[target]]], axis=1)
  print(normalizedDf.info())
  print(normalizedDf.describe())
  print(normalizedDf.head(15))
  
  normalizedDf.to_csv(output_file, header=False, index=False)

if __name__ == "__main__":
  main()