import pandas as pd
import numpy as np
from names import names, cols  

def main():
  input_file = 'dataset/horse-colic.data'
  output_file = 'dataset/horse-colic-clear.data'
  df = pd.read_csv(input_file, names=names, na_values='?', sep=' ', usecols=cols)
  df['age'] = df['age'].map({1: 1, 9: 2})
  
  print("Primeiras 15 linhas")
  print(df.head(15))
  print("\n")

  print("Informações gerais")
  print(df.info())
  print("\n")

  print("Descrição")
  print(df.describe())
  print("\n")

  print("Dados Faltantes")
  print(df.isnull().sum())
  print("\n")

  columns_missing_values = df.columns[df.isnull().any()]
  print(columns_missing_values)

  df = df.loc[df['abdominal distension'].notnull()]
  df = df.loc[df['surgery'].notnull()]
  df = df.loc[df['outcome'].notnull()]

  for c in columns_missing_values:
    method = 'mode'
    media = ['rectal temperature','pulse','packed cell volume','total protein','respiratory hate']
    if(c in media):
      method = 'mean'
    updateMissingValues(df, c, method)

  df.to_csv(output_file, header=False, index=False)

def updateMissingValues(df, column, method, number=0):
  if method == 'number':
    # Substituindo valores ausentes por um número
    df[column].fillna(number, inplace=True)
  elif method == 'median':
    # Substituindo valores ausentes pela mediana 
    median = df['Density'].median()
    df[column].fillna(median, inplace=True)
  elif method == 'mean':
    # Substituindo valores ausentes pela média
    mean = df[column].mean()
    df[column].fillna(mean, inplace=True)
  elif method == 'mode':
    # Substituindo valores ausentes pela moda
    mode = df[column].mode()[0]
    df[column].fillna(mode, inplace=True)

if __name__ == '__main__':
  main()

