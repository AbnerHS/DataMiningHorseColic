import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from names import cols, important

def main():
  input_file = 'dataset/horse-colic-clear.data'
  df = pd.read_csv(input_file, names=cols, usecols=important)

  #Medidas Centrais
  medias = ['rectal temperature', 'pulse', 'respiratory hate']
  modas = ['surgery','age','pain','abdominal distension']
  print("--------Tendencia Central--------")
  for medida in medias + modas:
    coluna = df.loc[:,medida]
    print(medida)
    if(medida in medias):
      print("Media: ",coluna.mean())
    if(medida in modas):
      print("Moda: ",coluna.mode()[0])
    print("--------------------------------")

  #Medidas de Dispersão
  print("\n\n-----------Dispersão-----------")
  for medida in medias + modas:
    coluna = df.loc[:, medida]
    print(medida)
    print("Amplitude: ", amplitude(coluna))
    print("Desvio Padrão: ", coluna.std())
    print("Variância: ", coluna.var())
    print("Coeficiente de Variação: ", coeficiente_variacao(coluna))
    print("--------------------------------")


  #Medidas de Posição Relativa
  print("\n\n--------Posição Relativa--------")
  for medida in medias + modas:
    coluna = df.loc[:, medida]
    print(medida)
    for Q in range(0, 100, 25):
      print("Q",int(Q/25),": ", coluna.quantile(Q/100))
    print("--------------------------------")

  #Medidas de Associação
  plt.figure()
  sns.heatmap(df.corr(), annot=True)
  plt.show()
  

def amplitude(coluna):
  return coluna.max() - coluna.min()

def coeficiente_variacao(coluna):
  cv = lambda x: np.std(x, ddof=1) / np.mean(x) * 100
  return cv(coluna)

if __name__ == "__main__":
  main()


  #surgery - moda
  #age - moda
  #rectal temperature - media
  #respiratory hate - media
  #pain - moda
  #abdominal distension - moda