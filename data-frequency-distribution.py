import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from names import important

def main():
  input_file = 'dataset/horse-colic-clear.data'

  df = pd.read_csv(input_file, names=important)
  _, ax = plt.subplots(1, 2, constrained_layout=True)

  # Distribuição de frequência de abdominal distension
  abdominal_distension = df.loc[:, 'abdominal distension']
  max = abdominal_distension.max()
  min = abdominal_distension.min()
  x = np.arange(min, max+1)
  y, _ = np.histogram(abdominal_distension, bins=np.arange(max+1))
  ax[0].bar(x, y)
  ax[0].set_title("Abdominal Distension")
  
  # Distribuição de frequência de surgery
  surgery = df.loc[:, 'surgery']
  y, _ = np.histogram(surgery, bins=np.arange(4))
  ax[1].pie(y[1:], labels=['Com', 'Sem'], colors=['g','r'], autopct='%.2f%%', explode=(0, 0.05), textprops={'color':'w', 'fontsize':14})
  ax[1].set_title("Surgery")

  # # Distribuição de frequência de pain
  # pain = df.loc[:, 'pain']
  # max = pain.max()
  # min = pain.min()
  # x = np.arange(min, max+1)
  # y, _ = np.histogram(pain, bins=np.arange(max+1))
  # ax[0].bar(x, y, color='orange')
  # ax[0].set_title("Pain")

  # # Distribuição de frequência de respiratory hate
  # sns.scatterplot(x = 'surgical lesion', y = 'respiratory hate', data = df)
  # ax[1].set_title("Respiratory Hate / Surgical Lesion")

  plt.show()
if __name__ == "__main__":
  main()