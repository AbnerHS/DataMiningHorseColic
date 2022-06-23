import pandas as pd
from sklearn.decomposition import PCA
from names import important
import matplotlib.pyplot as plt

def main():
  input_file = 'dataset/horse-colic-normalized.data'
  target = important[15]
  features = important[:15]
  df = pd.read_csv(input_file, names=important)

  x = df.loc[:, features].values
    
  #PCA projection
  pca = PCA()
  principalComponents = pca.fit_transform(x)

  principalDf = pd.DataFrame(data=principalComponents[:,0:2], 
                            columns=['principal component 1','principal component 2'])
  finalDf = pd.concat([principalDf, df[target]], axis=1)

  print(finalDf.info())
  print(finalDf.describe())
  print(finalDf.head(15))

  VisualizePcaProjection(finalDf, target)

def VisualizePcaProjection(finalDf, targetColumn):
    fig = plt.figure(figsize = (8,8))
    ax = plt.subplot(111)
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('2 component PCA', fontsize = 20)
    targets = [1, 2]
    legend = ['Cirúrgico', 'Não Cirúrgico']
    colors = ['r', 'g']
    for target, color in zip(targets,colors):
        indicesToKeep = finalDf[targetColumn] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1'],
                   finalDf.loc[indicesToKeep, 'principal component 2'],
                   c = color, s = 50)
    ax.legend(legend)
    ax.grid()
    plt.show()

if __name__ == '__main__':
  main()