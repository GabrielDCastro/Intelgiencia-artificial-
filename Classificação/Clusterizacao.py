import pandas as pd
from sklearn.cluster import KMeans #Biblioteca do skLearn que possui os algoritmos para mineração de dados
import matplotlib.pyplot as plt #biblioteca para geração de gráficos
import matplotlib.pyplot as plt2

nameColumns = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Class']
dataFrame = pd.read_csv('iris.csv', names=nameColumns)
print("Quantidade de linhas: %d ; quantidade de colunas: %d" % (len(dataFrame), len(dataFrame.columns)))

#Quando se aplica clusterização, é muito importante saber se os dados que serão utilizados para clusterizar estão em formato float
print(dataFrame.dtypes)
# for col in dataFrame.columns[0:4]:
#   dataFrame[col] = dataFrame[col].astype(float)

print(dataFrame.head(10))

#Vamos remover a coluna classe do dataframe para que o cluster agrupe as iris sem saber a qual classe cada uma pertence
dataFrameCluster = dataFrame.drop(columns=['Class'])
print(dataFrameCluster.head(7))


#Criar objeto que realizará a clusterização
objCluster = KMeans(n_clusters=3)

#Realização do treinamento
print(objCluster.fit(dataFrameCluster))

#Analisar os grupos que foram gerados para ver se o agrupamento realizado foi coerente
dataFrameResultado = objCluster.fit_predict(dataFrameCluster)
print(dataFrameResultado)

#Vamos ver a posição dos centróides
print(objCluster.cluster_centers_)

#Vamos ver a distiancia de cada flor para cada um dos centróides
distancia = objCluster.fit_transform(dataFrameCluster)
print(distancia)

#Vamos ver qual o grupo definido para cada flor
resultado = dataFrame[['Class']].copy()
resultado['grupo']=dataFrameResultado
pd.set_option('display.max_rows',150)
print(resultado)

#Como eu sei quantos clusters deve ter a minha base?
#A abordagem elbow nos auxilia nisso

wcss = [] # variancia
for i in range (1,11): # vai executar 10 interações (i) no log (i de 1 a 10)
  objCluster2 =KMeans(n_clusters=i) # cria um objeto para arupar com a quantidade de grupo i
  objCluster2.fit(dataFrameCluster) # executa o treinamento da base
  print(i,objCluster2.inertia_) # mostra o valor da variância
  wcss.append(objCluster2.inertia_) # adiciona a variância do vetor wcss


plt.plot(range(1,11), wcss)
plt.title("Análise de variância (Elbow)")
plt.xlabel("Número do cluster")
plt.ylabel("Variância (WCSS)")
plt.show()

#Uma vez que criamos, treinamos e validamos o objeto cluster. Vamos pedir para ele incluir novas flores em grupos
data = [
          [4.9,3.2,1.3,0.3], # setosa -> 1
          [7.0,3.2,4.7,1.4], # versicolor -> 2
          [6.3,3.3,5.4,2.2]  # virginica -> 0
      ]
objCluster.predict(data)

#É legal analisar os resultados em gráficos para termos melhor dedução de como os grupos são formados
plt.scatter(dataFrameCluster.iloc[:,0], dataFrameCluster.iloc[:,1], s = 100, c = objCluster.labels_)
plt.scatter(objCluster.cluster_centers_[:, 0], objCluster.cluster_centers_[:, 1], s = 300, c = 'red',label = 'Centroids')
plt.title('Iris Clusters and Centroids')
plt.xlabel('SepalLength')
plt.ylabel('SepalWidth')
plt.legend()

print(plt.show())

plt2.scatter(dataFrameCluster.iloc[:,2], dataFrameCluster.iloc[:,3], s = 100, c = objCluster.labels_)
plt2.scatter(objCluster.cluster_centers_[:, 2], objCluster.cluster_centers_[:, 3], s = 300, c = 'red',label = 'Centroids')
plt2.title('Iris Clusters and Centroids')
plt2.xlabel('PetalLength')
plt2.ylabel('PetalWidth')
plt2.legend()
print(plt2.show())
