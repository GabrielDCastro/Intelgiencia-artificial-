import pandas as pd
import matplotlib.pyplot as plt

#Nome das colunas colocamos referente ao que cada valor do csv representa
nameColumns= ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Class']
#Lendo o csv e dando nome as colunas
dataFrame = pd.read_csv('iris.csv', names= nameColumns)
print("Linhas: %d, Colunas: %d" % (len(dataFrame), len(dataFrame.columns)))

#Pega só os 5 primeiros valores
a= dataFrame.head()
print (a)

#mostra em gráfico
dataFrame['Class'].value_counts().plot(kind='pie')
plt.show()

#Exemplos do que se pode fazer com pandas
dataFrame['SepalArea'].mean()   #média
dataFrame['SepalArea'].min()    #mínimo
dataFrame['SepalArea'].max()    #máximo
dataFrame['SepalArea'].mode()   #moda
dataFrame['SepalArea'].median() #mediana


features = dataFrame.columns.difference(['Class'])
X = dataFrame[features].values  #Variáveis dependentes
y = dataFrame['Class'].values   #Variáveis independentes

#Colocando exemplos para serem testados na decisão. Serão esses três exemplos
# Iris-setosa
sample1 = [1.0, 2.0, 3.5, 1.0, 10.0, 3.5, False, False, False, False]
# Iris-versicolor
sample2 = [5.0, 3.5, 1.3, 0.2, 17.8, 0.2, False, True, False, False]
# Iris-virginica
sample3 = [7.9, 5.0, 2.0, 1.8, 19.7, 9.1, True, False, True, True]
