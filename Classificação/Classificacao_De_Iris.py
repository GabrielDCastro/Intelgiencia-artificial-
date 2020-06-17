import pandas as pd
import pydot
from sklearn import tree
import pydotplus
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from IPython.display import Image


nameColumns= ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Class']
dataFrame = pd.read_csv('iris.csv', names = nameColumns)

#Visualizar as prmeiras linhas da tabela
print(dataFrame.head())

#Contar quantas flores tem de cada categoria
dataFrame['Class'].value_counts()

#Gerar gráfico para distribuir
dataFrame['Class'].value_counts().plot(kind='pie')
plt.show()

#Criação de features
dataFrame['SepalArea'] = dataFrame['SepalLength'] * dataFrame['SepalWidth']
dataFrame['PetalArea'] = dataFrame['PetalLength'] * dataFrame['PetalWidth']
print(dataFrame.head())

#Análise de dados estatísticos do data frameround(dataFrame['SepalArea'].mean(),2)
print(round(dataFrame['SepalArea'].mean(),2))
print(dataFrame['SepalArea'].min())
print(dataFrame['SepalArea'].max())
print(dataFrame['SepalArea'].mode())
print(dataFrame['SepalArea'].median())

#Criação demais features
dataFrame['SepalLengthAboveMean'] = dataFrame['SepalLength'] > dataFrame['SepalLength'].mean()
dataFrame['SepalWidthAboveMean'] = dataFrame['SepalWidth'] > dataFrame['SepalWidth'].mean()
dataFrame['PetalLengthAboveMean'] = dataFrame['PetalLength'] > dataFrame['PetalLength'].mean()
dataFrame['PetalWidthAboveMean'] = dataFrame['PetalWidth'] > dataFrame['PetalWidth'].mean()
dataFrame.head()

#Recuperar o nome das colunas que formam as variáveis independentes
features = dataFrame.columns.difference(['Class'])
#features = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'SepalArea','PetalArea', 'SepalLengthAboveMean', 'SepalWidthAboveMean', 'PetalLengthAboveMean', 'PetalWidthAboveMean']
print(features)

#Criar armazenadores para as variáveis indepentes (X) e para a dependente (y)
X = dataFrame[features].values
y = dataFrame['Class'].values
print(X)

#Criar variáveis com valores que não estão entre os do treinamento para averiguar se a árvore foi criada corretamente
# Iris-setosa
sample1 = [1.0, 2.0, 3.5, 1.0, 10.0, 3.5, False, False, False, False]
# Iris-versicolor
sample2 = [5.0, 3.5, 1.3, 0.2, 17.8, 0.2, False, True, False, False]
# Iris-virginica
sample3 = [7.9, 5.0, 2.0, 1.8, 19.7, 9.1, True, False, True, True]

#Criação da árvore de decisão
classifier_dt = DecisionTreeClassifier(random_state=10, criterion='gini',max_depth=3)
classifier_dt.fit(X, y)

#Testar se a árvore de decisão consegue classificar corretamente os exemplos
classifier_dt.predict([sample1, sample2, sample3])

#Exibição da árvore de decisão que foi criada

# Create DOT data
dot_data = tree.export_graphviz(classifier_dt, out_file=None,
                                feature_names=features,
                                class_names=dataFrame.Class.unique())

# Draw graph
graph = pydotplus.graph_from_dot_data(dot_data)

Image(graph.create_png())
