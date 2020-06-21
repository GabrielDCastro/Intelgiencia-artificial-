import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error


dataFrame = pd.read_csv('kc_house_data.csv')#Importar o CSV com os dados das casas
print("Linhas: %d, Colunas: %d" % (len(dataFrame), len(dataFrame.columns)))

#Verificar os nomes das colunas do arquivo
print(dataFrame.columns)

#Primeiras linhas tabela
print(dataFrame.head())

#Tipos de cada dado do dataframe
print(dataFrame.dtypes)

#Construir gráfico para estudarmos a relação entre o preço e algumas variáveis
sns.pairplot(dataFrame,
             x_vars=["sqft_living","bedrooms","bathrooms","sqft_above","sqft_basement","sqft_lot"],
             y_vars=["price"])

#Gerar uma rgressão linear simples. Para isso, vamos fazer igual nos outros algoritmos que estudamos como o de clusterização, por exemplo
X = dataFrame[["sqft_living"]]
y = dataFrame[["price"]]

#Agora que somos Jedis em Data Science, vamos utilizar boas práticas. No caso, separaremos os dados para treinamento e para teste
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1)
print("Treinamento: %d, Teste: %d" % (len(X_train), len(X_test)))

#Agora, com base nos dados que separamos para treinamento e para teste, vamos aplicar modelo.
#Para isso, vamos iniciar com Regressão Linear simples, que possui apenas uma variável independente.
objRL = LinearRegression() #criação do objeto que efetuará a regressão(o treinamento)
print(objRL)

#Uma vez que o objeto está criado, vamos efetuaro treinamento.
objRL.fit(X_train,y_train) # solicitação para o objeto efetuar o treinamento

#Agora vamos executar um teste para prever os valores das variáveis de teste
y_pred = objRL.predict(X_test)
#6478-2910 / 410000.0
#771645.63372716
data = [
          [1180], # 221900.0
          [2570], #538000.0
          [770] #180000.0
        ]
print(objRL.predict(data))

#Agora que a regresão já foi construída, vamos mostrar o gráfico
plt.scatter(X_test,y_test, color='blue')
plt.plot(X_test,y_pred, color='red')
plt.show()

#Antes de partir para a regressão com mais de uma variável independente, vamos medir a acurácia do nosso modelo
print(f"R2 socre: {r2_score(y_test,y_pred)}") # quanto mais próximo de 1, melhor
print(f"MSE socre: {mean_squared_error(y_test,y_pred)}") # quanto mais próximo de zero, melhor

#Vamos aplicar uma regresão linear  om várias variáveis independentes
#Primeiro, vamos utilizar duas variáveis independentes
X2 = dataFrame[['sqft_living','sqft_lot']]
X2_train,X2_test,y2_train,y2_test = train_test_split(X2,y,test_size=0.3)
objRL2 = LinearRegression()
objRL2.fit(X2_train,y2_train)
y2_pred = objRL2.predict(X2_test)
print(f"R2 socre: {r2_score(y2_test,y2_pred)}") # quanto mais próximo de 1, melhor
print(f"MSE socre: {mean_squared_error(y2_test,y2_pred)}") # quanto mais próximo de zero, melhor

#Agora vamos utilizar todas as variáveis independentes possíveis
X3 = dataFrame.drop(columns=["id","date","price","zipcode"])
X3_train, X3_test, y3_train, y3_test = train_test_split(X3,y,test_size=0.1)
objRL3 = LinearRegression()
objRL3.fit(X3_train,y3_train)
y3_pred = objRL3.predict(X3_test)
print(f"R2 socre: {r2_score(y3_test,y3_pred)}") # quanto mais próximo de 1, melhor
print(f"MSE socre: {mean_squared_error(y3_test,y3_pred)}") # quanto mais próximo de zero, melhor

#Vamos gerar um gráfico de barras com a comparação do valor estimado para o valor real
y3_test.head()
y3_test["predict"]=y3_pred
dfTest = y3_test.head(30)
dfTest.head()
dfTest.plot(kind="bar")
plt.show()
