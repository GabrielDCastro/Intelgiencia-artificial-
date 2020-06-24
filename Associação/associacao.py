import pandas as pd
import apyori

dataFrame = pd.read_csv("order_data.csv", delimiter = " ", header=None)
print(f"Qtd. Linhas: {len(dataFrame)}; qtd colunas: {len(dataFrame.columns)}")

print(dataFrame.head(20))

"""Uma vez que os dados foram importados, vamos aplicar a tarefa de associação.**
No entanto, o algoritmo Apriori não trabalha com dataframe, ele trabalha com listas.
"""

dataList = []
for row in range (0,20):
    dataList.append([str(dataFrame.values[row,column]) for column in range (0,9)])
print(dataList)

#Agora, vamos aplicar a regra de associação
objAssociacao = apyori.apriori(dataList,min_support=0.4, min_confidence=0.3, min_lift=1.0, min_lenght=2)
result = list(objAssociacao)
len(result)

#Análise dos resultados gerados
ctResultados = len(result)
for i in range(0,ctResultados):
  print(f"A associação {i+1} é: {result[i]}")
  print("#"*40)
