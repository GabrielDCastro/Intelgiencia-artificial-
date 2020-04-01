import pandas as pd
nameColumns= ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Class']
dataFrame = pd.read_csv('iris.csv', names= nameColumns)
print("Linhas: %d, Colunas: %d" % (len(dataFrame), len(dataFrame.columns)))
a= dataFrame.head()
print (a)
