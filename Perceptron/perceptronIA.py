#Algoritmo para treinamento de perceptron
import numpy as np

tamAmostra = 6
peso = np.array([113, 122, 107, 98, 115, 120])
pH = np.array([6.8, 4.7, 5.2, 3.6, 2.9, 4.2])
print (peso)
print (pH)
y = np.array([-1,1,-1,-1,1,1])
X = np.vstack((peso,pH))
print(X)

#Vimos que nos algoritmos que desenvolvem um perceptron definem o valor do Bias, que é 1
bias = 1
#Definir a taxa de aprendizado
txAprendizado = 0.1
#Definir os pesos iniciais.
w = np.zeros([3])
print(w)
#Vamos criar um array para armazenar os erros.
e = np.zeros([6])
print(e)
def funcaoAtivacao(u):
  if u < 0.0:
    return (-1) #maçã
  else:
    return (1) #laranja
#numEpocas = 64000 # com tx=0.1 define a quantidade de ciclos de treinamento
numEpocas = 64000 # define a quantidade de ciclos de treinamento
for i in range(numEpocas):
  for j in range(tamAmostra):
      # Criação de um array que inclui o Bias junto com os valores de X da amostra j
      Xb = np.hstack((bias,X[:,j]))
      #print("Array com bias:"+str(Xb))
      u = np.dot(w,Xb) #multiplica cada peso com cada uma das colunas
      #print("Soma dos pesos vezes as entradas:"+str(u))
      Yr = funcaoAtivacao(u) # calcula a saáda do perceptron. Se -1, é Maçã, se 1, Laranja
      #print("Valor que o perceptron sugeriu:" + str(Yr))
      e[j] = y[j] - Yr
      #print("Formula do erros. Era para ser: " + str(y[j]) +", mas sugeriu: "+ str(Yr))
      #print("Erro calculado:" + str(e[j]))
      w = w +txAprendizado*e[j]*Xb
      #print("Pesos:"+str(w))

print("Vetor de erros (e):" + str(e))
print(w)

