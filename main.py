from matplotlib import pyplot as plt
import numpy as np
sum_error = 0
learning_rate = 0.1
iterations= 100000
data =np.array([[1,2,3,4],
                [1,3,4,5],
                [1,4,5,6],
                [1,5,6,7],
                [1,6,7,8],
                [1,7,8,9]])


Qs = np.random.rand(4,1)
qs_for_hypot = np.reshape(Qs,(1,4))
print(qs_for_hypot)
print("PARAMETERS",Qs)
outputs = np.array([10,13,16,19,22,25])



def hypothesis(x):
    return(np.dot(qs_for_hypot,x))


for iter in range (iterations):
    for i in range (len(data)):
        for j in range(len(Qs)):
            squared_error = ((hypothesis(data[i]) - outputs[i]) ** 2)
            Qs[j] = Qs[j]-learning_rate*1/6*(((hypothesis(data[i])-outputs[i])*data[i][j]))
            print("SQUARED_ERROR",squared_error)
    if (squared_error==0):
        break

print("PARAMETER",Qs)
print("PREDICT",hypothesis([1,2,5,5]))
