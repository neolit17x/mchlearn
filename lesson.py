import numpy as np

w=1.0

def forward(x):
    return x*w

def loss(x,y):
    y_hat=forward(x)
    return(y_hat-y)**2

x_soat=np.array([1,2,3])
y_baho=np.array([2,4,6])


for x,y in zip(x_soat,y_baho):
    loss(x,y)

    print("x: ",x,"y: ",y,"Loss: ",loss(x,y))