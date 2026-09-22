import numpy as np
import matplotlib.pyplot as plt

x_soat=np.array([1.0,2.0,3.0])
y_baho=np.array([2.0,4.0,6.0])


w_list=[]
mse_list=[]


def forward(x):
    return x*w

def loss(x,y):
    return (forward(x)-y)**2

def grad(x,y):
    return 2*(x*w-y)*x

for w in np.arange(0.0,4.1,0.1):
    print(f"\nw={w:.3f}")
    step=0
    for x_data,y_data in zip(x_soat,y_baho):
        y_hat=forward(x_data)
        loss_data=loss(x_data,y_data)
        step+=loss_data
        print("\t",f"{y_hat:.2f},{loss_data:.2f}")
    print("MSE: ",step/len(x_soat))
    w_list.append(w)
    mse_list.append(step/len(x_soat))

plt.plot(w_list,mse_list)
plt.ylabel("LOSS",color="black")
plt.xlabel("MSE",color="black")

plt.show()

