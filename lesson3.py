import numpy as np
import matplotlib.pyplot as plt
x_soat=np.array([1.0,2.0,3.0])
y_baho=np.array([2.0,4.0,6.0])

w_list=[]
mse_list=[]

w=1.0

a=0.01
def forwrd(x):
    return x*w
def loss(x,y):
    y_hat=forwrd(x)
    return (y_hat-y)**2
def grad(x,y):
    yp=forwrd(x)
    return 2*x*(yp-y)
print("Bashorat (traning dan avval)","4 soat o'qishda:",forwrd(4))  

step=0
for epoch in range(10):
    for x_hb_q,y_hb_q in zip(x_soat,y_baho):
        gd=grad(x_hb_q,y_hb_q)
        l=loss(x_hb_q,y_hb_q)
        step+=l
        w=w-a*gd
        print("\t grad: ",x_hb_q,y_hb_q,round(gd,2))
        l=loss(x_hb_q,y_hb_q)
    print("progress: ",epoch,"w=",round(w,2),"Loss=",round(l,2))
    print("MSE:",step/len(x_soat))
    w_list.append(w)
    mse_list.append(step/len(x_soat))
print("Bashorat (traning dan keyin)","4 soat o'qishda:",forwrd(4))


plt.plot(w_list)
plt.xlabel("Epoch")
plt.ylabel("w")
plt.title("How w learns")

plt.show()

plt.plot(mse_list)
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("How error decreases")

plt.show()