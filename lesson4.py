import numpy as np
import matplotlib.pyplot as plt
x_soat=np.array([1.0,2.0,3.0])
y_baho=np.array([2.0,4.0,6.0])

mse_list=[]
w_list=[]


w=1.0 #weight neuron 

a=0.01 #Learning rate

def forwrd(x):
    return x*w

def loss(x,y):
    y_hat=forwrd(x)
    return (y_hat-y)**2

def grad(x,y):
    y_hat=forwrd(x)
    return 2*x*(y_hat-y)
def upt(w,a,gd):
    return  w-a*gd

print("Bashorat (traning dan avval)","4 soat o'qishda:",forwrd(4))  
l_umum=0
for epoch in range(12):
    print("W: ",w)
    for x_new,y_new in zip(x_soat,y_baho):

        gd=grad(x_new,y_new) #Gradient
        w=upt(w,a,gd)#update

        print("x: ",x_new,"y:",y_new,"\t gradient: ",round(gd,2)) #This item show x_data y_data and gradent

        l=loss(x_new,y_new)
        l_umum+=l

        w_list.append(w)
        mse_list.append(l_umum/len(x_soat))
    print("progress:",epoch," ","w=",round(w,2),"Loss=",round(l,2))
print("Bashorat (traning dan keyin)","4 soat o'qishda:",forwrd(4))



plt.plot(w_list,mse_list)
plt.ylabel("LOSS",color="white")
plt.xlabel("W",color="white")
ax=plt.gca()
ax.set_facecolor("#030101")
plt.tick_params(color="white")
plt.grid(True,color="gray",linestyle="--",alpha=0.5)
plt.show()
