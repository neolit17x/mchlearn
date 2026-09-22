import matplotlib.pyplot as plt
import numpy as np

x_soat=np.array([1.0,2.0,3.0])
y_baho=np.array([2.0,4.0,6.0])


def forward(x):
    return x*w

def loss(x,y):
    y_hat=forward(x)
    return (y_hat-y)**2

w_list=[]
mse_list=[]


for w in np.arange(0.0,4.1,0.1):
    print(f"\nw = {w:.3f}")
    l_umum=0
    for x_hb_qiymat,y_hb_qiymat in zip(x_soat,y_baho):
        y_hb_bashorat=forward(x_hb_qiymat)
        loss_hb_qiymat=loss(x_hb_qiymat,y_hb_qiymat)
        l_umum+=loss_hb_qiymat
        print("\t",f"{x_hb_qiymat:.2f}",f"{y_hb_qiymat:.2f}",f"{y_hb_bashorat:.2f}",f"{loss_hb_qiymat:.2f}")
    print("MSE=",l_umum/len(x_soat))
    w_list.append(w)
    mse_list.append(l_umum/len(x_soat))


plt.plot(w_list,mse_list)
plt.ylabel("Loss",color="white")
plt.xlabel("W",color="white")
ax = plt.gca()
ax.set_facecolor("#030101")      # Grafik ichki foni
plt.gcf().patch.set_facecolor("#030101")  # Tashqi fon
ax.tick_params(colors="white")   # Sonlar oq rangda
plt.grid(True, color="gray", linestyle="--", alpha=0.5)
plt.show()
