#Yaxlitlash funksiyasi
# a=3.64

# def my_int(x):
#     butun = int(x)
#     kasr = x - butun

#     if kasr >= 0.5:
#         return butun + 1.0
#     elif kasr <= 0.5:
#         return butun
#     else:
#         return butun
a=3.64
#Yaxlitlash funksiyasi
def my_int(x):
    butun = int(x)
    kasr = x - butun

    if kasr >= 0.5:
        return float(butun + 1)
    else:
        return float(butun)
print("yaxlitlangan qiymat: ",my_int(a))