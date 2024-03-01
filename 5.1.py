import math
#考えたやつ
def check(a):
    flag=True
    #for i in range(2,a):
    for i in range(2,int(math.sqrt(a)+1)):
        if a%i==0:
            flag=False
            break
    return flag
Q=int(input())
x=[]
for i in range(Q):
    x.append(int(input()))
for i in range(len(x)):
    if check(x[i]):
        print("Yes")
    else:
        print("No")
#改善版
#素数判定においてすべての要素を調べる必要はない
