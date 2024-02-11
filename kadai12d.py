from operator import itemgetter
cnt1=0
cnt2=0
def saiki(b,q,n):
    global cnt1
    cnt1+=1
    if n==0:
        opt=0
        return opt
    else:
        x1=saiki(b,q,n-1)
        x2=saiki(b,q,q[n])
        if x1>=b[n][2]+x2:
            opt=x1
        else:
            opt=b[n][2]+x2
        return opt
def memo(b,q,n,O):
    global cnt2
    cnt2+=1
    if O[n]!=-1:
        opt=O[n]
        return opt
    else:
        x1=memo(b,q,n-1,O)
        x2=memo(b,q,q[n],O)
        if x1>=b[n][2]+x2:
            O[n]=x1
        else:
            O[n]=b[n][2]+x2
        opt=O[n]
        return opt

n=int(input())
a=[0]*(n+1)
a[0]=[0,0,0]
for i in range(1,n+1):
   a[i]=[int(i) for i in input().split()]
b=sorted(a,key=itemgetter(1))
q=[0]*(n+1)
for i in range(2,n+1):
    max=0
    for k in range(1,i):
        if b[k][1]<=b[i][0]:
            max=k
    q[i]=max
opt1=saiki(b,q,n)
print("opt:",opt1)
print("calls:",cnt1)
O=[-1]*(n+1)
O[0]=0
opt2=memo(b,q,n,O)
print("opt:",opt2)
print("calls:",cnt2)