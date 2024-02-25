def f(d,opt,s,t):
    p=""
    if s==t or s==t-1:
        for i in range(s,t+1):
            p=p+"A"+str(i+1)
        return(p)
    else:
        for k in range(s,t):
            a=opt[s][k]+opt[k+1][t]+(d[s]*d[k+1]*d[t+1])
            if opt[s][t]==a:
                break
        L=f(d,opt,s,k)
        R=f(d,opt,k+1,t)
        if s==k:
            p=L+"("+R+")"
        elif k+1==t:
            p="("+L+")"+R
        else:
            p="("+L+")("+R+")"
        return(p)

d=[int(i) for i in input().split()]
n=len(d)-1
opt=[[9999999]*(n) for i in range(n)]
for s in range(n):
    opt[s][s]=0
for i in range(0,n):
    for s in range(0,n-i):
        for k in range(s,s+i):
            t=opt[s][k]+opt[k+1][s+i]+(d[s]*d[k+1]*d[s+i+1])
            if t<opt[s][s+i]:
                opt[s][s+i]=t
p=f(d,opt,0,n-1)
print(p)
