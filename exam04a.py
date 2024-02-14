a=[int(i) for i in input().split()]
a.sort()
b=[0]*len(a)
#print(a)
count=0
for i in range(len(a)-1):
    if a[i]==a[i+1]:
        b[i]=1
        b[i+1]=1
#print(b)
c=[]
for i in range(len(b)):
    if b[i]==0:
        c.append(a[i])
print(*c)