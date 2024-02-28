n=int(input())
a=[]
for i in range(n):
    a.append(input())
a.sort()
#print(a)
b=[0]*len(a)
#print(a)
count=0
for i in range(len(a)-1):
    if a[i]==a[i+1]:
        b[i]=1
        b[i+1]=1
c=[]
for i in range(len(b)):
    if b[i]==0:
        c.append(a[i])
if len(c)==0:
    print("none")
for i in range(len(c)):
    print(c[i])