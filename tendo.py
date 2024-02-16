from operator import itemgetter
M,L=map(int,input().split())
N=int(input())
Name=[""]*(N+1)
Price=[0]*(N+1)
Level=[0]*(N+1)
for i in range(1,N+1):
    a=[i for i in input().split()]
    Name[i]=a[0]
    Price[i]=int(a[1])
    Level[i]=int(a[2])
#print(Name,Price,Level)
ans=[]
for i in range(1,N+1):
    print(ans)
    if L>=Level[i] and M>=Price[i]:
        st=Name[i]+" "+str(Price[i])
        ans.append(st)
for i in range(len(ans)):
    print(ans[i])