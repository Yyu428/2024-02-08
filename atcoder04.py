n=int(input())
a=[]
for i in range(n):
    a.append(int(input()))
ans=[0]*2
b=[0]*(n)
for i in range(len(a)):
    b[a[i]-1]+=1
if max(b)==1 and min(b)==1:
    print("Correct")
else:
    for i in range(1,len(b)):
        if b[i]==0:
            ans[0]=i+1
            for j in range(i,len(b)):
                if b[j]==2:
                    ans[1]=j+1
    print(ans[1],ans[0])



