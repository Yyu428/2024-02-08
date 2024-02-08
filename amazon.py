def remover(a):
    remove=[]
    for i in range(len(a)-1):
        for j in range(i+1,len(a)):
            if a[i]==a[j]:
                remove.append(a[i])
    result=list(set(remove))
    return result
def solution(A):
    b=[]
    ans=-1
    for i in range(len(A)):
        keta1=len(str(A[i]))
        b.append(keta1)
    r=remover(b)
    if r==0:
        ans=A[0]+A[1]
    else:
        count=0
        while True:
            #print(A,b,r)
            i=0
            while True:
                if r[count]==b:
                    del A[i]
                    del b[i]
                #print(A)
                i+=1
                if i==len(A):
                    break
            count+=1
            if count==len(r):
                break
        if len(A)==0:
            ans=-1
        else:
            ans=A[0]+A[1]
    return ans
A=[int(i) for i in input().split()]
A.sort(reverse=True)
print(solution(A))
