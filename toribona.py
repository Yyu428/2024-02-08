"""global count
def tribo(n,memo):
    if n<=3:
        count=1
        return memo[n-1]
    else:
        result=tribo(n-3,memo)+tribo(n-2,memo)+tribo(n-1,memo)
        memo.append(result)
        count+=1
        return count
n = int(input())
memo=[1,1,2]
result=tribo(n,memo)
print(count)
print(result)"""
def tribo1(n, memo={}):
    count=0

    def tribo2(n):
        nonlocal count
        if n in memo:
            return memo[n],0

        if n==1 or n==2:
            result=1
        elif n==3:
            result=2
        else:
            result1, _ =tribo2(n-3)
            result2, _ =tribo2(n-2)
            result3, _ =tribo2(n-1)
            result=result1+result2+result3

        memo[n]=result
        count+=1
        return result,count

    result,count=tribo2(n)
    return count,result

n=int(input())
count,result=tribo1(n)

if n>3:
    print(count-3)
else:
    print(count)
print(result)
