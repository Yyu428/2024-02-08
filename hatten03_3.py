#a,b=map(int,input().split())
#c,d=map(int,input().split())
a=int(input())
b=int(input())
c=int(input())
d=int(input())
#どちらかの種類の花束をでも一個作ろうとしたらアカ一本と青一本を消費する
#更に赤c-1本か青d-1本のどちらかを消費する
#a-kとb-kからc-1,d-1を取れる最大値は(a-k)//(c-1)+(b-k)//(d-1)
#これがk以上ならk個作れる
#10**18を最大値として
ok=0
ng=10**18
while ok+1<ng:
    k=(ok+ng)//2
    print(a-k,c-2,b-k,d-2)
    if (a-k)//(c-2)+(b-k)//(d-2)>=k:
        ok=k
    else:
        ng=k
print(ok)