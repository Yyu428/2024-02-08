n=int(input())
d={}
for i in range(n):
    key,val=map(str,input().split())
    if key not in d:
        d[key]=[]
    d[key].append(val)
for key in d:
    d[key].sort()
    print(key,*d[key])