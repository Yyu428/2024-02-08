n=int(input())
a=[]
for i in range(n):
    a.append(int(input()))
for i in range(1,n):
    if a[i-1]==a[i]:
        print("stay")
    elif a[i-1]<=a[i]:
        print("up",a[i]-a[i-1])
    else:
        print("down",a[i-1]-a[i])