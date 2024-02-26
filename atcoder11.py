s,t=map(str,input().split())
floor=["B9","B8","B7","B6","B5","B4","B3","B2","B1","1F","2F","3F","4F","5F","6F","7F","8F","9F"]
time_f=[]
for i in range(len(floor)):
    if s==floor[i]:
        time_f.append(i)
        break
for i in range(len(floor)):
    if t==floor[i]:
        time_f.append(i)
        break
print(abs(int(time_f[0])-int(time_f[1])))