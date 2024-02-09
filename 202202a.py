n=input()
ans=0
a=["0","1","2","3","4","5","6","7","8","9","a","b","c","d","e","f"]
for i in range(len(n)):
    for j in range(len(a)):
        if n[i]==a[j]:
            ans+=j*(16**(len(n)-1-i))
            break
print(ans)