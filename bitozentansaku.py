def f(s):
    n = len(s)
    for x in range(1 << n):
        combination = ""
        for i in range(n):
            if (x >> i) & 1:
                combination += s[i]
        if combination!="":
            print(combination)
n=input()
f(n)
