import numpy as np

# 入力を受け取る
element_type = input()
n=int(input())
if element_type=="int":
    matrix1=[]
    for _ in range(n):
        row = list(map(int,input().split()))
        matrix1.append(row)

    matrix2 = []
    for _ in range(n):
        row = list(map(int, input().strip().split()))
        matrix2.append(row)

    matrix1 = np.array(matrix1)
    matrix2 = np.array(matrix2)
    result = np.array_equal(matrix1, matrix2)
else:
    matrix1=[]
    for _ in range(n):
        row = list(map(float,input().split()))
        matrix1.append(row)

    matrix2 = []
    for _ in range(n):
        row = list(map(float, input().strip().split()))
        matrix2.append(row)

    matrix1 = np.array(matrix1)
    matrix2 = np.array(matrix2)
    result = np.allclose(matrix1, matrix2, atol=1e-8, rtol=1e-5)

print(result)
