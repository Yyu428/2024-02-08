H,W=map(int,input().split())
a=[int(i) for i in input().split()]
#4 1 のみで考えてみる
height=[]
for i in reversed(range(len(a))):
    height.append(H//a[i])
    H%=a[i]
vertical=[]
for i in reversed(range(len(a))):
    vertical.append(W//a[i])
    W%=a[i]
print(height,vertical)
all_num=0
if len(height)==1 and len(vertical)==1:
    all_num+=height[0]*vertical[0]
print(all_num)
"""
all_num=0
for i in range(len(a)):
    i
    all_num+=height[0]*vertical[0]

all_num=0
h_num=H//a[len(a)-1]
v_num=W//a[len(a)-1]
all_num+=h_num*v_num
print(all_num)
def count_tiles(H, W, a):
    total_tiles = 0
    current_height = H

    for size in reversed(a):
        num_tiles = (W // size) * (current_height // size)
        total_tiles += num_tiles
        W %= size
        if W == 0:
            break
        current_height = size

    return total_tiles

# 入力を受け取る
H, W = map(int, input().split())
a = list(map(int, input().split()))

# タイルの総数を計算する
result = count_tiles(H, W, a)

# 結果を出力する
print(result)


"""