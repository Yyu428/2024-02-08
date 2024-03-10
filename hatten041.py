def dfs(graph, start):
    visited[start] = True
    visit_order[start] = i[0]
    i[0] += 1
    
    for neighbor in reversed(graph[start]):
        if not visited[neighbor]:
            dfs(graph, neighbor)

# 入力の読み込み
n = int(input())
m = int(input())

edges = [[] for _ in range(n)]
for _ in range(m):
    u, v = map(int, input().split())
    edges[u].append(v)
    edges[v].append(u)
#初期化
visited = [False] * n
visit_order = [-1] * n
i = [0]
dfs(edges, 0)

# 結果の出力
print(" ".join(map(str, visit_order)))
