import networkx as nx
#グラフに点が含まれているかどうかの判定
def check(G,a):
    flag=False
    for i in G:
        if a==i:
            flag=True
    return flag
#隣接点逆順
def neibor(G,label):
    order=[]
    if check(G.nodes,label):
        for i in G.edges(label):
            order.append(i[1])
    return order
n=int(input())
m=int(input())
lines=[]
for i in range(m):
    lines.append(input())
G=nx.parse_edgelist(lines,nodetype=int)
L=[]
visit=[-1]*(n)
i=0
L.append(0)
visit[0]=0
while len(L)!=0:
    #先頭の要素抜き出す
    v=L.pop(0)
    visit[v]=i
    i+=1
    #先頭の要素の隣接点
    neiborhood=neibor(G,v)
    #neiborhood.reverse()
    #print(v,neiborhood,visit,L)
    for j in range(len(neiborhood)):
        if visit[neiborhood[j]]==-1:
            L.insert(0,neiborhood[j])
            #i+=1
            visit[neiborhood[j]]=0
print(*visit)