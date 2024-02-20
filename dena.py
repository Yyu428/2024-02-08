#商品番号と一致したインデックスを返す関数
def food_number(a,line):
    for i in range(len(a)):
        if a[i][0]==line:
            ans=i
            break
    return ans
#createの中にlineがあるか返す関数 unpexpected用
def completer(create,line):
    flag=False
    for i in create:
        if line==i:
            flag=True
    return flag
def search(create,line):
    for i in range(len(create)):
        if line==create[i]:
            ans=i
            break
    return ans
def searcher(create,line):
    for i in range(len(create)):
        if line==create[i][0]:
            ans=i
            break
    return ans
step=int(input())
if step==1:
    amount_menu=int(input())
    #配列メニュー料理番号、初期在庫、価格
    a=[[0]*3 for i in range(amount_menu)]
    for i in range(amount_menu):
        a[i][0],a[i][1],a[i][2]=map(int,input().split())
    line=[]
    #注文情報席番号、料理番号、注文数
    while True:
        try:
            line.append(input().split())
        except EOFError:
            break
    for i in range(len(line)):
        #注文受理不可能の場合
        if a[food_number(a,int(line[i][2]))][1]<int(line[i][3]):
            print("sold out",line[i][1])
        #注文許可の場合
        else:
            #print(line[i][1],line[i][3],a[food_number(a,line[i][2])][1],food_number(a,line[i][2]))
            a[food_number(a,int(line[i][2]))][1]-=int(line[i][3])
            #print(line[i][1],line[i][3],a[food_number(a,line[i][2])][1],food_number(a,line[i][2]))
            for k in range(int(line[i][3])):
                print("received order",line[i][1],line[i][2])
                
elif step==2:
    #メニュー、電子レンジの数
    amount_menu,k=map(int,input().split())
    #配列メニュー料理番号、初期在庫、価格
    a=[[0]*3 for i in range(amount_menu)]
    for i in range(amount_menu):
        a[i][0],a[i][1],a[i][2]=map(int,input().split())
    line=[]
    #注文情報席番号、料理番号、注文数
    """while True:
        try:
            line.append(input().split())
        except EOFError:
            break"""
    for i in range(6):
        line.append(input().split())
    #print(line)
    comp=["line"]
    a="line"
    """if completer(comp,a):
        print("YYYYYYYYY")
    else:
        print("NNNNNN")"""
    #作成中か否か判断、レンジの数、－１が動作なしの状態
    create=[]
    #注文作成待ち
    stack=[]
    print(len(line))
    i=0
    while True:
        if line[i][0]=="received" and len(create)<k:
                print(line[i][3])
                create.append(int(line[i][3]))
        elif line[i][0]=="received" and len(create)==k:
            print("wait")
            stack.append(line[i][3])
        elif line[i][0]=="complete":
            #print(line[i][1])
            #製造中にコンプリートの後の文字があるならかどうか
            #print(create,line[i][1])
            if completer(create,int(line[i][1])):
                #あって予約がない場合
                if len(stack)!=0:
                    del create[0]
                    create.append(int(stack[0]))
                    print("ok",stack[0])
                    del stack[0]
                    print(create)
                elif len(stack)==0:
                    print("ok")
            #製造中にない場合
            else:
                print("unexpected input")
        i+=1
        if i==len(line) and len(stack)==0:
            break
elif step==3:
    #メニュー数
    amount_menu=int(input())
    a=[[0]*3 for i in range(amount_menu)]
    for i in range(amount_menu):
        a[i][0],a[i][1],a[i][2]=map(int,input().split())
    #完成情報
    line=[]
    """while True:
        try:
            line.append(input().split())
        except EOFError:
            break"""
    for i in range(6):
        line.append(input().split())
    create=[]
    attend=[]
    for i in range(len(line)):
        if line[i][0]=="received":
            create.append(int(line[i][3])) 
            attend.append(int(line[i][2]))
        if line[i][0]=="complete":
            k=search(create,int(line[i][1]))
            print("ready",attend[k],line[i][1])
            del create[k]
            del attend[k]
elif step==4:
    #メニュー数
    amount_menu=int(input())
    a=[[0]*3 for i in range(amount_menu)]
    #料理番号、初期在庫数、価格
    for i in range(amount_menu):
        a[i][0],a[i][1],a[i][2]=map(int,input().split())
    #完成情報
    line=[]
    """while True:
        try:
            line.append(input().split())
        except EOFError:
            break"""
    for i in range(7):
        line.append(input().split())
    #席の注文状況
    create = [[] for _ in range(10001)]
    price = [0 for _ in range(10001)]
    #print(price)
    #print(create[int(line[0][2])])
    for i in range(len(line)):
        if line[i][0]=="received":
            create[int(line[i][2])].append(int(line[i][3]))
        elif line[i][0]=="ready":
            #print(create[int(line[i][1])])
            del create[int(line[i][1])]
            #print(line[i][2],a)
            price[int(line[i][1])]+=int(a[searcher(a,int(line[i][2]))][2])
        elif line[i][0]=="check":
            if len(create[int(line[i][1])])==0:
                print(price[int(line[i][1])])
                price[int(line[i][1])]=0
            else:
                print("please wait")
        print(price[10],create[10])