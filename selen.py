# selen system 
# 最終更新日:2022/12/15
#======================================================================
# import OpenCv
import cv2
# ファイル読み込み用
import glob 
# import numpy
import numpy as np
# PILライブラリを利用する。
from PIL import Image, ImageDraw, ImageFont,ImageTk,ImageOps
import sys
# 乱数
import random
# GUI <-pip install PySimpleGUI
import PySimpleGUI as sg
# osモジュール
import os
# tkinter (PILのImageTkは上記でインポート済み)
import sys
import tkinter as tk
import tkinter.messagebox as tkm #ダイアログ
import tkinter.ttk as ttk #コンボボックス
# 時間
import time
import datetime
# ハッシュ関数を実装するためのhashlibモジュールをインポート
import hashlib
import bcrypt as bp #bcrypt
# 音声合成用
import win32com.client as wincl
voice = wincl.Dispatch("SAPI.SpVoice")
#=====================================================
#AKAZE特徴量を使用
pic_path='faces/'
prePic_path='preFaces/'

borderU=130
borderD=90

IMG_SIZE = (200,200)    #顔の大きさ
#=====================================================
# 元画像のスクラムブル化関数(Kamimura)
def shuffle(img):
    gy,gx,gc=img.shape[:3]
    gy2,gx2,gc2=img.shape[:3]
    img2=img.copy()
    for y in range(0,gy2):
        np.random.seed(y)
        a=np.arange(gx2)
        np.random.shuffle(a)
        for x in range(0,gx2):
            img2[y,x]=img[y,a[x]]
    img3=cv2.resize(img2,(gx,gy))
    return img3

# スクラムブル画像の復元関数(Kamimura)
def shuffle_back(img):
    gy,gx,gc=img.shape[:3]
    gy2,gx2,gc2=img.shape[:3]
    img2=img.copy()
    for y in range(gy2-1,-1,-1):
        np.random.seed(y)
        a=np.arange(gx2)
        np.random.shuffle(a)
        for x in range(gx2-1,-1,-1):
            img2[y,a[x]]=img[y,x]
    img3=cv2.resize(img2,(gx,gy))
    return img3

# 終了処理(Takei)
def endProcess():
    global face_Nlist
    global pic_path
    print("終了処理")
    for name in face_Nlist:
        deletepicture(pic_path+name+".jpg")
        print("削除: ",pic_path+name+".jpg")

# データ更新(Miyazaki)
def data_update():
    files =glob.glob(prePic_path + '*')
    global face_list
    global face_Nlist
    global entering
    global exiting

    # faces内のjpgファイル削除
    for name in face_Nlist:
        deletepicture(pic_path+name+".jpg") 
    
    new_face_list = []
    new_face_Nlist = []
    new_entering = [] #0:未入場,1:入場,2:遅刻,3:欠席
    new_exiting = []

    # ファイル探査(データの更新)
    print(" --- Data List (updata)--- ")
    i=0
    for f in files:
        img = cv2.imread(f)    
        img = shuffle_back(img)  #スクランブル化 解除
        f = f.replace('preFaces\\', '')
        f = f.replace('.jpg', '')   
        f = f.replace('.png', '')   
        cv2.imwrite(pic_path+f+".jpg", img)
        print(i+1,": ",f)
        img = face_cut(img)
        new_face_Nlist.append(f)
        new_face_list.append(img)
        new_entering.append(0)
        new_exiting.append(False)
        # 記録データ引継ぎ
        for t in range(len(face_Nlist)):
            if face_Nlist[t] == f:
                new_entering[i]=entering[t]
                new_exiting[i]=exiting[t]
        i+=1
    
    face_list = new_face_list
    face_Nlist = new_face_Nlist
    entering = new_entering
    exiting = new_exiting
    print(" ------------------------- ")
    print("")
                    
# ファイル削除(Miyazaki)
def deletepicture(filename):
    os.remove(filename)

#入力: 二つの顔画像, 出力: 相違度(Takei)
def similarity(face1,face2):    
    akaze = cv2.AKAZE_create()
    (face1_kp, face1_des) = akaze.detectAndCompute(face1, None) #kpは特徴点の座標
    (face2_kp, face2_des) = akaze.detectAndCompute(face2, None) #desは特徴点点数

    #ここからマッチング
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.match(face1_des, face2_des)
    #距離計算
    dist = [m.distance for m in matches]
    if len(dist) != 0:
        ret = sum(dist) / len(dist)
        return int(ret)
    return -1 

#入力: 画像のパス, 出力: 顔画像(Takei)
def face_cut(img1):
    
    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

    #グレースケール
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    #顔領域
    img1_faces = face_cascade.detectMultiScale(img1_gray, minSize=(100,100))

    #座標取得
    img1_face_rect =img1_faces[0]
    x1,y1,w1,h1 = img1_face_rect[0],img1_face_rect[1],img1_face_rect[2],img1_face_rect[3]

    #大きさ補正
    img1_face = img1[y1:y1+h1, x1:x1+w1]
    img1_face = cv2.resize(img1_face,IMG_SIZE)

    return img1_face

#入力: 顔画像データベース/比較したい顔画像, 出力: 相違度最小の添え字(Takei)
def compareAll(list, face_gray):    
    score_list=[]
    for file in list:
        score_list.append(similarity(face_gray,file))
    #print(" score:               ",score_list)
    minS=min(score_list)
    if minS==-1 or minS>borderU or minS<borderD:
        return -1
    return score_list.index(min(score_list))

picture_num =len( glob.glob("./illustration/" + '*') )
# 顔にイラストを合成する関数(Miyazaki)
def effect_face_func(img, rect):
    global picture_num
    (s_x, s_y, f_x, f_y) = rect
    w = f_x - s_x + 1 #顔の領域の幅
    h = f_y - s_y + 1 #顔の領域の高さ

    #背景画像の設定
    img_gp=img

    #合成するイラストの設定(アルファ画像の読込) 
    rnd=random.random()
    #乱数より、イラストの選択   //selen17
    #print("effect_face_func: picture_num=",picture_num)
#    for i in range(n):
#        if rnd <= float(i+1)/n:
#            num=i+1
#            break      
    num = int(rnd*picture_num)+1
    picture_path='./illustration/'+str(num).zfill(3)+'.png'

    add_img = cv2.imread(picture_path, -1) # アルファチャンネルで読み込み 
    add_img = cv2.resize(add_img,(w,h))
    
    #マージする
    alpha = add_img[:,:,3]  # アルファチャンネルだけ抜き出す(要は2値のマスク画像)

    alpha = cv2.cvtColor(alpha, cv2.COLOR_GRAY2BGR) # grayをBGRに
    alpha = alpha / 255.0    # 0.0〜1.0の値に変換

    fg = add_img[:,:,:3]

    #合成時のサイズを計測し、h(領域の高さ)と同じでない場合はそのまま画像を返す
    height,width,channels = img[s_y:h+s_y, s_x:w+s_x].shape[:3]
    #カメラの上下の認識範囲外に顔が一部出てしまった場合のエラー対処
    if height != h:
        #print("re")
        return img
    #カメラの左右の認識範囲外に顔が一部出てしまった場合のエラー対処
    if width != w:
        #print("re")
        return img

    img[s_y:h+s_y, s_x:w+s_x] = (img[s_y:h+s_y, s_x:w+s_x] * (1.0 - alpha)).astype('uint8') # アルファ以外の部分を黒で合成
    img[s_y:h+s_y, s_x:w+s_x] = (img[s_y:h+s_y, s_x:w+s_x] + (fg * alpha)).astype('uint8')  # 合成

    return img

#文字表示関数<日本語に対応>(Miyazaki)
def img_put_Text(img, text , org, colorBGR):
    # フォントパス＆サイズの指定
    font_path = './SourceHanSerifK-Light.otf'
    font_size = 24
    # PILライブラリを利用してフォントを定義
    font = ImageFont.truetype(font_path, font_size) 
    # cv2(numpy.ndarray)型の画像を、PILライブラリに合わせた型の画像へ変換
    img = Image.fromarray(img) 
    # 文字列を挿入するために、描画用のDraw関数を用意
    draw = ImageDraw.Draw(img)
    # 文字列を描画(位置, 文字列, フォント情報, 文字列色)
    draw.text(org, text, font=font, fill=colorBGR)
    # PILライブラリに合わせた型の画像から、cv2(numpy.ndarray)型の画像へ変換
    img = np.array(img)

    return img

#レイアウト(Miyazaki)
def layout(img,name,idx,mode):
    #背景画像(デザイン)の取り込み
    picture_path='./groundimg.jpg'
    gimg = cv2.imread(picture_path, -1)

    #背景画像のサイズ取得 
    gy,gx,gc=gimg.shape[:3]

    #カメラサイズの取得
    y,x,c=img.shape[:3]

    #カメラの領域サイズの取得
    w = (int)(gx*0.9) #幅
    h = (int)(w*y/x) #高さ
    
    #カメラ映像左上の座標取得
    s_x=(int)(gx*0.05)
    s_y=(int)(gy*0.1) 
        
    #カメラ映像のリサイズ
    img2 = cv2.resize(img , (w,h))
    
    #合成
    gimg[s_y:h+s_y, s_x:s_x+w] = img2 
    
    #テクスチャ
    text_name = name
    if idx == -1:
        text_message = "カメラを見てください"
    else:
        #モードの設定 0:入場モード 1:退場モード(selen15追加)
        if mode==0:
            text_message = "認証しました‼入場を記録"
        elif mode==1:
            text_message = "認証しました‼退場を記録"
    #org_name=((int)(w/2-2*s_x),(int)(s_y+h+(gy-s_y-h)/2-s_y)) #左下基準
    org_name=((int)(4*s_x),(int)(s_y+h+(gy-s_y-h)/2-s_y)) 
    org_message=((int)(4*s_x),(int)(s_y+h+(gy-s_y-h)/2)) #左下基準
    color_name=(0, 0, 0) #BGR
    color_message=(0, 0, 255) #BGR
    rimg=img_put_Text(gimg,text_name,org_name,color_name)
    rrimg=img_put_Text(rimg,text_message,org_message,color_message)

    return rrimg

# 精度向上用 (takeFaceNum)回の判定から判別 (Takei)
import collections
queue = []
takeFaceNum = 10    # 決定するために必要な枚数
def decideName(idx,mode):
    if idx < 0:
        return -1
    global queue
    global takeFaceNum
    global entering
    global exiting
    queue.append(idx)
    #print("queue: ",queue)
    if len(queue) < 10:
        return -1
    else:
        c = collections.Counter(queue)
        ansIdx = c.most_common()[0][0]
        num = c.most_common()[0][1]
        #print("類似率:",num/takeFaceNum*100, "%")
        queue.pop(0)
        if num > 7/10*takeFaceNum:
            # モードの設定 0:入場モード 1:退場モード(selen15追加)       #selen17 修正
            if mode==0:
                entering[ansIdx] = 1
            elif mode ==1:
                exiting[ansIdx] = True
            return ansIdx
        else:
            #print("-failed-")
            queue = []
            return -1

# entering[idx]=1 で入室 (0:未入場,1:入場,2:遅刻,3:欠席) 
# exiting[idx]=True で退室
# 終了時処理 入力、出力ともになし (Takei)
def endGui(mode):
    global face_Nlist
    global entering
    global exiting
    n = len(face_Nlist)
    sg.theme("BrightColors")

    appName = "Facial Recognition"
    entryText = "Entering person"
    newMes = "[ Someone Still Remains ! ]"
    finishText = "End"
    continueText = "Continue"
    if mode == 1:
        entryText = "Exiting person"

    layout = []
    layout.append([sg.Text(entryText)])
    for j in range(n):
        # モードの設定 0:入場モード 1:退場モード(selen15追加)
        stats = ""
        #print("enterList: ",entering)
        if entering[j]==0:
            stats="[未入場] 修正してください"
        elif entering[j]==1:
            stats="入場"
        elif entering[j]==2:
            stats="遅刻"
        elif entering[j]==3:
            stats="欠席"


        if mode == 0:
            items = ["入場","遅刻","欠席"]

            layout.append([ sg.Text(face_Nlist[j]),sg.Combo(items, default_value=stats, size=(30,1),key=j) ])
        elif mode == 1:
            if entering[j] ==1:
                layout.append([ sg.Checkbox(face_Nlist[j], default=exiting[j], key=j) ])
                #layout.append([ sg.Text(face_Nlist[j]),sg.Combo(items, default_value=stats, size=(30,1),key=j) ])
            elif entering[j]==2:
                layout.append([ sg.Checkbox(face_Nlist[j], default=exiting[j], key=j), sg.Text("(遅刻)") ])
            elif entering[j]==3:
                layout.append([ sg.Checkbox(face_Nlist[j], default=True, key=j), sg.Text("(欠席)") ])

    layout.append([sg.Text("", key="-MES-")])
    layout.append([sg.Button(finishText),sg.Button(continueText)])
    window = sg.Window(appName, layout,size = (450,800))

    while True:
        event, values = window.read()   #固定
        if event == sg.WIN_CLOSED or event == finishText:   #終了
            flag=True
            i=0
            #print("values: ",values.items())
            for value in values.items():

                #print(value)
                if mode==0: #入場mode

                    if value[1] == "[未入場] 修正してください":
                        #print("ERR: ",face_Nlist[i],value)
                        flag= False
                        window["-MES-"].update(newMes)
                    elif value[1] == "入場":
                        entering[i] = 1
                    elif value[1] == "遅刻":
                        entering[i] = 2
                    elif value[1] == "欠席":
                        entering[i] = 3
                    #print("entering:",entering)

                if mode==1: #退場mode
                    print("exit: ",(value[1]))
                    if (not value[1]):
                        #print("ERR: ",face_Nlist[i],"remains")
                        flag= False
                        window["-MES-"].update(newMes)
                        #selen017 警告入れるならここ

                    exiting[i] = value[1]
                    
                    
                i+=1

            if flag:
                print("endGUI: EnterList",entering)
                print("endGUI: ExitList",exiting)
                break


        if event == continueText:
            #print("endGUI: continue")
            flag=False
            break

    #print("endGUI: while break, flag=",flag)
    window.close()
    return flag


# 入退場管理システム  (Takei&Miyazaki)
# mode=0:入場モード mode=1:退場モード
def main_system(mode):
    global face_Nlist
    global entering
    global exiting
    global queue
    if mode == 0:
        system_name='selen entrance system'
    else:
        system_name='selen exit system'
    # カメラから映像を取得
    cap = cv2.VideoCapture(0)

    nonReflectedTime = 0
    while True:
        name = "・・認識しています・・"
        flag = -1
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        for x, y, w, h in faces:
            #-----ここから顔認識-----
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face = img[y: y + h, x: x + w]
            face_gray = gray[y: y + h, x: x + w]

            #---------#
            
            nonReflectedTime=0
            idx = decideName( compareAll(face_list, face),mode )     ##
            #print("end: ",entering)

            if idx == -1:
                flag=-1
                #print(" - failed - ")
            else:
                flag=idx
                #print("result",face_Nlist[idx])
                # レイアウト/文字表示
                name = face_Nlist[idx]+"さん"
                name = name.replace('faces\\', '')
                name = name.replace('.jpg', '')
                
                # 顔にイメージをつける
                img = effect_face_func(img,(x-10, y-10, x+w+10, y+h+10))
            #print()  
                    
        nonReflectedTime+=1
        if nonReflectedTime>10:
            queue=[]
        #print("queue: ",queue)
                
        img = layout(img,name,flag,mode)

        cv2.imshow(system_name, img)
        key=cv2.waitKey(1)
        if flag == -1:
            key = cv2.waitKey(10) 
        else:
            # 音声で認識をお知らせする
            if mode==0:
                voice.Speak(name+" こんにちは")
            elif mode==1:
                voice.Speak(name+" さようなら")
            key = cv2.waitKey(1000)

        if key == 27:  # ESCキーで終了
            if endGui(mode):
                #print("finish")
                break #終了
            #else:
                #print("continue")

    print("main System: break")
    cap.release()
    cv2.destroyAllWindows()
    home() #HOMEへ
    return False
  
# 登録システム (Takei)
def register(pic_name):
    face_cascade_path = './haarcascades/haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    cap = cv2.VideoCapture(0)

    #pic_name = input(" [登録者名を入力]: ")
    #print()
    cnt=5
    i=0
    j=0
    flag=True
    capture=False

    while flag:
        #print("i: ",i, "j: ",j, "cnt: ",cnt)
        j+=1
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for x, y, w, h in faces:
            i+=1
            if i>10:    #1sごと
                cnt-=1
                i=0
                j=0
            if cnt==0:
                capture = True
                cnt=5

            if capture:
                cv2.imwrite(pic_path+pic_name+".jpg",img)
                
                #yn=input("この写真でよろしいですか？ y / n: ")
                #if(yn=="y"):
                flag=False
                #print("登録完了  ようこそ ",pic_name," さん")
                #else:
                #    deletepicture(pic_path+pic_name+".jpg") #画像削除
                #    capture=False
                    
            else:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(img, str(cnt), (0, 0),cv2.FONT_HERSHEY_PLAIN,5,(255, 0, 0),2,cv2.LINE_4)

        if j>30:    #制限超過
            i=0
            j=0
            cnt=5
        cv2.putText(img,
                text=str(cnt),
                org=(100, 300),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1.0,
                color=(0,0,0),
                thickness=2,
                lineType=cv2.LINE_4)

        cv2.imshow('video image', img)
        key = cv2.waitKey(10)
        if key == 27:  # ESCキーで終了
            break  

    cap.release()
    cv2.destroyAllWindows()

# 新規登録機能~GUI~ (Miyazaki)
addname=None
addfilename=None
def New_registration_func():
    global addname
    global addfilename

    # Enterボタンが押された時に実行する処理
    def button_click():
        global photo_image #表示するためのテクニック(遅れる読み込みを、参照させ保持)
        global addname
        global addfilename
        addname = input_pass.get() #入力された文字列を取得
        # 登録名の重複をチェック
        files = glob.glob(prePic_path + '*')
        for f in files:
            name=f
            name = name.replace('preFaces\\', '')
            name = name.replace('.jpg', '')
            name = name.replace('.png', '') 
            if name == addname:
                addname = None
                break
        
        if addname == None: #Enterが押されている<=>入力されていない場合は""
            msg.configure(text="登録済みです。他の名前を入力してください",foreground='#CC0000')
        elif addname != "":
            #登録システム 起動
            register(addname)
            #==============================================================
            # 取得画像の表示
            addfilename=pic_path+addname+'.jpg'
            # PIL.Imageで開く
            pil_image = Image.open(addfilename)
            # 画像のアスペクト比（縦横比）を崩さずに指定したサイズ（キャンバスのサイズ）全体に画像をリサイズする
            pil_image = ImageOps.pad(pil_image, (400,300), color = back_color)
            #PIL.ImageからPhotoImageへ変換する
            photo_image = ImageTk.PhotoImage(image=pil_image,master=nrf)
            # 画像の描画
            canvas.create_image(200,150,image=photo_image)
            #==============================================================  
            msg.configure(text="この写真でよろしいですか?",foreground='#CC0000') 
        else:
            msg.configure(text="名前を入力してください",foreground='#CC0000')

    # ボタンが押された時に実行する処理
    def button_func(event):
        global addname
        global addfilename
        text=str(event.widget["text"])
        if text == "HOME":
            if tkm.askokcancel('確認', 'ホームへ戻りますか?')==True:
                nrf.destroy()
                home() 
        if text == "Yes":
            if addname != None:
                #--------------------------------------------------------
                # 登録画像をスクランブル化し保存
                img=cv2.imread(addfilename)   
                img=shuffle(img)
                cv2.imwrite(prePic_path+addname+'.png',img)
                deletepicture(addfilename) #元画像の削除
                #--------------------------------------------------------
                data_update() #データ更新
                tkm.showinfo('welcome', addname+'さん‼ようこそ(登録完了)')
                nrf.destroy()
                addname=None
                home()
            else:
                msg.configure(text="名前を入力してください",foreground='#CC0000')
        if text == "No":
            if addname != None:
                deletepicture(addfilename) #画像削除
                addname=None
            tkm.showinfo('メッセージ','もう一度やり直してください')
            nrf.destroy()
            New_registration_func() 

    nrf = tk.Tk()  
    # ウィンドウ設定
    nrf.title(u"New registration function") #タイトル    
    nrf.geometry('460x680') #サイズ(横幅x高さ)
    nrf.configure(background='#CCCCCC')    

    # ===ラベル=== topメッセージ
    top = tk.Label(nrf,text=u'新規登録される方の名前を入力してください',font=("Helvetica",15,"bold"),foreground='#000000',background='#CCCCCC')
    top.place(x=230, y=50,anchor=tk.CENTER) 

    # ===ラベル=== name
    ipl = tk.Label(nrf,text="name",font=("Helvetica",14,"bold"),foreground='#000000',background='#CCCCCC')
    ipl.place(x=50, y=70) 
    
    # 入力欄の作成
    input_pass = tk.Entry(nrf,width=60)
    input_pass.place(x=50, y=100) 

    # 実行ボタン
    Button10 = tk.Button(nrf,text=u'Enter',width=10,font=("Helvetica",12),foreground='#FFFFFF',background='#CC0000',command=button_click)
    Button10.place(x=320, y=130) 
    
    # ===Canvas=== 画像表示用
    back_color = "#008B8B" # 背景色
    canvas = tk.Canvas(nrf,width=400,height=300,relief=tk.RIDGE,bg=back_color)   
    canvas.place(x=230,y=330,anchor=tk.CENTER)  

    # ===ラベル=== 登録確認メッセージ
    msg = tk.Label(nrf,text=u'',font=("Helvetica",15,"bold"),foreground='#000000',background='#CCCCCC')
    msg.place(x=230, y=520,anchor=tk.CENTER) 

    # ===Button=== Yes
    Button11= tk.Button(nrf,text=u'Yes',width=5,font=("Helvetica",14,"bold"),foreground='#FFFFFF',background='#CC0000')
    Button11.bind("<Button>", button_func) #ボタンのイベント処理
    Button11.place(x=115, y=560,anchor=tk.CENTER)

    # ===Button=== No
    Button12= tk.Button(nrf,text=u'No',width=5,font=("Helvetica",14,"bold"),foreground='#FFFFFF',background='#CC0000')
    Button12.bind("<Button>", button_func) #ボタンのイベント処理
    Button12.place(x=345, y=560,anchor=tk.CENTER)

    # ===Button=== システム終了
    Button13= tk.Button(nrf,text=u'HOME',width=5,font=("Helvetica",14,"bold"),foreground='#FFFFFF',background='#000000')
    Button13.bind("<Button>", button_func) #ボタンのイベント処理
    Button13.place(x=50, y=620,anchor=tk.CENTER)

    nrf.mainloop()        

# データ削除システム (Miyazaki)
def Delete_data():
    # コンボボックスが選択された時の処理
    def onSelected(event):
        global photo_image
        back_color = "#008B8B" # 背景色
        file=combobox.get()
        dmsg.config(text='こちらの'+file+'を削除しますか?')
        #==============================================================
        filename=pic_path+file+'.jpg'
        # PIL.Imageで開く
        pil_image = Image.open(filename)
        # 画像のアスペクト比（縦横比）を崩さずに指定したサイズ（キャンバスのサイズ）全体に画像をリサイズする
        pil_image = ImageOps.pad(pil_image, (400,300), color = back_color)
        #PIL.ImageからPhotoImageへ変換する
        photo_image = ImageTk.PhotoImage(image=pil_image,master=Del_data)
        # 画像の描画
        canvas.create_image(200,150,image=photo_image)
        #==============================================================    
    
    # ボタンが押された時に実行する処理
    def button_func(event):
        text=str(event.widget["text"])
        if text == "HOME":
            if tkm.askokcancel('確認', 'ホームへ戻りますか?')==True:
                Del_data.destroy()
                home() 
        if combobox.get()!="" and text == "確定":
            file=combobox.get()   
            filename=pic_path+file+'.jpg' 
            prePicfilename=prePic_path+file+'.png'   
            if tkm.askokcancel('確認', file+'を削除しますか?')==True:
                #deletepicture(filename) #faces内のJPG画像を削除する
                deletepicture(prePicfilename) #preFaces内のスクランブル画像を削除する
                data_update() #データ更新
                tkm.showinfo('メッセージ', file+'を削除しました')
            Del_data.destroy()
            Delete_data()

    Del_data = tk.Tk()   
    # ウィンドウ設定
    Del_data.title(u"Data deletion system") #タイトル    
    Del_data.geometry('460x680') #サイズ(横幅x高さ)
    Del_data.configure(background='#CCCCCC')    

    # ===ラベル=== topメッセージ
    top = tk.Label(Del_data,text=u'消去する方を選択してください',font=("Helvetica",15,"bold"),foreground='#000000',background='#CCCCCC')
    top.place(x=230, y=50,anchor=tk.CENTER) 

    # ===ドロップダウンリスト=== 
    v = tk.StringVar() #値を保持する変数
    module = [] #ドロップダウンリストに表示するデータ
    # リストをセットする
    files = glob.glob(pic_path + '*')
    for f in files:
        name=f
        name = name.replace('faces\\', '')
        name = name.replace('.jpg', '')
        module.append(name)
    combobox = ttk.Combobox(Del_data,font=("Courier",16),width=15,state="readonly",textvariable= v,values=module)
    combobox.bind("<<ComboboxSelected>>",onSelected)
    combobox.place(x=230, y=100,anchor=tk.CENTER) 

    # ===Button=== 確定
    Button8= tk.Button(Del_data,text=u'確定',width=5,font=("Helvetica",14,"bold"),foreground='#FFFFFF',background='#CC0000')
    Button8.bind("<Button>", button_func) #ボタンのイベント処理
    Button8.place(x=230, y=580,anchor=tk.CENTER)
    
    # ===Canvas=== 画像表示用
    back_color = "#008B8B" # 背景色
    canvas = tk.Canvas(Del_data,width=400,height=300,relief=tk.RIDGE,bg=back_color)   
    canvas.place(x=230,y=300,anchor=tk.CENTER)  

    # ===ラベル=== 削除確認メッセージ
    dmsg = tk.Label(Del_data,text=u'',font=("Helvetica",15,"bold"),foreground='#000000',background='#CCCCCC')
    dmsg.place(x=230, y=520,anchor=tk.CENTER) 

    # ===Button=== システム終了
    Button9= tk.Button(Del_data,text=u'HOME',width=5,font=("Helvetica",14,"bold"),foreground='#FFFFFF',background='#000000')
    Button9.bind("<Button>", button_func) #ボタンのイベント処理
    Button9.place(x=50, y=620,anchor=tk.CENTER)

    Del_data.mainloop()

# データ参照システム (Takei)
def Browse_data():
    # コンボボックスが選択された時の処理
    def onSelected(event):
        global photo_image
        back_color = "#008B8B" # 背景色
        file=combobox.get()
        #==============================================================
        filename=pic_path+file+'.jpg'
        # PIL.Imageで開く
        pil_image = Image.open(filename)
        # 画像のアスペクト比（縦横比）を崩さずに指定したサイズ（キャンバスのサイズ）全体に画像をリサイズする
        pil_image = ImageOps.pad(pil_image, (400,300), color = back_color)
        #PIL.ImageからPhotoImageへ変換する
        photo_image = ImageTk.PhotoImage(image=pil_image,master=Del_data)
        # 画像の描画
        canvas.create_image(200,150,image=photo_image)
        #==============================================================    
    
    # ボタンが押された時に実行する処理
    def button_func(event):
        text=str(event.widget["text"])
        if text == "HOME":
            if tkm.askokcancel('確認', 'ホームへ戻りますか?')==True:
                Del_data.destroy()
                home() 

    Del_data = tk.Tk()   
    # ウィンドウ設定
    Del_data.title(u"Data Browse System") #タイトル    
    Del_data.geometry('460x680') #サイズ(横幅x高さ)
    Del_data.configure(background='#CCCCCC')    

    # ===ラベル=== topメッセージ
    top = tk.Label(Del_data,text=u'閲覧する方を選択してください',font=("Helvetica",15,"bold"),foreground='#000000',background='#CCCCCC')
    top.place(x=230, y=50,anchor=tk.CENTER) 

    # ===ドロップダウンリスト=== 
    v = tk.StringVar() #値を保持する変数
    global face_Nlist
    combobox = ttk.Combobox(Del_data,font=("Courier",16),width=15,state="readonly",textvariable= v,values=face_Nlist)
    combobox.bind("<<ComboboxSelected>>",onSelected)
    combobox.place(x=230, y=100,anchor=tk.CENTER) 
    
    # ===Canvas=== 画像表示用
    back_color = "#008B8B" # 背景色
    canvas = tk.Canvas(Del_data,width=400,height=300,relief=tk.RIDGE,bg=back_color)   
    canvas.place(x=230,y=300,anchor=tk.CENTER)  

    # ===ラベル=== 削除確認メッセージ
    dmsg = tk.Label(Del_data,text=u'',font=("Helvetica",15,"bold"),foreground='#000000',background='#CCCCCC')
    dmsg.place(x=230, y=520,anchor=tk.CENTER) 

    # ===Button=== システム終了
    Button14= tk.Button(Del_data,text=u'HOME',width=5,font=("Helvetica",14,"bold"),foreground='#FFFFFF',background='#000000')
    Button14.bind("<Button>", button_func) #ボタンのイベント処理
    Button14.place(x=50, y=620,anchor=tk.CENTER)

    Del_data.mainloop()

# パスワード認証 (Miyazaki)
def password_check(inputword):
    check=None
    # ================================================================
    # ハッシュ値(b)をファイルから読み込む
    f = open('password.txt', 'rb')
    password = f.read()
    f.close()
    # ================================================================
    #パスワードチェック
    check = bp.checkpw(inputword.encode(),password)
    return check


##selen17 entrance_record がない場合作成
import os
# 入退場記録のテキストデータ書き込み (Miyazaki)
def data_write():
    global entering #0:未入場,1:入場,2:遅刻,3:欠席
    global exiting
    global dt_st 
    dt_fin = datetime.datetime.now()
    dt_str= dt_st.strftime('%Y%m%d')
    # ================================================================
    # 入場記録をファイルに書き込む
    if  not os.path.exists("data"):
        os.makedirs("data")
    
    f = open('data/entrance_record'+dt_str+'.txt','a',encoding='UTF-8')
    f.write('--------------------------------------- \n')
    f.write('開始日時:'+dt_st.strftime('%Y/%m/%d %H:%M:%S')+'\n')
    #datalist = ['開始日時:\n', 'dt_st.strftime('%Y/%m/%d %H:%M:%S')\n','以下、記録を表示\n']
    #f.writelines(datalist)
    for i in range(len(face_Nlist)):
        if entering[i] == 1:
            f.write(face_Nlist[i]+': 入場\n')
        elif entering[i] == 2:
            f.write(face_Nlist[i]+': 遅刻\n')
        elif entering[i] == 3:
            f.write(face_Nlist[i]+': 欠席\n')
        else:
            f.write(face_Nlist[i]+':\n')
    f.write('終了日時:'+dt_fin.strftime('%Y/%m/%d %H:%M:%S')+'\n')
    f.write('--------------------------------------- \n')
    f.close()
    # ================================================================
    # 退場記録をファイルに書き込む
    f = open('data/exit_record'+dt_str+'.txt','a',encoding='UTF-8')
    f.write('------------------------------------------------------- \n')
    f.write('開始日時:'+dt_st.strftime('%Y/%m/%d %H:%M:%S')+'\n')
    for i in range(len(face_Nlist)):
        if exiting[i] == True:
            if entering[i] == 1 or entering[i] == 2:
                f.write(face_Nlist[i]+': 退場\n')
            elif entering[i] == 3:
                f.write(face_Nlist[i]+': 欠席\n')
            else:
                f.write(face_Nlist[i]+':\n')
        elif exiting[i] == False and entering[i] == 1:
            f.write(face_Nlist[i]+':【警告】入場しているが退場していません\n')
        else:
            f.write(face_Nlist[i]+':\n')
    f.write('終了日時:'+dt_fin.strftime('%Y/%m/%d %H:%M:%S')+'\n')
    f.write('------------------------------------------------------- \n')
    f.close()

# GUI 初期画面 (Miyazaki)
gf = False
def home():
    global gf #True:ログイン
    ending = False  #終了条件を満たす

    # ×ボタン無効化
    def click_close():
        pass

    # パスワード正誤判定処理
    def button_click():
        global gf 
        input_value = input_pass.get() #入力された文字列を取得
        iv=password_check(input_value) #パスワード照合
        if gf==True:
            input_pass.delete(0, tk.END) 
            msg.config(text='既に設定モードです',foreground='#000000')
            root.update()
        else:
            if iv==True:
                gf=True
                #tkm.showinfo('メッセージ', '認証されました!')
                input_pass.delete(0, tk.END) 
                # メッセージ更新
                msg.config(text='パスワードが認証されました(設定モード)',foreground='#CC0000')
                root.update()
            elif iv==False:
                input_pass.delete(0, tk.END)  #Entryの値を削除
                # メッセージ更新
                msg.config(text='パスワードが異なります',foreground='#CC0000')
                root.update()
    
    # ボタンが押された時に実行する処理
    def button_func(event):
        global gf
        global face_Nlist

        text=str(event.widget["text"])
        if text == "入場管理システム":
            if len(face_Nlist) == 0:
                tkm.showinfo('Info','顔情報が登録されていません。「新規登録」を選んで下さい。')
            else:
                root.destroy()
                gf = False
                ending = main_system(0) #入場管理システム起動

        if text == "退場管理システム":
            if len(face_Nlist) == 0:
                tkm.showinfo('Info','顔情報が登録されていません。「新規登録」を選んで下さい。')
            else:
                #selen17
                root.quit()
                root.destroy()
                gf = False
                ending = main_system(1) #退場管理システム起動

        if text == "新規登録" :
            if gf==True :
                if tkm.askyesno('メッセージ', '顔情報を新規登録しますか?') == True:
                    gf = False
                    root.destroy()
                    New_registration_func() #新規登録システム起動
                    #register() #新規登録システム起動
            else:
                tkm.showinfo('メッセージ', 'パスワードを入力してください')
                # メッセージ更新
                msg.config(text='パスワードを入力してください',foreground='#CC0000')
                root.update()
        
        if text == "データ消去":
            if gf==True:
                #tkm.showerror('エラー', '現在、画像の削除はOFFにしております')
                if tkm.askyesno('メッセージ', '顔データを消去しますか?') == True:
                    root.destroy()
                    gf = False
                    Delete_data() #データ削除システム起動
            else:
                tkm.showinfo('メッセージ', 'パスワードを入力してください')
                # メッセージ更新
                msg.config(text='パスワードを入力してください',foreground='#CC0000')
                root.update()
        
        if text == "データの閲覧":
            if gf==True:
                #tkm.showerror('エラー', 'この機能は準備中です。ご利用になれません。')
                root.destroy()
                gf=False
                Browse_data()
            else:
                tkm.showinfo('メッセージ', 'パスワードを入力してください')
                # メッセージ更新
                msg.config(text='パスワードを入力してください',foreground='#CC0000')
                root.update()
    
        if text == "終了":
            if tkm.askokcancel('メッセージ', 'システムを終了しますか?')==True:
                data_write()
                endProcess()
                tkm.showinfo('情報', '入退場データが記録されました!適切に管理してください')
                root.destroy()
                ending=True
                sys.exit(0)
                
        if gf==True and text == "設定モード終了":   
            if tkm.askokcancel('メッセージ', '設定モードを終了しますか?')==True:
                gf = False
                # メッセージ更新
                msg.config(text='',foreground='#000000')
                root.update()

    #最初のGUIの画面に戻る
    def return_view():
        root.destroy()
        home()
    
    root = tk.Tk()

    # ウィンドウ設定
    root.title(u"selen") #タイトル
    root.geometry('460x680') #サイズ(横幅x高さ)
    root.configure(background='#CCCCCC')
    
    # ===ラベル=== ウェルカムメッセージ
    welcome = tk.Label(root,text=u'こんにちは',font=("Helvetica",20,"bold"),foreground='#CC9933',background='#CCCCCC')
    welcome.place(x=230, y=50,anchor=tk.CENTER) 
    
    # ===Button=== 入場管理システムを起動
    Button1 = tk.Button(root,text=u'入場管理システム',width=15,font=("Helvetica",14,"bold"),foreground='#FFFFFF',background='#CC0000')
    Button1.bind("<Button>", button_func)
    Button1.place(x=115, y=150,anchor=tk.CENTER) 

    # ===Button=== 退場管理システムを起動
    Button15 = tk.Button(root,text=u'退場管理システム',width=15,font=("Helvetica",14,"bold"),foreground='#FFFFFF',background='#CC0000')
    Button15.bind("<Button>", button_func)
    Button15.place(x=345, y=150,anchor=tk.CENTER) 
    
    # ===ラベル=== 【管理 設定】
    welcome = tk.Label(root,text=u'【管理 設定】',font=("Helvetica",20,"bold"),foreground='#CC9933',background='#CCCCCC')
    welcome.place(x=230, y=250,anchor=tk.CENTER) 
    
    # ===Button=== 新規登録
    Button2 = tk.Button(root,text=u'新規登録',width=15,font=("Helvetica",14,"bold"),foreground='#FFFFFF',background='#CC0000')
    Button2.bind("<Button>", button_func) #ボタンのイベント処理
    Button2.place(x=115, y=300,anchor=tk.CENTER)

    # ===Button=== データ消去
    Button3 = tk.Button(root,text=u'データ消去',width=15,font=("Helvetica",14,"bold"),foreground='#FFFFFF',background='#CC0000')
    Button3.bind("<Button>", button_func) #ボタンのイベント処理
    Button3.place(x=345, y=300,anchor=tk.CENTER)

    # ===Button=== データ閲覧
    Button4 = tk.Button(root,text=u'データの閲覧',width=20,font=("Helvetica",14,"bold"),foreground='#FFFFFF',background='#CC0000')
    Button4.bind("<Button>", button_func)
    Button4.place(x=230, y=370,anchor=tk.CENTER) 

    # メッセージ表示(パスワード入力)'パスワードを入力してください'
    msg = tk.Label(root,text="",font=("Helvetica",15),foreground= '#000000',background='#CCCCCC')
    msg.place(x=230, y=430,anchor=tk.CENTER) 

    # userID
    #input_address_label = tk.Label(text="name")
    #input_address = tk.Entry(width=40)
       
    # ===ラベル=== パスワード
    ipl = tk.Label(root,text="password",font=("Helvetica",14,"bold"),foreground='#000000',background='#CCCCCC')
    ipl.place(x=50, y=470) 
    
    # パスワード入力欄の作成
    input_pass = tk.Entry(root,show='*',width=60)
    input_pass.place(x=50, y=500) 

    # 実行ボタン
    Button5 = tk.Button(root,text=u'Enter',width=10,font=("Helvetica",12),foreground='#FFFFFF',background='#CC0000',command=button_click)
    Button5.place(x=320, y=550) 

    # ===Button=== システム終了
    Button6= tk.Button(root,text=u'終了',width=5,font=("Helvetica",14,"bold"),foreground='#FFFFFF',background='#000000')
    Button6.bind("<Button>", button_func) #ボタンのイベント処理
    Button6.place(x=50, y=620,anchor=tk.CENTER)

    # ===Button=== ログアウト
    Button7= tk.Button(root,text=u'設定モード終了',width=15,font=("Helvetica",14,"bold"),foreground='#FFFFFF',background='#000000')
    Button7.bind("<Button>", button_func) #ボタンのイベント処理
    Button7.place(x=350, y=620,anchor=tk.CENTER)
    
    root.protocol("WM_DELETE_WINDOW",click_close)
    root.mainloop()


#全ファイル走査
files = glob.glob(prePic_path + '*')
face_list =[]
face_Nlist =[]
entering = []   ##入室したか否か 0:未入場,1:入場,2:遅刻,3:欠席
exiting = []  ##退室したか否か boolean型
print(" --- Data List --- ")
i=0
for f in files:
    img = cv2.imread(f)    
    img = shuffle_back(img)  #スクランブル化 解除
    f = f.replace('preFaces\\', '')
    f = f.replace('.jpg', '')   
    f = f.replace('.png', '')   
    cv2.imwrite(pic_path+f+".jpg", img)
    print(i+1,": ",f)
    img = face_cut(img)

    face_Nlist.append(f)
    face_list.append(img)
    entering.append(0)
    exiting.append(False)
    i+=1
print(" ----------------- ")
print("")

face_cascade_path = './haarcascades/haarcascade_frontalface_default.xml'
eye_cascade_path = './haarcascades/haarcascade_eye.xml'

face_cascade = cv2.CascadeClassifier(face_cascade_path)
eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

# 初期画面
dt_st = datetime.datetime.now()
home()
