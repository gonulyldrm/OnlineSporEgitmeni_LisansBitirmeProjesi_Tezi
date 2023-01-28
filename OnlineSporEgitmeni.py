import tkinter as tk
from tkinter.ttk import *
import sqlite3
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import cv2
from tkinter import filedialog
import mediapipe as mp
import sklearn.model_selection as sss
from tkinter import messagebox
from tkinter import *
from PIL import Image,ImageTk
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression



con=sqlite3.connect("onlinespor2.db")
cursor=con.cursor()
def tabloolustur():
    cursor.execute("CREATE TABLE IF NOT EXISTS zayif (gunId int,kurallar TEXT)")
    cursor.execute("CREATE TABLE IF NOT EXISTS normal (gunId int,kurallar TEXT)")
    cursor.execute("CREATE TABLE IF NOT EXISTS kilolu (gunId int,kurallar TEXT)")
    cursor.execute("CREATE TABLE IF NOT EXISTS obez (gunId int,kurallar TEXT)")
    cursor.execute("CREATE TABLE IF NOT EXISTS asiri_obez (gunId int,kurallar TEXT)")
    cursor.execute("CREATE TABLE IF NOT EXISTS editörler (kullaniciad TEXT,sifre TEXT)")

def degerekle():
    cursor.execute("INSERT INTO zayif VALUES ('1','zayıf kişilerinde diyeti mutlaka kişiye özgü olmalıdır.Diyetin enerji değeri harcanan enerjiden daha fazla olmalıdır.Diyetin protein kalitesi yüksek olmalıdır. ...Diyet vitamin ve minerallerden zengin olmalıdır.')")
    cursor.execute("INSERT INTO normal VALUES ('1','BKI değeriniz ideal kiloda olsa da sağlıklı besleniyor olmak her zaman için en öenmli hedef olamalıdır. Sağlıklı beslenme denilince ilk akla gelen sebze ve meyvelerdir. Ancak sebzenin pişirilme şekli yenilen meyvenin miktarı göz ardı edilmemelidir. Bununla birlikte süt – yoğurt, tahıllar, et ürünleri ve baklagiller de tabi ki beslenmenin önemli parçalarıdır. Kızartma ya da kavurma yerine haşlama, ızgara ya sulu pişirme yöntemlerini tercih etmenizi öneririm. Şeker ve tuz tüketimini azaltın. Izgaranızı sadece et tavuk ve balık için değil, sebzeler için de kullanın. Domates, biber, kabak, mantar vs. Beyaz ekmek yerine tam buğday ekmek sizin için daha sağlıklı bir seçim olacaktır')")
    cursor.execute("INSERT INTO kilolu VALUES ('1','Kilo almaktan korunmak için günlük kalori ihtiyacınız kadar beslenmek ve mutlaka egzersiz yapmak gereklidir. Özellikle hazır gıdalardan ve restoran yemeklerinden uzak durulmalıdır. Haftanın en azından 4-5 günü toplam süre en az 150 dk/hafta olacak şekilde egzersiz yapılmalıdır. Ayrıca yürüme mesafesindeki yerler içim taşıt kullanılmaması, otobüsten bir durak önce inilmesi, asansör yerine merdiven kullanılması, arabanın mümkün olduğunca uzak yerlere park edilmesi yararlı olabilir. Egzersiz yapmayı bir yaşam biçimi olarak benimsemek ve sürdürmek önemlidir.')")
    cursor.execute("INSERT INTO obez VALUES ('1','3 ana öğün dışında, 3 ara öğün yenilmeli. Ara öğün metabolizmayı hızlandırır. Yenilen yemeğin saati önemli değil. Önemli olan yemeği yedikten sonra üzerinden en az iki saat geçtikten sonra uyumak. Yemeği yavaş yemek çok önemli. Çünkü beyne tokluk sinyali, yemek yemeye başladıktan 20 dakika sonra gider. Yemek yavaş yendiğinde mide besinleri parçalama işlemini çok daha verimli bir biçimde yapar. Böylece göbeklenmenin de önüne geçilir.Günde 12 bardak su içmeli')")
    cursor.execute("INSERT INTO asiri_obez VALUES ('1','Her gün 10 dakika egzersiz yapın. ...Güne kahvaltı ile başlayın. ...Tetikleyici gıdalardan kurtulun. ...Özel kalori, protein ve karbonhidrat hedefi olan yemek tarifleri bulun. ...yeni, sağlıklı besinler için alışverişe çıkın.')")
    cursor.execute("INSERT INTO editörler VALUES ('ali1','123')")
    con.commit()
    con.close()
def selectveri(tabloadi,yazi):
    sql='SELECT kurallar FROM '+tabloadi
    cursor.execute(sql)
    veri=cursor.fetchall()
    for i in veri:
        print(i)
        yazi["text"]=str(i)

tabloolustur()
#degerekle()
def anasayfa():
    def editorpen():
        def girisyap():
            kullanicii=kullanici.get()
            sifree=sifre.get()

            sqll = f"SELECT kullaniciad from editörler WHERE kullaniciad='{kullanicii}' AND sifre = '{sifree}';"
            cursor.execute(sqll)
            if not cursor.fetchone():
                messagebox.showerror("showerror", "Hatali Giris... TEkRAR DENEYİN...")
            else:
                pen5.destroy()
                penEd=tk.Tk()
                def calisanMaas():
                    data = pd.read_csv('kidem_maas.csv')
                    print(data.head())
                    feature_cols = ['Kidem']
                    X = data[feature_cols]
                    y = data.Maas
                    X_train, X_test, y_train, y_test = sss.train_test_split(X, y, test_size=0.25, random_state=0)
                    lr = LogisticRegression()
                    lr.fit(X_train, y_train)
                    y_pred = lr.predict(X_test)
                    plt.scatter(X_train, y_train, color='red')
                    plt.title('Kıdeme Göre Maaş Tahmini Regresyon Modeli')
                    plt.xlabel('Kıdem')
                    plt.ylabel('Maaş')
                    plt.show()

                penEd.configure(bg='#94af8b')
                penEd.title("Spor Egitmeni")
                penEd.geometry("501x500")
                lbl = Label(text="Umut Spor Salonu", font="verdana 22 bold", bg='#94af8b')
                lbl.place(x=85, y=210)
                def anasayffa():
                    penEd.destroy()
                    anasayfa()
                ansayfa = tk.Button(text="Anasayfa", font="verdana 14 bold", command=anasayffa)
                ansayfa.place(x=10, y=450)
                logo = ImageTk.PhotoImage(Image.open("C://Users//LENOVO//OneDrive//Masaüstü//İAÜ//4.sınıf//PROJE GELİŞTİRMENİN TEMELLERİ//logoo.png"))
                logoo = Label(image=logo).place(x=132, y=20)
                tablo = tk.Button(text="TABLO GÖSTER ", font="verdana 16 bold", command=calisanMaas)
                tablo.place(x=150, y=300)
                penEd.mainloop()
                print("Welcome")

        pen.destroy()
        pen5=tk.Tk()
        pen5.configure(bg='#c9edae')
        pen5.title("Spor Egitmeni")
        pen5.geometry("501x500")
        lbll = tk.Label(text="Umut Spor Salonu", font="verdana 22 bold",bg='#c9edae')
        lbll.place(x=90, y=210)
        def anasayffa():
            pen5.destroy()
            anasayfa()
        ansayfa = tk.Button(text="Anasayfa", font="verdana 14 bold", command=anasayffa)
        ansayfa.place(x=10, y=450)
        kullanicilbl = tk.Label(text="Kullanıcı Adı:", font="verdana 16  ",bg='#c9edae')
        kullanicilbl.place(x=15, y=280)
        kullanici = tk.Entry(font="verdana 18 ")
        kullanici.place(x=150, y=280)
        sifre = tk.Entry(font="verdana 18 ",show="*")
        sifre.place(x=150, y=320)
        sifrelbl = tk.Label(text="Şifre :", font="verdana 18  ",bg='#c9edae')
        sifrelbl.place(x=75, y=320)
        giris = tk.Button(text="Giriş Yap ", font="verdana 16 bold", command=girisyap)
        giris.place(x=170, y=360)
        logo = ImageTk.PhotoImage(Image.open("C://Users//LENOVO//OneDrive//Masaüstü//İAÜ//4.sınıf//PROJE GELİŞTİRMENİN TEMELLERİ//logoo.png"))
        logoo = Label(image=logo)
        logoo.place(x=150, y=20)
        pen5.mainloop()

    def kullanicipen():
        def index():
            def indexsonuc():
                beslenmelbl = tk.Label(text="beslenme",font="verdana 11 ",wraplength=500)
                beslenmelbl.place(x=5, y=250)
                sonuclbl = tk.Label(text="sonuc", font="verdana 15 ")
                sonuclbl.place(x=200, y=210)
                boyy=int(boy.get())
                kiloo=int(kilo.get())
                cinss=int(k)
                data = pd.read_csv("500_Person_Gender_Height_Weight_Index.csv")
                data=data.replace({"Male":1,"Female":0})
                print(data.head())
                feature_cols = ['Gender','Height', 'Weight']
                X = data[feature_cols]
                y = data.Index
                X_train, X_test, y_train, y_test = sss.train_test_split(X, y, test_size=0.25, random_state=0)
                lm = LinearRegression()
                lm.fit(X_train, y_train)
                y_pred = lm.predict(X_test)
                xn = [[cinss,boyy,kiloo]]
                yn = lm.predict(xn)
                #sonuclbl["text"]=str(yn)
                print(yn)
                tabloadii="aa"
                if int(yn)==1:
                    tabloadii="zayif"
                    sonuclbl["text"] = tabloadii
                elif int(yn)==2:
                    tabloadii="normal"
                    sonuclbl["text"] = tabloadii
                elif int(yn)==3:
                    tabloadii="kilolu"
                    sonuclbl["text"] = tabloadii
                elif int(yn)==4:
                    tabloadii="obez"
                    sonuclbl["text"] = tabloadii
                elif int(yn)==5:
                    tabloadii="asiri_obez"
                    sonuclbl["text"] = tabloadii
                else:
                    sonuclbl["text"]="Hesaplanamadı"
                selectveri(tabloadii,beslenmelbl)



            pen2.destroy()
            pen3 = tk.Tk()
            pen3.configure(bg='#c9abc0')
            pen3.title("Spor Egitmeni")
            pen3.geometry("501x500")
            lbll = tk.Label(text="Umut Spor Salonu", font="verdana 22 bold",bg='#c9abc0')
            lbll.place(x=70, y=10)
            boylbl= tk.Label(text="Boy(cm)", font="verdana 18  ",bg='#c9abc0')
            boylbl.place(x=20,y=90)
            boy=tk.Entry(font="verdana 18 ")
            boy.place(x=130,y=90)
            kilo = tk.Entry(font="verdana 18 ")
            kilo.place(x=130, y=130)
            kilolbl = tk.Label(text="Kilo(kg)", font="verdana 18  ",bg='#c9abc0')
            kilolbl.place(x=20,y=130)
            indexbtn=tk.Button(text="Sonuc ve Beslenme Kurallari",font="verdana 14 bold",command=indexsonuc)
            indexbtn.place(x=100,y=170)
            def anasayffa():
                pen3.destroy()
                anasayfa()
            ansayfa = tk.Button(text="Anasayfa", font="verdana 14 bold", command=anasayffa)
            ansayfa.place(x=10, y=450)
            cinslbl=tk.Label(text="Cinsiyet", font="verdana 18  ",bg='#c9abc0')
            cinslbl.place(x=40,y=50)
            combo=Combobox()
            combo["font"]=("verdana 16")
            combo["values"]=("KADIN","ERKEK")
            combo.place(x=145,y=50)
            k=5
            if combo.get()=="KADIN":
                k=0
            elif combo.get()=="ERKEK":
                k=1
            pen3.mainloop()
        def diyabet():
            def diabetsonuc():
                data = pd.read_csv("diabetes.csv")
                print(data.head())
                bmii=bmi.get()
                glikozz=glikoz.get()
                yass=yas.get()
                tansiyonn=tansiyon.get()
                feature_cols = ['Glucose', 'BloodPressure', 'BMI', 'Age']
                X = data[feature_cols]
                y = data.Outcome
                X_train, X_test, y_train, y_test = sss.train_test_split(X, y, test_size=0.25, random_state=0)
                model = KMeans(n_clusters=2)
                model.fit(X)
                predY = np.choose(model.labels_, [1, 0, 2]).astype(np.int64)
                xn = [[glikozz, tansiyonn, bmii,yass]]
                yn = model.predict(xn)
                sonuclbl = tk.Label(text="sonuc", font="verdana 12  bold ")
                sonuclbl.place(x=5, y=290)
                if int(yn)==1:
                    sonuclbl["text"]="Degerlere göre diyabet hastası olma ihtimaliniz YÜKSEK"
                elif int(yn)==0:
                    sonuclbl["text"]="Degerlere göre diyabet hastası olma ihtimaliniz DÜŞÜK"
                else:
                    sonuclbl["text"]="Hesaplanamadı"

            pen2.destroy()
            pen4 = tk.Tk()
            pen4.title("Spor Egitmeni")
            pen4.configure(bg='#6dabfa')
            pen4.geometry("501x500")
            lbl = tk.Label(text="Umut Spor Salonu", font="verdana 22 bold",bg='#6dabfa')
            lbl.place(x=60, y=50)
            glikozlbl = tk.Label(text="Glikoz:", font="verdana 18  ",bg='#6dabfa')
            glikozlbl.place(x=45, y=110)
            glikoz = tk.Entry(font="verdana 18 ")
            glikoz.place(x=130, y=110)
            tansiyon = tk.Entry(font="verdana 18")
            tansiyon.place(x=130, y=140)
            tansiyonlbl = tk.Label(text="Tansiyon:", font="verdana 18  ",bg='#6dabfa')
            tansiyonlbl.place(x=10, y=140)
            def anasayffa():
                pen4.destroy()
                anasayfa()
            ansayfa = tk.Button(text="Anasayfa", font="verdana 14 bold", command=anasayffa)
            ansayfa.place(x=10, y=450)
            bmi = tk.Entry(font="verdana 18 ")
            bmi.place(x=130, y=170)
            bmilbl = tk.Label(text="BMİ:", font="verdana 18  ",bg='#6dabfa')
            bmilbl.place(x=60, y=170)
            yas = tk.Entry(font="verdana 18 ")
            yas.place(x=130, y=200)
            yaslbl = tk.Label(text="Yaş:", font="verdana 18  ",bg='#6dabfa')
            yaslbl.place(x=60, y=200)
            diabetbut = tk.Button(text="Diyabet Tahmin", font="verdana 16 bold", command=diabetsonuc)
            diabetbut.place(x=140, y=240)
            pen4.mainloop()

        def hareketkontrol():
            def waistcurl():
                filepath = filedialog.askopenfilename()
                cap = cv2.VideoCapture(filepath)
                mpcizim = mp.solutions.drawing_utils
                mppoz = mp.solutions.pose
                counter = 0
                stage = None

                def acilar(a, b, c):
                    a = np.array(a)
                    b = np.array(b)
                    c = np.array(c)
                    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
                    aci = np.abs(radians * 180.0 / np.pi)
                    if aci > 180.0:
                        aci = 360 - aci
                    return aci

                with mppoz.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
                    while cap.isOpened():
                        ret, frame = cap.read()
                        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        image.flags.writeable = False
                        results = pose.process(image)
                        image.flags.writeable = True
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                        try:
                            landmarks = results.pose_landmarks.landmark
                            omuz = [landmarks[mppoz.PoseLandmark.RIGHT_SHOULDER.value].x,
                                        landmarks[mppoz.PoseLandmark.RIGHT_SHOULDER.value].y]
                            bel = [landmarks[mppoz.PoseLandmark.RIGHT_HIP.value].x,
                                     landmarks[mppoz.PoseLandmark.RIGHT_HIP.value].y]
                            diz = [landmarks[mppoz.PoseLandmark.RIGHT_KNEE.value].x,
                                     landmarks[mppoz.PoseLandmark.RIGHT_KNEE.value].y]

                            aci = acilar(omuz,bel, diz)
                            cv2.putText(image, str(aci),
                                        tuple(np.multiply(bel, [640, 480]).astype(int)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                            if aci > 70:
                                stage = "kaldir"
                            if aci < 60 and stage == 'kaldir':
                                stage = "indir"
                                counter += 1
                                print(counter)


                        except:
                            pass

                        cv2.rectangle(image, (0, 0), (225, 73), (245, 117, 16), -1)
                        cv2.putText(image, 'tekrar', (15, 12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                        cv2.putText(image, str(counter),
                                    (10, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

                        cv2.putText(image, 'asamasi', (65, 12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                        cv2.putText(image, stage,
                                    (60, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)




                        mpcizim.draw_landmarks(image, results.pose_landmarks, mppoz.POSE_CONNECTIONS,
                                                  mpcizim.DrawingSpec(color=(245, 117, 66), thickness=2,
                                                                         circle_radius=2),
                                                  mpcizim.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

                        cv2.imshow('Mediapipe  ', image)

                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

                    cap.release()
                    cv2.destroyAllWindows()

            def armcurl():
                filepath = filedialog.askopenfilename()
                cap = cv2.VideoCapture(filepath)
                mpcizim = mp.solutions.drawing_utils
                mppoz = mp.solutions.pose
                counter = 0
                stage = None

                def acilar(a, b, c):
                    a = np.array(a)
                    b = np.array(b)
                    c = np.array(c)
                    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
                    aci = np.abs(radians * 180.0 / np.pi)
                    if aci > 180.0:
                        aci = 360 - aci
                    return aci

                with mppoz.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
                    while cap.isOpened():
                        ret, frame = cap.read()
                        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        image.flags.writeable = False
                        results = pose.process(image)
                        image.flags.writeable = True
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                        try:
                            landmarks = results.pose_landmarks.landmark
                            omuz = [landmarks[mppoz.PoseLandmark.LEFT_SHOULDER.value].x,
                                        landmarks[mppoz.PoseLandmark.LEFT_SHOULDER.value].y]
                            dirsek = [landmarks[mppoz.PoseLandmark.LEFT_ELBOW.value].x,
                                     landmarks[mppoz.PoseLandmark.LEFT_ELBOW.value].y]
                            bilek = [landmarks[mppoz.PoseLandmark.LEFT_WRIST.value].x,
                                     landmarks[mppoz.PoseLandmark.LEFT_WRIST.value].y]

                            aci = acilar(omuz,dirsek, bilek)
                            cv2.putText(image, str(aci),
                                        tuple(np.multiply(dirsek, [640, 480]).astype(int)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                            if aci > 160:
                                stage = " asagi"
                            if aci < 30 and stage == ' asagi':
                                stage = " yukari"
                                counter += 1
                                print(counter)

                            omuz2 = [landmarks[mppoz.PoseLandmark.RIGHT_SHOULDER.value].x,
                                    landmarks[mppoz.PoseLandmark.RIGHT_SHOULDER.value].y]
                            dirsek2 = [landmarks[mppoz.PoseLandmark.RIGHT_ELBOW.value].x,
                                      landmarks[mppoz.PoseLandmark.RIGHT_ELBOW.value].y]
                            bilek2 = [landmarks[mppoz.PoseLandmark.RIGHT_WRIST.value].x,
                                     landmarks[mppoz.PoseLandmark.RIGHT_WRIST.value].y]
                            aci2 = acilar(omuz2, dirsek2, bilek2)
                            cv2.putText(image, str(aci),
                                        tuple(np.multiply(dirsek2, [640, 480]).astype(int)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2, cv2.LINE_AA)

                            if aci2 > 160:
                                stage2 = "  asagi"
                            if aci2 < 30 and stage2 == '  asagi':
                                stage2= "  yukari"
                                counter += 1
                                print(counter)
                        except:
                            pass

                        cv2.rectangle(image, (0, 0), (225, 73), (245, 117, 16), -1)
                        cv2.putText(image, 'tekrar', (15, 12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                        cv2.putText(image, str(counter),
                                    (10, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

                        cv2.putText(image, 'asamasi', (65, 12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                        cv2.putText(image, stage,
                                    (60, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)




                        mpcizim.draw_landmarks(image, results.pose_landmarks, mppoz.POSE_CONNECTIONS,
                                                  mpcizim.DrawingSpec(color=(245, 117, 66), thickness=2,
                                                                         circle_radius=2),
                                                  mpcizim.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

                        cv2.imshow('Mediapipe  ', image)

                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

                    cap.release()
                    cv2.destroyAllWindows()

            def bacak_curl():
                filepath = filedialog.askopenfilename()
                cap = cv2.VideoCapture(filepath)
                mpcizim = mp.solutions.drawing_utils
                mppoz = mp.solutions.pose
                counter = 0
                stage = None

                def acilar(a, b, c):
                    a = np.array(a)
                    b = np.array(b)
                    c = np.array(c)
                    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
                    aci = np.abs(radians * 180.0 / np.pi)
                    if aci > 180.0:
                        aci = 360 - aci
                    return aci

                with mppoz.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
                    while cap.isOpened():
                        ret, frame = cap.read()
                        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        image.flags.writeable = False
                        results = pose.process(image)
                        image.flags.writeable = True
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                        try:
                            landmarks = results.pose_landmarks.landmark
                            kalca = [landmarks[mppoz.PoseLandmark.LEFT_HIP.value].x,
                                        landmarks[mppoz.PoseLandmark.LEFT_HIP.value].y]
                            diz = [landmarks[mppoz.PoseLandmark.LEFT_KNEE.value].x,
                                     landmarks[mppoz.PoseLandmark.LEFT_KNEE.value].y]
                            bilek = [landmarks[mppoz.PoseLandmark.LEFT_ANKLE.value].x,
                                     landmarks[mppoz.PoseLandmark.LEFT_ANKLE.value].y]

                            aci = acilar(kalca,diz, bilek)
                            cv2.putText(image, str(aci),
                                        tuple(np.multiply(diz, [640, 480]).astype(int)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                            if aci > 160:
                                stage = " asagi"
                            if aci < 30 and stage == ' asagi':
                                stage = " yukari"
                                counter += 1
                                print(counter)

                            kalca2 = [landmarks[mppoz.PoseLandmark.RIGHT_HIP.value].x,
                                    landmarks[mppoz.PoseLandmark.RIGHT_HIP.value].y]
                            diz2 = [landmarks[mppoz.PoseLandmark.RIGHT_KNEE.value].x,
                                      landmarks[mppoz.PoseLandmark.RIGHT_KNEE.value].y]
                            bilek2 = [landmarks[mppoz.PoseLandmark.RIGHT_ANKLE.value].x,
                                     landmarks[mppoz.PoseLandmark.RIGHT_ANKLE.value].y]

                            aci2 = acilar(kalca2, diz2, bilek2)
                            cv2.putText(image, str(aci),
                                        tuple(np.multiply(diz2, [640, 480]).astype(int)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2, cv2.LINE_AA)

                            if aci2 > 160:
                                stage2 = "  asagi"
                            if aci2 < 90 and stage2 == '  asagi':
                                stage2= "  yukari"
                                counter += 1
                                print(counter)
                        except:
                            pass

                        cv2.rectangle(image, (0, 0), (225, 73), (245, 117, 16), -1)
                        cv2.putText(image, 'tekrar', (15, 12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                        cv2.putText(image, str(counter),
                                    (10, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

                        cv2.putText(image, 'asamasi', (65, 12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                        cv2.putText(image, stage,
                                    (60, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)




                        mpcizim.draw_landmarks(image, results.pose_landmarks, mppoz.POSE_CONNECTIONS,
                                                  mpcizim.DrawingSpec(color=(245, 117, 66), thickness=2,
                                                                         circle_radius=2),
                                                  mpcizim.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

                        cv2.imshow('Mediapipe  ', image)

                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

                    cap.release()
                    cv2.destroyAllWindows()

            pen2.destroy()
            pen6 = tk.Tk()
            pen6.title("Spor Egitmeni")
            pen6.configure(bg='#fdff7f')
            pen6.geometry("501x500")
            lbl = tk.Label(text="Umut Spor Salonu", font="verdana 22 bold",bg='#fdff7f')
            lbl.place(x=85, y=150)
            def anasayffa():
                pen6.destroy()
                anasayfa()
            ansayfa = tk.Button(text="Anasayfa", font="verdana 14 bold", command=anasayffa)
            ansayfa.place(x=10, y=450)
            but = tk.Button(text="Arms Curl", font="verdana 18 bold", command=armcurl)
            but2 = tk.Button(text="Legs Curl", font="verdana 18 bold", command=bacak_curl)
            but3 = tk.Button(text="Waist Thinning", font="verdana 18 bold", command=waistcurl)
            but.place(x=150, y=230)
            but2.place(x=50, y=300)
            but3.place(x=210, y=300)
            pen2.mainloop()

        pen.destroy()
        pen2=tk.Tk()
        pen2.title("Spor Egitmeni")
        pen2.configure(bg='#e78eae')
        pen2.geometry("501x500")
        lbl = tk.Label(text="Umut Spor Salonu", font="verdana 22 bold",bg='#e78eae')
        lbl.place(x=110, y=200)
        def anasayffa():
            pen2.destroy()
            anasayfa()
        ansayfa = tk.Button(text="Anasayfa", font="verdana 14 bold", command=anasayffa)
        ansayfa.place(x=10, y=450)
        logo = ImageTk.PhotoImage(Image.open("C://Users//LENOVO//OneDrive//Masaüstü//İAÜ//4.sınıf//PROJE GELİŞTİRMENİN TEMELLERİ//logoo.png"))
        logoo = Label(image=logo)
        logoo.place(x=150, y=20)
        but=tk.Button(text="İndexe göre Beslenme",font="verdana 18 bold",command=index)
        but2=tk.Button(text="Hareket Kontrol",font="verdana 18 bold",command=hareketkontrol)
        but3=tk.Button(text="Diyabet Tahmin",font="verdana 18 bold",command=diyabet)
        but.place(x=100,y=280)
        but2.place(x=10,y=350)
        but3.place(x=260,y=350)
        pen2.mainloop()

    pen=tk.Tk()
    pen.minsize()
    pen.title("Spor Egitmeniii")
    pen.configure(bg='#c28eae')
    pen.geometry("500x500")
    lbl=Label(text="Umut Spor Salonu",font="verdana 22 bold",bg='#c28eae')
    lbl.place(x=70,y=210)
    logo=ImageTk.PhotoImage(Image.open("C://Users//LENOVO//OneDrive//Masaüstü//İAÜ//4.sınıf//PROJE GELİŞTİRMENİN TEMELLERİ//logoo.png"))
    logoo=Label(image=logo).place(x=120,y=20)
    b=Button(text="Editor Girisi",font="verdana 20  ",command= editorpen).place(x=135,y=340)
    b2=Button(text="Kullanıcı Girisi",font="verdana 18  ",command=kullanicipen).place(x=135,y=280)
    pen.mainloop()

anasayfa()