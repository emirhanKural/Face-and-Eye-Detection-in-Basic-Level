import cv2
import imageio

#Oluşturulmuş modelleri aldık.
face_cascade=cv2.CascadeClassifier("haarcascade-frontalface-default.xml")
eye_cascade=cv2.CascadeClassifier("haarcascade-eye.xml")

def detect(frame):
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) #Griye çevirik resmimizi

    # yakalayacağımız resmi 1.3 oranında küçülttük ve konrtolün 5 pencere etrafında olmasını sağladık
    faces=face_cascade.detectMultiScale(gray,1.3,5)

    #şimdi yüzleri belirlemek için kare oluşturucaz.Her yüze çizmek için
    #x,y kordinatı; w,h genişliği ve yüksekliği temsil ediyor.
    for(x,y,w,h) in faces:
        #Dikdörgen çiziyoruz yüzler için
        #aldığımız resim,resmin sol-üstün koordinatları,sağ-alt koordinat,kırmızı bir dikdörtgen,2 kalınlığında
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

        #şimdi yüz içindeki gözleri belirlemek istiyoruz
        #Resimde tam yüzün olduğu kısmı alıyoruz
        #y'den y+h'a kadar, x'den x+w'ya kadar. yani tam dikdörtgeni alıyoruz.
        gray_face=gray[y:y+h,x:x+w]

        #aynı koordinatları orjinal resim içinde alıyoruz.
        color_face=frame[y:y+h,x:x+w]

        #face için yaptığımız şeyi şimdi göz için yapıyoruz.
        eyes = eye_cascade.detectMultiScale(gray_face, 1.1, 3)

        #gözleri bulmak için for döngüsü oluşturalım şimdide
        for (ex, ey, ew, eh) in eyes:
            #bulduğumuz değerleri orjinal resimde gösterme vakti
            cv2.rectangle(color_face,(ex,ey),(ex+ew,ey+eh),(0,0,255),2)

    return frame

reader=imageio.get_reader("1.mp4") #İnceleyeceğimiz şeyi aldık
fps=reader.get_meta_data()["fps"]#videodaki frameleri alıyoruz
writer=imageio.get_writer("out.mp4",fps=fps)#çıkan videonun ismi ve kaç fps olacağı. Giren videonunki neyse okadar olsun.

#reader ile aldığımı her frame için for döngüsü dönecek.
#enumerate'in for'dan farkı 2 parametre alıyor.
# Biri işlenecek olan değişkeni sayan sayaç(i)
# İkincisi değişkenimizim kendisi. Her frame'e bir sayı atılıyor.
for i,frame in enumerate(reader):
    frame=detect(frame) #ve fonksiyona yolluyoruz.
    writer.append_data(frame)#Çizilmiş frameleri alıp output videosuna ekliyoruz.
    print(i)
writer.close()












