#kullanılacak kütüphanelerin eklenmesi
import numpy as np #nd.array  ve çeşitli istatiksel hesaplamaları yapabilmek için numpy kütüphanesinin eklenmesi
import pandas as pd #csv uzantılı veriyi okuyabilmek ve dataframe ya da serileri kullanabilmek için pandas kütüphanesinin eklenmesi
import matplotlib.pyplot as plt #grafik çizdirmek için matplotlib.pyplot eklenmesi
from scipy import stats as st #mod almak için kullanılan scipy  modülü
from sklearn.linear_model import LinearRegression #regresyon modelini uygulayabilmek için kullanılan sklearn.linear_model modülü 
from math import sqrt #kare almak için kullanılan math modülü
from sklearn.metrics import mean_squared_error #mse bulabilmek için kullanılan sklearn.metrics modülü
df=pd.read_csv("vize dataset/forbes_celebrity_100.csv") #csv uzantılı dosyayı pandas ile okuma.
print(type(df))#df nin tipini gösterme
print(df) 
print("ndim : {}".format(df.ndim)) #dataframe boyutunu bulma(rank)
print("shape : {}".format(df.shape))#satr,sütun sayısını bulma
print("size : {}".format(df.size))#toplam eleman sayısını verir.
# print(df.head(10))# veri setinden ilk beş satırı gösterir.
# print(df.tail(5))# veri setinden son beş satırı gösterir(her ikisine de istenilen satır sayısı kadar sayı gönderebiliriz)
print(df.columns) #sütun isimlerini bir liste içerisinde döndürür.
# print("dtype : {}".format(df.dtypes)) dataframe in içerdiği verilerin tipini döndürür.
print(df.info()) #dataframe hakında ayrıntılı bilgi verir. # veride her sütun için null veri olup olmadığını burada da görebiliriz 
print(df.iloc[0:4][["Name"]]) #sütun bazlı arama [0,4)
# print(df.loc[0:99,["Pay (USD millions)","Name","Year"]]) #indis bazlı arama [0,99]
print(df[["Name","Category"]].tail(7)) # name ve category sütunlarının son 7 satırını gösterir.
d0=df[["Pay (USD millions)"]]#dataframeler iki boyutlu olduğu için çift parantezle df nin Pay (USD millions) sütunun değerlerini d0 değişkenine atama
print(type(d0)) #d0 ın tipini gösterme
print(df["Pay (USD millions)"])#df de Pay (USD millions) sütununu arama bulunan sütunun ismini uzunluğunu ve içerdiği veri tipini döndürme 
d1 =np.array(d0)#d0 dataframe ini ndarraye çevirme 
print("Ortalama  = {} ".format(np.mean(d1))) #pay sütunun değerlerinin ortalamasını bulma
print("Mod       = {} ".format(st.mode(d1)))#pay sütunun modunu bulma #mod değişmez duyarsız bir değer.
print("Medyan    = {} ".format(np.median(d1)))#pay sütunun ortancasını bulma
print("Aralık    = {} ".format(np.ptp(d1)))# pay sütünun aralığını bulma
print("Varyans   = {} ".format(np.var(d1)))#pay sütunun varyansını bulma  #mod ve medyan duyarlı olmayan ortalamalar diye geçer
print("Standart Sapma = {} ".format(np.std(d1)))#pay sütunun değerlerinin standart sapmasını bulma.  
print("Maksimum Değer  = {} ".format(np.max(d1))) # pay sütünun maximum değerini bulma
print("Minumum  Değer  = {} ".format(np.min(d1))) # pay sütünun minimum değerini bulma
#kartiller
print("Q1 : {} ".format(np.percentile(d1,25))) # ilk çeyrek verilerin % 25 ini içerir
print("Q2 : {} ".format(np.percentile(d1,50))) # ikinci çeyrek verilerin % 50 sini içerir
print("Q3 : {} ".format(np.percentile(d1,75))) # üçüncü çeyrek verilerin % 75 ini içerir.
Q1 = np.percentile(d1,25)
Q2 = np.percentile(d1,50)
Q3 = np.percentile(d1,75)
IQR = (Q3-Q1)*1.5 #çeyrekler aralığının bulunması 
print("IQR : {}".format(IQR))					
print("Alt Uç Değer : {} ".format(Q1-1.5*IQR))  # Alt Uç Değer : -39.25 böyle bir kazanç bulunamayacağından alt uç değerimiz yok
print("Üst Uç Değer : {} ".format(Q3+1.5*IQR))  #Üst Uç Değer : 120.25
print("Uç değerler \n",df[df["Pay (USD millions)"]>=120][["Pay (USD millions)","Name","Year","Category"]]) # uç değer kabul edilen verilerin tüm bilgilerini gösterir.
std=np.std(d1)
ort=np.mean(d1)
med=np.median(d1)
çk=((ort - med)/std) * 3 #çarpıklık katsayısı formulü ve bulunması.
"""
Sağa çarğık dağılımlarda ; 
Değerlerin yarıdan fazlası aritmetik ortalamanın altındadır yani başarısız bir gruptur. 
Dağılım düşük puanlarda yığılır. Pozitif kayışlıdır. Test zordur. Başarı düşüktür ve öğrenme düzeyi düşüktür.
çarpıklık katsayısı = 0.5521916630565843 verim sağ çarpık çünkü çarpıklık katsayısının işareti pozitif.
sağa çarğık dağılım olduğu için Mod = 30 < medyan = 39 < aritmetik ortalama = 46.16
"""
print("çarpıklık katsayısı={}".format(çk))
#kayıp veri kontrolü
NaN=df["Pay (USD millions)"].isnull().sum() #kayıp verilerin toplam kaç adet olduğunu  bulma
pay=df["Pay (USD millions)"]#pay değişkenine df nin pay isimli sütununu atama
d1=pay.ravel() #sütunu düzleştirip ndarray içinde döndürme
yüzde=(NaN/d1.size)*100 #verilerin ne kadarının kayıp olduğunu % cinsinden hesaplama
print("verilerin %{} eksik".format(yüzde))
# d1=np.random.normal(46.1,38.9,d0.size) #çan eğrisi için gerekli normal dağılıma uygun ve ortalama civarında oluşturulan ndarray
print(d1) #düzleştirilen ndarrayi ekrana yazdırma 
#bu histogram hangi yıllık kazanca kaç kişinin sahip olduğunu gösteren bir dağılım göstermektedir.
plt.hist(d1,50)#d1 verisinin değerlerini 50 adet bar içeren bir histogram çizme(50 bar bize doğru kişi sayılarını vermez ancak verinin nasıl bir dağılıma sahip olduğunu daha net görmemizi sağlar)
plt.grid() #ızgara ekleme (eğer doğru kişi sayılarını görmek istiyorsak toplam kişi sayısı kadar bar çizdirmeliyiz.)
#Yukarıda bulunan çarpıklık katsıyısı ile vardığımız sonucu dağılımı görerek inceledik gerçektende verimizin sağa çarpık dağılım olduğunu gördük.
plt.title("YILLIK KAZANCA GÖRE ÜNLÜ KİŞİ SAYISI")#grafik başlığı(veri kazanç sütununa göre nasıl bir dağılıma sahip gözlemleme)
plt.xlabel("YILLIK KAZANÇ (MİLYON DOLAR)") #x ekseninin etiketi
plt.ylabel("ÜNLÜ KİŞİ SAYISI")#y ekseninin etiketi
plt.xlim(0,625) #max değeri görebilmek için belirlenen x ekseni sınırları
plt.show()#histogramı gösterme
y=df["Year"]#yıl sütununu y isimli değişkene atama
d2=y.ravel()#y dataframe ini düzleştirerek ndarray içinde döndürür
print(d2)#2014 yılının max kazancı tüm veri setinin max kazancı olmuş(620 milyon dolar bu değer aynı zamanda bir uç değer)
plt.title("YILLARA GÖRE YILLIK KAZANÇ")
plt.ylabel("YILLIK KAZANÇ (MİLYON DOLAR)") #yıllık kazançların yıllara göre dağılımı 
plt.xlabel("YILLAR (2005-2019)")#burada görüldüğü gibi verimiz doğru orantılı olarak artmamakta ya da azalmamakta ve bu veriye lineer regresyon uygulanırsa mse ve rmse değerleri çok çok yüksek olacağı(yani modelin düşük performanslı olacağı) söylenebilir.
plt.scatter(d2,d1)#d1 ve d2 değişkenlerine bağlı bir serpme grafiği döndürür.                                                          
plt.show() # çizlen herhangi bir grafiği gösterir 
plt.title("YILLARA GÖRE VERİ SAYISI")
plt.hist(df.Year,10)#Her bir yılın içerdiği veri sayısı
plt.show()
### Gruplama ###
adetler = df["Category"].value_counts() #Category isimli sütunun değerelerinden kaçar tane içerdiğini gösterir 
print(adetler)#category sütunun içerisindeki meslekleri sayılarına göre  gruplamış olduk.
yüzn=df["Name"].value_counts()/df.Category.size*100#ünlü isimlerin en çok kazanan ünlüler listesine girebilme yüzdelikleri 
print(yüzn)
#örnekleme #
#yıllara göre max kazançları bulma(lineer regresyon yapabilmek için gerekli veriyi elde ettim çünkü yukarıda açıklandığı gibi veri setinin sayısal sütunları lineer regresyona uygun değil)
a=df[df["Year"]==2005]["Pay (USD millions)"].max()  
b=df[df["Year"]==2006]["Pay (USD millions)"].max()
c=df[df["Year"]==2007]["Pay (USD millions)"].max()
d=df[df["Year"]==2008]["Pay (USD millions)"].max()
e=df[df["Year"]==2009]["Pay (USD millions)"].max()
f=df[df["Year"]==2010]["Pay (USD millions)"].max()
g=df[df["Year"]==2011]["Pay (USD millions)"].max()
h=df[df["Year"]==2012]["Pay (USD millions)"].max()
i=df[df["Year"]==2013]["Pay (USD millions)"].max()
j=df[df["Year"]==2014]["Pay (USD millions)"].max()
k=df[df["Year"]==2015]["Pay (USD millions)"].max()
l=df[df["Year"]==2016]["Pay (USD millions)"].max()
m=df[df["Year"]==2017]["Pay (USD millions)"].max()
n=df[df["Year"]==2018]["Pay (USD millions)"].max()
o=df[df["Year"]==2019]["Pay (USD millions)"].max()
#lineer regresyon
maxlar=[a,b,c,d,e,f,g,h,i,j,k,l,m,n,o]
"""
maxlar.sort()
rmse ve mse değerlerini yani tahmin hatalarını azaltmak için max değerleri yani 
x(bağımsız değişkeni)(maxlar)küçükten büyüğe sıralanabilir ancak yılların max kazançları grafiğe doğru dağıtılmaz.
örneğin 2014 yılında 620 milyon dolar maximum kazançken sıralama yapıldığı için bu değer 2019 yılının max kazancı olarak gösterilecektir.
yani hata oranları düşmüş olsa bile grafik yanlış verileri gösterecektir.Bu durum verinin yanlış yorumlanmasına neden olacaktır.
"""
yıllar=np.arange(2005,2020)
x= pd.DataFrame(yıllar)# lineer regresyon da kullanılacak değişkenler.(x,y)
y= pd.DataFrame(maxlar) 
print("x shape", x.shape)#lineer regresyon modelini çizdirebilmek için dataframeler aynı boyutta(aynı satır,sütun sayısında olmalı)
print("y shape", y.shape)#lineer regresyon modelini çizdirebilmek için satır,sütun sayısının eşitliğini kontrol etme.
plt.title("Yıllara Göre Maksimum Kazançlar ")
plt.xlabel("Yıllar") # x ekseninin etiketi
plt.ylabel("Maksimum Kazançlar") # y ekseninin etiketi
plt.grid(True) # ızgara ekleme
model = LinearRegression() # regresyon modelini ekleme
model.fit(x,y) #verileri lineer regresyon modeline göre eğitir.(x = bağımsız değişken,y = tahmin edilen değer)
y_head = model.predict(x)#tahmin edilen değerler
plt.scatter(x,y, color="r") #gerçek değerleri serpme grafiğinde gösterme(bu grafikte değerlerin doğru orantılı olarak artmadığını görmekteyiz)
plt.plot(x,y_head) #x değerlerine göre modelimizin tahmin ettiği y değerlerini çizer.
fd=model.predict(2019)#herhangi bir değer için tahmin de bulunma 
fd1=model.predict(2020) # 2020 yılında tahmin edilen max kazanç
print("2019 yılında tahmin edilen max kazanç : ",fd) # 227.79 milyon dolar ancak gerçek değer 185 milyon dolar tahmin de hata oranı(sapma) oldukça yüksek
plt.show()
print(model.coef_)#lineer regresyon model denkleminin kat sayısı
print(model.intercept_)#çizilen regresyon doğrusunun y eksenini kestiği nokta 
MSE = mean_squared_error(y_true=y, y_pred=y_head)# mse = gerçek değerler ile tahmin edilen değerler arasındaki uzaklık(hata payı)(düşük olmalı)
RMSE = sqrt(MSE) #çizdiğimiz regresyon çizgisinin gerçek tabloya göre toplam varyasyonunu gösterir. 
"""
verilere göre lineer regresyon modeli uygulandığından veriler doğru orantılı olarak artmadığı için lineer regresyon modeli
yüksek yüzdelikli bir tahmin doğruluğu ifade etmez.Bu sonucun sağlamasını mse ve rmse değerlerinin yüksek olmasından anlamaktayız.
(x değişkeninin değerlerinin sıralanmadan önceki hata değerleri)
MSE : 12684.34198412701   RMSE : 112.62478405807049
(x değişkeninin değerlerinin sıralanmadan önceki hata değerleri) 
MSE : 3693.913412698466   RMSE : 60.77757327089052
veriler sıralanarak doğru orantıya yaklaştırılmaya çalışılmıştır ve hata oranları düşürülmüştür
ancak veriyi yanlış yorumlamamak adına bu sıralamadan vazgeçilmiştir.   
"""
print("MSE : {}".format(MSE))
print("RMSE : {}".format(RMSE))#çizilen regresyon çizgisinin ne kadar iyi performans gösteriyor sorusunu yanıtlar.

