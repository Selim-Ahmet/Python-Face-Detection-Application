import cv2
import os

# Haar cascade dosyasının yolunu kontrol et
haar_cascade_path = "haarcascade_frontalface_default.xml"
if not os.path.isfile(haar_cascade_path):
    print(f"Error: Haar cascade file not found at {haar_cascade_path}")
    exit()

# Yüz tespit sınıflandırıcısını yükle
face_cascade = cv2.CascadeClassifier(haar_cascade_path)

# Resmi oku
img = cv2.imread("test.jpg")

# Resmin düzgün yüklendiğini kontrol et
if img is None:
    print("Error: Could not read the image.")
    exit()

# Ekran boyutunu al
screen_width = 1920  # Örnek olarak 1920
screen_height = 1080  # Örnek olarak 1080

# Resmi ekran boyutuna göre yeniden boyutlandır
img_resized = cv2.resize(img, (screen_width, screen_height))

# Resmi gri tonlamaya çevir
gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

# Yüzleri tespit et
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

# Tespit edilen yüzlerin etrafına dikdörtgen çiz
for (x, y, w, h) in faces:
    cv2.rectangle(img_resized, (x, y), (x + w, y + h), (255, 0, 0), 2)

# Resmi göster
cv2.imshow("img", img_resized)
cv2.waitKey(0)  # Bir tuşa basılmasını bekle
cv2.imwrite("face_detected.jpg", img_resized)  # İşlenmiş resmi kaydet
cv2.destroyAllWindows()