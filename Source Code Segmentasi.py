import cv2
import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# Fungsi Untuk Menormalkan Magnitude ke uint8 (0–255)
# =====================================================
def to_uint8(mag):
    mag = np.nan_to_num(mag)             # ganti NaN jadi 0
    mag = np.abs(mag)                    # semua positif
    mag = (mag / mag.max()) * 255        # normalisasi
    return mag.astype(np.uint8)


# =====================================================
# METODE ROBERTS
# =====================================================
def roberts(img):
    img = img.astype(np.float32)

    kx = np.array([[1, 0],
                   [0, -1]], np.float32)
    ky = np.array([[0, 1],
                   [-1, 0]], np.float32)

    gx = cv2.filter2D(img, cv2.CV_32F, kx)
    gy = cv2.filter2D(img, cv2.CV_32F, ky)

    mag = np.sqrt(gx**2 + gy**2)
    return to_uint8(mag)


# =====================================================
# METODE PREWITT
# =====================================================
def prewitt(img):
    img = img.astype(np.float32)

    kx = np.array([[-1, 0, 1],
                   [-1, 0, 1],
                   [-1, 0, 1]], np.float32)

    ky = np.array([[-1, -1, -1],
                   [0,   0,  0],
                   [1,   1,  1]], np.float32)

    gx = cv2.filter2D(img, cv2.CV_32F, kx)
    gy = cv2.filter2D(img, cv2.CV_32F, ky)

    mag = np.sqrt(gx**2 + gy**2)
    return to_uint8(mag)


# =====================================================
# METODE SOBEL
# =====================================================
def sobel(img):
    img = img.astype(np.float32)

    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)

    mag = np.sqrt(gx**2 + gy**2)
    return to_uint8(mag)


# =====================================================
# METODE FREI-CHEN
# =====================================================
def frei_chen(img):
    img = img.astype(np.float32)

    a = np.sqrt(2)

    kx = np.array([[-1, 0, 1],
                   [-a, 0, a],
                   [-1, 0, 1]], np.float32)

    ky = np.array([[-1, -a, -1],
                   [0,   0,  0],
                   [1,   a,  1]], np.float32)

    gx = cv2.filter2D(img, cv2.CV_32F, kx)
    gy = cv2.filter2D(img, cv2.CV_32F, ky)

    mag = np.sqrt(gx**2 + gy**2)
    return to_uint8(mag)


# =====================================================
# LOAD GAMBAR
# =====================================================
g1 = cv2.imread(r"d:\KULIAH\SEMESTER 3\PRAKTIKUM SD\JUDUL 3\potraitt.jpg", 0)
g2 = cv2.imread(r"d:\KULIAH\SEMESTER 3\PRAKTIKUM SD\JUDUL 3\landscapee.jpg", 0)

if g1 is None or g2 is None:
    raise Exception("Gambar tidak ditemukan! Periksa path file dengan benar.")

# =====================================================
# PROSES SEMUA METODE DAN GABUNGKAN
# =====================================================
hasil = {
    "G1 Roberts": roberts(g1),
    "G1 Prewitt": prewitt(g1),
    "G1 Sobel": sobel(g1),
    "G1 Frei-Chen": frei_chen(g1),

    "G2 Roberts": roberts(g2),
    "G2 Prewitt": prewitt(g2),
    "G2 Sobel": sobel(g2),
    "G2 Frei-Chen": frei_chen(g2)
}

# =====================================================
# TAMPILKAN
# =====================================================
plt.figure(figsize=(14, 10))
plt.suptitle("Segmentasi Discontinuity: Roberts, Prewitt, Sobel, Frei-Chen", fontsize=18)

i = 1
for judul, img in hasil.items():
    plt.subplot(4, 2, i)
    plt.imshow(img, cmap='gray')
    plt.title(judul)
    plt.axis("off")
    i += 1

plt.tight_layout()
plt.show()