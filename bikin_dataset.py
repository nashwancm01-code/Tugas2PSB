import math

# Kita mau bikin 1024 baris data (biar pas sama default aplikasi STFT)
N = 1024
fs = 1024 # Frekuensi sampling 1024 Hz

print("Sedang memasak data ECG palsu...")

# Buka (atau buat) file txt baru untuk ditulis
with open("data_ecg_palsu.txt", "w") as file:
    for i in range(N):
        t = i / fs
        
        # 1. Gelombang dasar (naik turun pelan seperti orang bernapas)
        napas = 0.5 * math.sin(2 * math.pi * 1 * t)
        
        # 2. Sinyal bergetar halus (frekuensi 15 Hz)
        getaran = 0.3 * math.sin(2 * math.pi * 15 * t)
        
        # 3. Bikin lonjakan tajam (seperti detak jantung / QRS complex)
        # Tiap kelipatan 250 data, kita kasih lonjakan tinggi
        if i % 250 < 15:
            detak = 3.0 * math.sin(2 * math.pi * (1 / 15) * (i % 250))
        else:
            detak = 0.0
            
        # Gabungkan semuanya jadi satu data yang utuh
        sinyal_total = napas + getaran + detak
        
        # Tulis ke dalam file txt, satu angka per baris
        file.write(f"{sinyal_total:.5f}\n")

print("Selesai! File 'data_ecg_palsu.txt' sudah berhasil dibuat di foldermu.")
print("Sekarang kamu bisa upload file ini ke aplikasi Streamlit STFT!")
