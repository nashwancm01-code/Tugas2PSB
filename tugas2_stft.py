import streamlit as st
import math
import cmath
import matplotlib.pyplot as plt

# ==========================================
# 1. FUNGSI MATEMATIKA & DSP
# ==========================================

def next_power_of_2(x):
    """Mencari nilai pangkat 2 terdekat ke atas (untuk padding FFT)."""
    return 1 if x == 0 else 2**(x - 1).bit_length()

def radix2_fft(x):
    """Algoritma Fast Fourier Transform (FFT) murni Python."""
    N = len(x)
    if N <= 1: return x
    even = radix2_fft(x[0::2])
    odd = radix2_fft(x[1::2])
    T = [cmath.exp(-2j * cmath.pi * k / N) * odd[k] for k in range(N // 2)]
    return [even[k] + T[k] for k in range(N // 2)] + [even[k] - T[k] for k in range(N // 2)]

def get_window(window_name, L):
    """Membuat bentuk 'Pisau Potong' (Window) sesuai rumus di slide PDF."""
    w = []
    for n in range(L):
        if window_name == "Rectangular":
            w.append(1.0)
        elif window_name == "Bartlett":
            w.append(1.0 - 2.0 * abs(n - (L - 1) / 2.0) / (L - 1))
        elif window_name == "Hanning":
            w.append(0.5 - 0.5 * math.cos((2 * math.pi * n) / (L - 1)))
        elif window_name == "Hamming":
            w.append(0.54 - 0.46 * math.cos((2 * math.pi * n) / (L - 1)))
        elif window_name == "Blackman":
            w.append(0.42 - 0.5 * math.cos((2 * math.pi * n) / (L - 1)) + 0.08 * math.cos((4 * math.pi * n) / (L - 1)))
    return w

def generate_sinyal_buatan(N, fs):
    """
    Membuat sinyal gabungan seperti di aplikasi dosen:
    Sinyal berubah-ubah: 50Hz -> 150Hz -> 250Hz -> 350Hz.
    """
    x = []
    for i in range(N):
        t = i / fs
        # Sinyal berubah frekuensi tiap 1/4 bagian waktu
        if i < N // 4:
            f = 50
        elif i < 2 * N // 4:
            f = 150
        elif i < 3 * N // 4:
            f = 250
        else:
            f = 350
        x.append(math.sin(2 * math.pi * f * t))
    return x

# ==========================================
# 2. KONFIGURASI TAMPILAN STREAMLIT
# ==========================================
st.set_page_config(page_title="STFT Analysis", layout="wide")
st.title("Time-Frequency Analysis - STFT")

# --- SIDEBAR (PANEL KONTROL KIRI) ---
with st.sidebar:
    st.header("Panel Kontrol")
    
    st.subheader("Data")
    jenis_sinyal = st.radio("Pilih Sinyal:", ["Sinyal Buatan", "Sinyal ECG (Upload)"])
    
    if jenis_sinyal == "Sinyal ECG (Upload)":
        uploaded_file = st.file_uploader("Upload file .txt", type=["txt"])
    
    N_data = st.number_input("Jumlah Data (N):", value=1024, step=128)
    fs = st.number_input("Frek. Sampling (fs):", value=1024, step=128)
    
    st.markdown("---")
    st.subheader("Windowing")
    window_type = st.radio("Pilih Jenis Window:", ["Rectangular", "Bartlett", "Hanning", "Hamming", "Blackman"], index=3)
    
    lebar_window = st.number_input("Lebar Window (Panjang Potongan):", value=100, step=10)
    irisan = st.number_input("Irisan (Overlap):", value=50, step=10)

# ==========================================
# 3. PROSES UTAMA (LOGIKA STFT)
# ==========================================
# Siapkan Data Input
x_input = []
if jenis_sinyal == "Sinyal Buatan":
    x_input = generate_sinyal_buatan(N_data, fs)
else:
    if uploaded_file is not None:
        raw_text = uploaded_file.getvalue().decode('utf-8').splitlines()
        x_input = [float(line.strip()) for line in raw_text if line.strip()]
        N_data = len(x_input)
    else:
        st.warning("Silakan upload file ECG terlebih dahulu.")
        st.stop()

# Bikin Sumbu Waktu (Time array)
waktu = [i / fs for i in range(N_data)]

# Bikin Window (Pisau Potong)
w = get_window(window_type, lebar_window)
hop_size = lebar_window - irisan
if hop_size <= 0: hop_size = 1 # Mencegah error kalau irisan kepanjangan

# --- HITUNG FFT FULL (Buat Grafik Kanan Atas) ---
N_fft_full = next_power_of_2(N_data)
x_padded_full = x_input + [0.0] * (N_fft_full - N_data)
fft_full = radix2_fft(x_padded_full)
mag_full = [abs(c) / N_fft_full for c in fft_full[:N_fft_full // 2]]
freq_full = [k * fs / N_fft_full for k in range(N_fft_full // 2)]

# --- HITUNG STFT (Proses Pemotongan & FFT Berulang) ---
stft_matrix = []
time_bins = []
nfft_stft = next_power_of_2(lebar_window)

# Menyimpan 3 window pertama untuk digambar (seperti di Delphi)
contoh_window_plot = [] 

for start_idx in range(0, N_data - lebar_window + 1, hop_size):
    # 1. Potong sinyal (Truncate)
    segment = x_input[start_idx : start_idx + lebar_window]
    
    # 2. Kalikan dengan Window (w(t))
    windowed_segment = [segment[i] * w[i] for i in range(lebar_window)]
    
    # Simpan sampel potongan untuk grafik (Maksimal 3 potong pertama)
    if len(contoh_window_plot) < 3:
        contoh_window_plot.append((start_idx, windowed_segment))
    
    # 3. Padding nol biar panjangnya kelipatan 2 (syarat Radix-2 FFT)
    padded_segment = windowed_segment + [0.0] * (nfft_stft - lebar_window)
    
    # 4. Hitung FFT dari potongan tersebut
    X_k = radix2_fft(padded_segment)
    
    # Ambil setengah magnitudonya (karena simetris)
    mag_stft = [abs(c) / nfft_stft for c in X_k[:nfft_stft // 2]]
    
    # 5. Simpan ke dalam Matriks Spectrogram
    stft_matrix.append(mag_stft)
    
    # Titik tengah dari potongan window sebagai penanda waktu
    waktu_tengah = (start_idx + lebar_window / 2) / fs
    time_bins.append(waktu_tengah)

# Bikin Sumbu Frekuensi untuk STFT
freq_bins = [k * fs / nfft_stft for k in range(nfft_stft // 2)]

# Transpose matriks STFT (Baris = Frekuensi, Kolom = Waktu) agar bisa di-plot Matplotlib
stft_transposed = [[stft_matrix[col][row] for col in range(len(stft_matrix))] for row in range(len(stft_matrix[0]))]

# ==========================================
# 4. PLOTTING GRAFIK (MENIRU DELPHI)
# ==========================================
st.markdown("### Hasil Analisis")

col1, col2 = st.columns([2, 1])

with col1:
    # Grafik 1: Sinyal Input Utuh
    fig1, ax1 = plt.subplots(figsize=(10, 2))
    ax1.plot(waktu, x_input, color='red', linewidth=0.8)
    ax1.set_title("Sinyal Input", fontsize=10)
    ax1.set_xlabel("Waktu (s)", fontsize=8)
    ax1.set_ylabel("Amplitude", fontsize=8)
    ax1.grid(True, linestyle=':')
    st.pyplot(fig1)
    
    # Grafik 2: Visualisasi Sinyal Hasil Windowing (Tumpang tindih potongan)
    fig2, ax2 = plt.subplots(figsize=(10, 2))
    ax2.plot(waktu, x_input, color='lightgray', linewidth=0.5, label="Sinyal Asli") # Background sinyal asli
    
    warna = ['red', 'blue', 'black']
    for idx, (start_idx, windowed_segment) in enumerate(contoh_window_plot):
        waktu_segmen = [ (start_idx + i) / fs for i in range(lebar_window) ]
        ax2.plot(waktu_segmen, windowed_segment, color=warna[idx], linewidth=1.2, label=f"Window {idx+1}")
        
    ax2.set_title(f"Sinyal Hasil Windowing (Mendemonstrasikan pemotongan)", fontsize=10)
    ax2.set_xlabel("Waktu (s)", fontsize=8)
    ax2.set_ylabel("Amplitude", fontsize=8)
    ax2.legend(fontsize=7)
    ax2.grid(True, linestyle=':')
    st.pyplot(fig2)

with col2:
    # Grafik 3: Spectrum FFT dari Sinyal Input Utuh
    fig3, ax3 = plt.subplots(figsize=(5, 4.5))
    ax3.plot(freq_full, mag_full, color='red', linewidth=1)
    ax3.set_title("Spectrum Sinyal Input (FFT Total)", fontsize=10)
    ax3.set_xlabel("Frekuensi (Hz)", fontsize=8)
    ax3.set_ylabel("Magnitude", fontsize=8)
    ax3.set_xlim(0, fs / 2)
    ax3.grid(True, linestyle=':')
    st.pyplot(fig3)

# Grafik 4: Spectrogram (Hasil Akhir STFT 2D)
st.markdown("### Spectrogram STFT 2D (Frekuensi vs Waktu)")
fig4, ax4 = plt.subplots(figsize=(12, 4))
c = ax4.pcolormesh(time_bins, freq_bins, stft_transposed, shading='gouraud', cmap='jet')
fig4.colorbar(c, ax=ax4, label="Kekuatan Sinyal (Magnitude)")
ax4.set_ylabel("Frekuensi (Hz)")
ax4.set_xlabel("Waktu (s)")
ax4.set_ylim(0, fs / 2) # Tampilkan hanya sampai batas Nyquist
st.pyplot(fig4)
