import streamlit as st
import math
import cmath
import matplotlib.pyplot as plt
import plotly.graph_objects as go 

# ==========================================
# 1. FUNGSI MATEMATIKA & DSP MURNI
# ==========================================
def next_power_of_2(x):
    return 1 if x == 0 else 2**(x - 1).bit_length()

def radix2_fft(x):
    N = len(x)
    if N <= 1: return x
    even = radix2_fft(x[0::2])
    odd = radix2_fft(x[1::2])
    T = [cmath.exp(-2j * cmath.pi * k / N) * odd[k] for k in range(N // 2)]
    return [even[k] + T[k] for k in range(N // 2)] + [even[k] - T[k] for k in range(N // 2)]

def get_window(window_name, L):
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
    x = []
    for i in range(N):
        t = i / fs
        if i < N // 4: f = 50
        elif i < 2 * N // 4: f = 150
        elif i < 3 * N // 4: f = 250
        else: f = 350
        x.append(math.sin(2 * math.pi * f * t))
    return x

# ==========================================
# 2. INISIALISASI MEMORI (SESSION STATE)
# ==========================================
if 'data_mentah' not in st.session_state: st.session_state.data_mentah = []
if 'data_siap' not in st.session_state: st.session_state.data_siap = []
if 'hasil_stft' not in st.session_state: st.session_state.hasil_stft = None
if 'fs' not in st.session_state: st.session_state.fs = 1024

# ==========================================
# 3. KONFIGURASI TAMPILAN STREAMLIT
# ==========================================
st.set_page_config(page_title="Time-Frequency Analysis", layout="wide")
st.title("Time-Frequency Analysis - STFT")

# --- SIDEBAR (PANEL KONTROL) ---
with st.sidebar:
    st.markdown("### Data")
    jenis_sinyal = st.radio("Pilih Sinyal:", ["Sinyal Buatan", "Sinyal ECG"])
    
    uploaded_file = None
    if jenis_sinyal == "Sinyal ECG":
        uploaded_file = st.file_uploader("Upload file .txt", type=["txt"])
        
    N_data = st.number_input("Jumlah Data:", value=1024, step=128)
    fs_input = st.number_input("Frek. Sampling:", value=1024, step=128)
    
    if st.button("Proses Data"):
        st.session_state.fs = fs_input
        if jenis_sinyal == "Sinyal Buatan":
            st.session_state.data_mentah = generate_sinyal_buatan(N_data, fs_input)
            st.session_state.data_siap = st.session_state.data_mentah.copy()
            st.session_state.hasil_stft = None 
        else:
            if uploaded_file is not None:
                raw_text = uploaded_file.getvalue().decode('utf-8').splitlines()
                st.session_state.data_mentah = [float(line.strip()) for line in raw_text if line.strip()]
                st.session_state.data_siap = st.session_state.data_mentah.copy()
                st.session_state.hasil_stft = None
            else:
                st.error("Upload file dulu ya!")

    st.markdown("---")
    
    st.markdown("### Set Panjang Data")
    tipe_panjang = st.radio("Pilih Mode:", ["Full Data", "Ambil Data ke :"])
    
    c1, c2 = st.columns(2)
    with c1: start_idx = st.number_input("Mulai", value=280, min_value=0)
    with c2: end_idx = st.number_input("Sampai", value=579, min_value=1)
    
    if st.button("Proses Potong Data"):
        if len(st.session_state.data_mentah) > 0:
            if tipe_panjang == "Full Data":
                st.session_state.data_siap = st.session_state.data_mentah.copy()
            else:
                st.session_state.data_siap = st.session_state.data_mentah[start_idx:end_idx+1]
            st.session_state.hasil_stft = None
        else:
            st.warning("Proses Data Utama dulu di atas!")

    st.markdown("---")
    
    st.markdown("### Windowing")
    window_type = st.radio("Pilih Window:", ["Rectangular", "Bartlett", "Hanning", "Hamming", "Blackman"], index=3)
    
    irisan = st.number_input("Irisan (Overlap):", value=0, min_value=0)
    lebar_window = st.number_input("Lebar Window:", value=100, min_value=2)
    
    hop_size = lebar_window - irisan
    if hop_size <= 0: hop_size = 1
    
    # Hitung estimasi maksimal window yang muat
    jml_data_aktif = len(st.session_state.data_siap)
    max_window = 1
    if jml_data_aktif > 0:
        max_window = max(1, (jml_data_aktif - lebar_window) // hop_size + 1)
    
    # SEKARANG JUMLAH WINDOW JADI INPUT MANUAL
    jumlah_window_input = st.number_input("Jumlah Window:", value=max_window, min_value=1)
    
    if st.button("STFT", type="primary"):
        if jml_data_aktif > 0:
            x_input = st.session_state.data_siap
            fs = st.session_state.fs
            w = get_window(window_type, lebar_window)
            nfft_stft = next_power_of_2(lebar_window)
            
            stft_data = [] 
            stft_matrix = []
            time_bins = []
            
            # SEKARANG LOOPING BERDASARKAN INPUT JUMLAH WINDOW DARI USER
            for idx in range(jumlah_window_input):
                start = idx * hop_size
                
                # Ambil potongan data (kalau kurang dari lebar_window, ujungnya ditambah 0)
                segment = x_input[start : start + lebar_window]
                if len(segment) < lebar_window:
                    segment = segment + [0.0] * (lebar_window - len(segment))
                
                windowed = [segment[i] * w[i] for i in range(lebar_window)]
                padded = windowed + [0.0] * (nfft_stft - lebar_window)
                
                X_k = radix2_fft(padded)
                mag = [abs(c) / nfft_stft for c in X_k[:nfft_stft // 2]]
                
                stft_data.append({
                    'start_idx': start,
                    'end_idx': start + lebar_window - 1,
                    'windowed_signal': windowed,
                    'spectrum': mag
                })
                
                stft_matrix.append(mag)
                time_bins.append((start + lebar_window / 2) / fs)
            
            freq_bins = [k * fs / nfft_stft for k in range(nfft_stft // 2)]
            stft_transposed = [[stft_matrix[col][row] for col in range(len(stft_matrix))] for row in range(len(stft_matrix[0]))]
            
            st.session_state.hasil_stft = {
                'stft_data': stft_data,
                'time_bins': time_bins,
                'freq_bins': freq_bins,
                'matrix_2d': stft_transposed,
                'lebar_window': lebar_window,
                'nfft': nfft_stft
            }
        else:
            st.warning("Data kosong, jalankan Proses Data dulu!")

# ==========================================
# 4. PLOTTING AREA (BAGIAN KANAN)
# ==========================================
if len(st.session_state.data_siap) > 0:
    x_plot = st.session_state.data_siap
    fs_plot = st.session_state.fs
    N_plot = len(x_plot)
    waktu_plot = [i / fs_plot for i in range(N_plot)]
    
    N_fft_full = next_power_of_2(N_plot)
    x_padded_full = x_plot + [0.0] * (N_fft_full - N_plot)
    fft_full = radix2_fft(x_padded_full)
    mag_full = [abs(c) / N_fft_full for c in fft_full[:N_fft_full // 2]]
    freq_full = [k * fs_plot / N_fft_full for k in range(N_fft_full // 2)]

    col1, col2 = st.columns([2, 1])
    with col1:
        fig1, ax1 = plt.subplots(figsize=(10, 2.5))
        ax1.plot(range(N_plot), x_plot, color='red', linewidth=0.8)
        ax1.set_title("Sinyal Input", fontsize=10)
        ax1.set_xlabel("n sample", fontsize=8)
        ax1.set_ylabel("Amplitude", fontsize=8)
        ax1.grid(True, linestyle=':')
        st.pyplot(fig1)
        
    with col2:
        fig2, ax2 = plt.subplots(figsize=(5, 2.5))
        ax2.plot(freq_full, mag_full, color='red', linewidth=1)
        ax2.set_title("Spectrum Sinyal Input", fontsize=10)
        ax2.set_xlabel("Frekuensi (Hz)", fontsize=8)
        ax2.set_ylabel("Magnitude", fontsize=8)
        ax2.set_xlim(0, fs_plot / 2)
        ax2.grid(True, linestyle=':')
        st.pyplot(fig2)

if st.session_state.hasil_stft is not None:
    res = st.session_state.hasil_stft
    stft_data = res['stft_data']
    
    st.markdown("---")
    
    jml_w = len(stft_data)
    if jml_w > 0:
        w_pilihan = st.slider(f"Pilih Window (w = 0 sampai {jml_w - 1})", 0, jml_w - 1, 0)
        
        data_terpilih = stft_data[w_pilihan]
        waktu_window = range(data_terpilih['start_idx'], data_terpilih['end_idx'] + 1)
        
        col3, col4 = st.columns([2, 1])
        with col3:
            fig3, ax3 = plt.subplots(figsize=(10, 2.5))
            ax3.plot(range(N_plot), x_plot, color='lightgray', linewidth=0.5)
            
            if w_pilihan == 0 and jml_w >= 3:
                w0 = stft_data[0]
                w1 = stft_data[1]
                w2 = stft_data[2]
                
                ax3.plot(range(w0['start_idx'], w0['end_idx'] + 1), w0['windowed_signal'], color='red', linewidth=1.2, label='w=0')
                ax3.plot(range(w1['start_idx'], w1['end_idx'] + 1), w1['windowed_signal'], color='blue', linewidth=1.2, label='w=1')
                ax3.plot(range(w2['start_idx'], w2['end_idx'] + 1), w2['windowed_signal'], color='black', linewidth=1.2, label='w=2')
                
                ax3.set_title(f"Sinyal Hasil Windowing (Overlap w=0, 1, 2)", fontsize=10)
                ax3.legend(fontsize=7, loc='upper right')
            else:
                ax3.plot(waktu_window, data_terpilih['windowed_signal'], color='black', linewidth=1.2)
                ax3.set_title(f"Sinyal Hasil Windowing (w = {w_pilihan}, Data = {data_terpilih['start_idx']}->{data_terpilih['end_idx']})", fontsize=10)
            
            ax3.set_xlabel("n sample", fontsize=8)
            ax3.set_ylabel("Amplitude", fontsize=8)
            ax3.grid(True, linestyle=':')
            st.pyplot(fig3)
            
        with col4:
            fig4, ax4 = plt.subplots(figsize=(5, 2.5))
            freq_bins_w = res['freq_bins']
            ax4.plot(freq_bins_w, data_terpilih['spectrum'], color='red', linewidth=1)
            ax4.set_title("Spectrum Sinyal Hasil Windowing", fontsize=10)
            ax4.set_xlabel("Frekuensi (Hz)", fontsize=8)
            ax4.set_ylabel("Magnitude", fontsize=8)
            ax4.set_xlim(0, fs_plot / 2)
            ax4.grid(True, linestyle=':')
            st.pyplot(fig4)

    col5, col6 = st.columns([1, 1])
    with col5:
        st.markdown("##### 2D Spectrogram")
        fig5, ax5 = plt.subplots(figsize=(6, 4))
        c = ax5.pcolormesh(res['time_bins'], res['freq_bins'], res['matrix_2d'], shading='gouraud', cmap='jet')
        fig5.colorbar(c, ax=ax5)
        ax5.set_ylabel("Frekuensi (Hz)", fontsize=8)
        ax5.set_xlabel("Waktu (s)", fontsize=8)
        ax5.set_ylim(0, fs_plot / 2)
        st.pyplot(fig5)
        
    with col6:
        st.markdown("##### 3D Spectrogram")
        fig6 = go.Figure(data=[go.Surface(
            z=res['matrix_2d'], 
            x=res['time_bins'], 
            y=res['freq_bins'], 
            colorscale='Jet'
        )])
        fig6.update_layout(
            margin=dict(l=0, r=0, b=0, t=0),
            scene=dict(
                xaxis_title='Waktu (s)',
                yaxis_title='Frekuensi (Hz)',
                zaxis_title='Magnitude'
            ),
            height=400
        )
        st.plotly_chart(fig6, use_container_width=True)
