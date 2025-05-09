import numpy as np
from scipy import signal
from scipy.integrate import simpson

def extract_de(epochs, channels = 62,
               num_band = 5, bands = [1, 4, 8, 13, 31, 50], fq = 200,
               overlap = 0, NFFT = 256, num_time=0):
    # time_len= epochs.shape[-1] // fq
    file_de = np.zeros([channels, num_time, num_band])
    de = []
    for i in range(channels):
        session_spec, f_, _, _ = plt.specgram(epochs[i, :],
                                            Fs=fq,
                                            noverlap = overlap,
                                            NFFT = NFFT)
        # session_spec = session_spec.swapaxes(0, 1) # t X fq
        bins = np.digitize(f_, bands)

        for j in range(num_band):
            file_de[i, :, j] = ((session_spec[bins == j+1] - session_spec[bins == j+1].mean(axis=0))**2).mean(axis=0)

    file_de_tmp = 0.5 * np.log(file_de) + 0.5 * np.log(2 * np.pi * np.e)
    de = np.array(file_de_tmp)
    return de
                   
def compute_psd(eeg_signal,
                       fs = 200, sec = 1, nfft = 256, noverlap = 128):
    """
    Вычисление мощности ЭЭГ.

    Параметры:
    ----------
    eeg_signal : array-like
        Одноканальный сигнал ЭЭГ.
    fs : float
        Частота дискретизации сигнала (Гц).

    Возвращает:
    -----------
    freqs : 1d list
    psd: 2d list
    """
    # Вычисление периодограммы методом Уэлча
    freqs, psd = signal.welch(eeg_signal, fs=fs, nfft=nfft, noverlap = fs*sec//2, nperseg=fs*sec)

    return (freqs, psd)


def compute_band_power(eeg_signal,
                       bands = {
                           'delta': (1, 4),
                           'theta': (4, 8),
                           'alpha': (8, 13),
                           'beta': (13, 30),
                           'gamma': (30, 51)
                       },
                       fs = 200, sec = 1, nfft = 256, noverlap = 128):
    """
    Вычисление мощности в частотных диапазонах ЭЭГ.

    Параметры:
    ----------
    eeg_signal : array-like
        Одноканальный сигнал ЭЭГ.
    fs : float
        Частота дискретизации сигнала (Гц).
    bands : dict
        Словарь с частотными диапазонами, например:
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 45)
        }

    Возвращает:
    -----------
    band_powers : dict
        Словарь с мощностями в заданных диапазонах.
    """
    # Вычисление периодограммы методом Уэлча
    freqs, psd = signal.welch(eeg_signal, fs=fs, nfft=nfft, noverlap = nfft//2, nperseg=fs*sec)

    band_powers = {}
    for band, (f_low, f_high) in bands.items():
        # Находим индексы частот в нужном диапазоне
        idx = np.logical_and(freqs >= f_low, freqs < f_high)

        # Интегрируем PSD в диапазоне (метод Симпсона)
        band_power = simpson(psd[idx], x = freqs[idx])

        # Сохраняем результат
        band_powers[band] = band_power

    # Нормировка на общую мощность (опционально)
    total_power = sum(band_powers.values())
    # band_powers = {band: power / total_power for band, power in band_powers.items()}
    band_powers = [power / total_power for band, power in band_powers.items()]

    return band_powers
