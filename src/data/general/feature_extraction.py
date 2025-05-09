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

def extract_dasm(data, channels):
    channels_left = np.array([
        'FP1', 'F7', 'F3', 'T7', 'P7', 'C3',
        'P3', 'O1', 'AF3', 'FC5', 'FC1', 'CP5',
        'CP1', 'PO3'
    ])
    channels_right = np.array([
        'FP2', 'F8', 'F4', 'T8', 'P8', 'C4',
        'P4', 'O2', 'AF4', 'FC6', 'FC2', 'CP6',
        'CP2', 'PO4'
    ])
    mask_left = np.array([np.where(idx == channels)[0][0] for idx in channels_left])
    de_left = data[:,:,mask_left,:]
    mask_right = np.array([np.where(idx == channels)[0][0] for idx in channels_right])
    de_right = data[:,:,mask_right,:]
    dasm = de_left - de_right
    return dasm

def extract_rasm(data, channels):
    channels_left = np.array([
        'FP1', 'F7', 'F3', 'T7', 'P7', 'C3',
        'P3', 'O1', 'AF3', 'FC5', 'FC1', 'CP5',
        'CP1', 'PO3'
    ])
    channels_right = np.array([
        'FP2', 'F8', 'F4', 'T8', 'P8', 'C4',
        'P4', 'O2', 'AF4', 'FC6', 'FC2', 'CP6',
        'CP2', 'PO4'
    ])
    mask_left = np.array([np.where(idx == channels)[0][0] for idx in channels_left])
    de_left = data[:,:,mask_left,:]
    mask_right = np.array([np.where(idx == channels)[0][0] for idx in channels_right])
    de_right = data[:,:,mask_right,:]
    rasm = de_left/de_right
    return rasm

def extract_dcau(data, channels):
    channels_frontal = np.array([
        'FC5', 'FC1', 'FC2', 'FC6', 'F7', 'F3',
        'FZ', 'F4', 'F8', 'FP1', 'FP2'
    ])
    channels_posterior = np.array([
        'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3',
        'PZ', 'P4', 'P8', 'O1', 'O2'
    ])
    mask_frontal = np.array([np.where(idx == channels)[0][0] for idx in channels_frontal])
    de_frontal = data[:,:,mask_frontal,:]
    mask_posterior = np.array([np.where(idx == channels)[0][0] for idx in channels_posterior])
    de_posterior = data[:,:,mask_posterior,:]
    dcau = de_frontal - de_posterior
    return dcau

def reshape_3d(data, channels, num_channels):
    matrix_62 = np.array([
        np.array([0, 0, 0, "FP1", "FPZ", "FP2", 0, 0, 0]),
        np.array([0, 0, 0, "AF3", 0, "AF4", 0, 0, 0]),
        np.array(["F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8"]),
        np.array(["FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8"]),
        np.array(["T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8"]),
        np.array(["TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8"]),
        np.array(["P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8"]),
        np.array([0, "PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8", 0]),
        np.array([0, 0, "CB1", "O1", "OZ", "O2", "CB2", 0, 0])
    ])
    matrix_32 = np.array([
        np.array([0, 0, 0, "FP1", 0, "FP2", 0, 0, 0]),
        np.array([0, 0, 0, "AF3", 0, "AF4", 0, 0, 0]),
        np.array(["F7", 0, "F3", 0, "FZ", 0, "F4", 0, "F8"]),
        np.array([0, "FC5", 0, "FC1", 0, "FC2", 0, "FC6", 0]),
        np.array(["T7", 0, "C3", 0, "CZ", 0, "C4", 0, "T8"]),
        np.array([0, "CP5", 0, "CP1", 0, "CP2", 0, "CP6", 0]),
        np.array(["P7", 0, "P3", 0, "PZ", 0, "P4", 0, "P8"]),
        np.array([0, 0, 0, "PO3", 0, "PO4", 0, 0, 0]),
        np.array([0, 0, 0, "O1", "OZ", "O2", 0, 0, 0])
    ])
    if num_channels == 62:
        matrix = matrix_62
    else:
        matrix = matrix_32
        
    data = data[:, :, np.sum((channels.reshape(-1, 1) == matrix.reshape(-1)).astype(int)*np.array(range(1,len(channels)+1)).reshape(-1, 1), axis=0) - 1, :]
    data[:, :, matrix.reshape(-1) == '0'] = 0

    data = np.moveaxis(data, 3, 1).reshape((data.shape[0], data.shape[-1], 9, 9))
    in_channels = 128
    return data, in_channels, matrix
