import numpy as np

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
                   
