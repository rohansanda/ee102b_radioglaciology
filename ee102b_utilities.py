import numpy as np
import matplotlib.pyplot as plt
import scipy.signal


def pulse_compress_one_trace(trace_input, ref_chirp_time_domain, blanking=50, detrend_type='linear'):
    trace = np.zeros_like(trace_input)
    trace[blanking:] = trace_input[blanking:]
    detrended = scipy.signal.detrend(trace, type=detrend_type)
    dft = np.fft.fft(detrended)
    ref_chirp_freq_domain = np.fft.fft(ref_chirp_time_domain)
    product_freq_domain = np.multiply(ref_chirp_freq_domain, dft)
    return np.fft.ifft(product_freq_domain)

def pulse_compress(d):
    data_out = np.zeros((len(d["slow_time"]), len(d["fast_time"])), dtype='>i4')
    
    ref_chirp = d["reference_chirp_real"] + 1j * d["reference_chirp_imag"]

    for idx, t in enumerate(d["slow_time"]):
        pulse_compressed = pulse_compress_one_trace(d["data"][idx,:], ref_chirp_time_domain=ref_chirp)
        data_out[idx, :] = (20*np.log10(np.abs(pulse_compressed))).astype('>i4')
    
    return data_out

def plot_radargram(data_pulse_compressed, title="", bed_x=None, bed_y=None, bed=False):
    fig, ax = plt.subplots(figsize=(6,6), facecolor='white')
    im = ax.imshow(data_pulse_compressed.T, cmap='gray', vmin=100, vmax=230)
    if bed:
        ax.plot(bed_x, bed_y, 'r', linewidth=2)
    ax.set_xlabel('Traces (slow time)')
    ax.set_ylabel('Samples (fast time)')
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Power [dB]")
    return fig, ax

def find_bed(data_pulse_compressed):
    bed_range_min = 1800
    bed_range_max = 2200
    bed_max_ind = np.argmax(data_pulse_compressed[:,bed_range_min:bed_range_max], axis=1)
    bed_max_ind += bed_range_min

    # return indices of bed pick
    return bed_max_ind 

def estimate_snr(data_in, bed_inds):
    # get max signal power (bed echo power) within tol pixels of the provided bed pick
    tol = 10
    sig_pwr = np.zeros_like(bed_inds)
    for bp, st in zip(bed_inds, np.arange(0, data_in.shape[0])):
        idx_min = int(np.maximum(0, bp-tol))
        idx_max = int(np.minimum(data_in.shape[1], bp+tol+1))
        sig_pwr[st] = np.amax(data_in[st,idx_min:idx_max])
    
    # convert to linear
    sig_pwr = np.power(10, sig_pwr/10)

    # get rms noise power
    noise_ind_min = 2500
    noise_ind_max = 3200
    noise_pwr = data_in[:, noise_ind_min:noise_ind_max]
 
    # convert to linear
    noise_pwr = np.power(10, noise_pwr/10)
    # compute rms of noise power
    noise_pwr_rms = np.std(noise_pwr, axis=1)

    # compute snr
    snr = sig_pwr / noise_pwr_rms
    snr_avg = np.mean(snr)
    
    return snr, snr_avg # in linear units
    