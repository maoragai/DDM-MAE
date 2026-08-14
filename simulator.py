import numpy as np
import plotly.graph_objects as go
import plotly.express as px 
from scipy.constants import speed_of_light
from typing import Optional


def plot_complex_signal(signal_c):
    fig= go.Figure()
    fig.add_trace(go.Scatter(y=signal_c.real,name='Real'))
    fig.add_trace(go.Scatter(y=signal_c.imag,name='Imaginary'))
    return fig

def target_to_indices(target_range:float, 
                      target_velocity:float, 
                      range_gate_res:float,
                      pri:float,
                      fft_len:int,
                      doppler_bin_res:Optional[float]):
    range_index = np.floor(target_range/range_gate_res)
    f_d = 2*target_velocity/speed_of_light*3e9
    doppler_index = np.floor(f_d/doppler_bin_res) if doppler_bin_res is not None else np.floor(f_d/(1/(pri*fft_len)))
    return int(range_index), int(doppler_index)

# constants
c = speed_of_light  # Speed of light (m/s)

# radar parameters
antenna_gain = 1
f_c = 3e9  # Carrier frequency (Hz)
pulse_power = 1.0  # Transmit power (W)
pulse_amplitude = np.sqrt(pulse_power)  # Pulse amplitude (V)   
wavelength = c/f_c
noise_var = 0.1
f_sample = 40e6 
dt = 1/f_sample
  

# dwell parameters
# dwell

num_pulses = 256             # 10 [us]
pri = 10e-6
samples_in_pri = pri/dt
cpi_duration =  pri * num_pulses
prf = 1/pri

# pulse_params
lfm_bw = 10e6           # [Hz]
duty_cycle = 5          # [%]
pulse_duration = pri*duty_cycle/100
lfm_slope = lfm_bw/pulse_duration
samples_in_pulse = pulse_duration/dt


fft_len = 2**8
range_gate_res = c/(2*lfm_bw)
doppler_res = 1/cpi_duration
doppler_bin_res = prf/fft_len

# snr calculation
snr_gain_autocorr = 10*np.log10(samples_in_pulse)
snr_gain_coher_integration = 10*np.log10(num_pulses)
snr_orig_db = 10*np.log10(pulse_power/noise_var)

# target params
target_snr = 5 #10  # [dB]
target_rcs = 1          # [m^2]
target_velocity = 1000   # [m/s]

target_range    = 1000  # [m] 
target_delay = 2*target_range/c  # [s]
f_d = 2*target_velocity/wavelength
target_theta = 2*np.pi*f_d*pri 
target_power = 10**(target_snr/10)*noise_var if target_snr is not None and target_snr > 0 else\
                pulse_power*(antenna_gain**2)*(wavelength**2)*target_rcs/(((4*np.pi)**3)*(target_range**4)) # radar equation

target_amplitude = np.sqrt(target_power)  # [V]


# pulse generation
pulse_lin_t = np.linspace(0,pulse_duration,int(samples_in_pulse),endpoint=False)
pulse_samples = pulse_amplitude*np.exp(1j*2*np.pi*lfm_slope*pulse_lin_t**2/2)

dwell_lin_t = np.linspace(0,cpi_duration,int(cpi_duration/dt),endpoint=False)
noise_samples = np.random.normal(0,np.sqrt(noise_var/2),len(dwell_lin_t)) + 1j*np.random.normal(0,np.sqrt(noise_var/2),len(dwell_lin_t))
dwell_samples = np.zeros(len(dwell_lin_t),dtype=complex)
dwell_samples += noise_samples
dwell_mask = np.zeros(len(dwell_lin_t))
target_theta_offset = np.random.uniform(0,2*np.pi)
matched_filter = np.conj(pulse_samples[::-1])
num_samples_pulse = int(round(samples_in_pulse))
num_samples_pri = int(round(samples_in_pri))

for pulse_idx in range(num_pulses):

    pulse_start_idx = pulse_idx * num_samples_pri

    # ------------------------------------------------------------
    # Transmitted pulse
    # ------------------------------------------------------------
    dwell_samples[
        pulse_start_idx:pulse_start_idx + num_samples_pulse
    ] += pulse_samples

    dwell_mask[
        pulse_start_idx:pulse_start_idx + num_samples_pulse
    ] += 1

    # ------------------------------------------------------------
    # Target echo
    # ------------------------------------------------------------

    # Time within the current PRI
    t = np.arange(num_samples_pri) * dt

    # Time relative to the transmitted pulse after propagation delay
    t_delayed = t - target_delay

    # Target exists only during the delayed pulse
    target_valid = (
        (t_delayed >= 0) &
        (t_delayed < pulse_duration)
    )

    # Create delayed target waveform
    target_echo = np.zeros(num_samples_pri, dtype=complex)

    target_echo[target_valid] = (
        pulse_amplitude
        * np.exp(
            1j * np.pi
            * lfm_slope
            * t_delayed[target_valid] ** 2
        )
    )

    # ------------------------------------------------------------
    # Target phase
    # ------------------------------------------------------------

    # Propagation phase
    propagation_phase = np.exp(
        -1j * 2 * np.pi * f_c * target_delay
    )

    # Doppler phase
    doppler_phase = np.exp(
        1j * 2 * np.pi * f_d * (pulse_idx * pri + t)
    )

    # Initial random phase
    initial_phase = np.exp(
        1j * target_theta_offset
    )

    # Apply target amplitude and phase
    target_echo *= (
        target_amplitude
        * propagation_phase
        * doppler_phase
        * initial_phase
    )

    # ------------------------------------------------------------
    # Inject target
    # ------------------------------------------------------------

    dwell_samples[
        pulse_start_idx:pulse_start_idx + num_samples_pri
    ] += target_echo

    dwell_mask[
        pulse_start_idx:pulse_start_idx + num_samples_pri
    ] += target_valid.astype(float)

compressed_full = (
    np.convolve(
        dwell_samples,
        matched_filter,
        mode='full'
    )
    # .reshape((num_pulses, int(samples_in_pri)))
)
# Matched filter delay
mf_delay = len(matched_filter) - 1

# Align the matched-filter output so that:
# output[n] corresponds to a target whose echo starts at sample n
compressed = compressed_full[
    mf_delay:mf_delay + len(dwell_samples)
]
# Reshape into [pulse, range]
compressed_map = compressed.reshape(
    num_pulses,
    num_samples_pri
)
compressed_map = compressed_map[:, num_samples_pulse:]
# Window in slow time
slow_time_window = np.hanning(num_pulses)[:, None]

# Apply Doppler FFT along pulse dimension
doppler_map = np.fft.fft(
    compressed_map * slow_time_window,
    n=fft_len,
    axis=0
)

# Shift zero Doppler to the center
# doppler_map = np.fft.fftshift(
#     doppler_map,
#     axes=0
# )

# Power in dB
doppler_power_db = 10 * np.log10(
    np.abs(doppler_map)**2 + 1e-12
)

fig=go.Figure(
    go.Heatmap(
        z=doppler_power_db
    )
)
range_gate_res_f_sampling=c/(2*f_sample)
doppler_bin_res_f_sampling=prf/fft_len
range_bin,doppler_bin=target_to_indices(target_range=target_range,
                                        target_velocity=target_velocity,
                                        range_gate_res=range_gate_res_f_sampling,
                                        pri=pri,
                                        fft_len=fft_len,
                                        doppler_bin_res=doppler_bin_res_f_sampling)


fig.add_trace(go.Scatter(
    x=[np.round(range_bin-num_samples_pulse)],
    y=[np.round(doppler_bin)],
    # y=[np.round( f_d/(prf/fft_len))],
    mode='markers',
    marker=dict(
        color='red',
        size=10,
        symbol='x'
    ),
    name='Target'
))
fig.update_layout(
    title='Doppler Map',
    xaxis_title='Range Bin',
    yaxis_title='Doppler Bin'
)
fig.show()


