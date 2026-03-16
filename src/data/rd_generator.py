from src.data.base_generator import BaseDDMGenerator
import numpy as np
import tensordict
from scipy.constants import speed_of_light
import plotly.graph_objects as go

class RDGenerator(BaseDDMGenerator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs) 
        self.target_range_interval = kwargs.get('target_range_interval', (0, 1.0))
        self.max_target_range_m = kwargs.get('max_target_range_m', 1.0)
        self.target_doppler_interval = kwargs.get('target_doppler_interval_mps', (-1.0, 1.0))
        self.max_target_doppler_mps = kwargs.get('max_target_doppler_mps', 1.0)
        self.num_targets = kwargs.get('num_targets', (1, 5))
        self.rcs_m2 = kwargs.get('rcs_m2', (0, 20))
        self.num_pulses_interval = kwargs.get('num_pulses', (16, 16))
        self.dwell_time_s_interval = kwargs.get('dwell_time_s_interval', (1.0, 1.0))
        self.sampling_rate_hz = kwargs.get('sampling_rate_hz', 40e6)  # 40 MHz sampling rate
        # Radar parameters (placeholders, can be expanded as needed)
        self.speed_of_light_mps = speed_of_light  # Speed of light in m/s
        self.carrier_frequency_hz = kwargs.get('carrier_frequency_hz', 1.0*1e9)# 1 GHz
        self.p_transmit = kwargs.get('p_transmit', 1.0)  # Transmit power
        self.antenna_gain = kwargs.get('antenna_gain', 1.0)  # Antenna gain
        self.wavelength =  self.speed_of_light_mps/self.carrier_frequency_hz  # Wavelength (m)
        self.duty_cycle = kwargs.get('duty_cycle', 0.1)  # Duty cycle for pulse generation
        self.pulse_bw_hz = kwargs.get('pulse_bw_hz', 0.5*1e6)  # Pulse bandwidth in Hz
        self.pulse_modulation = kwargs.get('pulse_modulation', 'LFM')  # Pulse modulation type (e.g., 'LFM')
    
    def get_scaled_pulse(self,scale_factor):
        # Placeholder for actual pulse scaling logic
        return scale_factor * np.ones((self.delay_bins, self.doppler_bins), dtype=np.float32)

    def draw_num_pusles(self):
        return self.rng.integers(*self.num_pulses_interval) if self.num_pulses_interval[0] != self.num_pulses_interval[1] else self.num_pulses_interval[0]

    def generate_pulse(self, pulse_duration_s):

        N = int(self.sampling_rate_hz * pulse_duration_s)
        pulse_t = np.linspace(0, pulse_duration_s, N, endpoint=False)

        if self.pulse_modulation == "LFM":

            pulse = np.exp(
                1j * 2 * np.pi * self.carrier_frequency_hz * pulse_t +
                1j * np.pi * self.pulse_bw_hz / pulse_duration_s * pulse_t**2
            )

        elif self.pulse_modulation == "triangular":

            x = pulse_t / pulse_duration_s
            triangle = 1 - np.abs(2*x - 1)

            inst_freq = self.carrier_frequency_hz + self.pulse_bw_hz * (triangle - 0.5)

            phase = 2*np.pi*np.cumsum(inst_freq)/self.sampling_rate_hz

            pulse = np.exp(1j*phase)

        else:
            raise ValueError(f"Unsupported modulation: {self.pulse_modulation}")

        return pulse

    def simulate_ddm(self) -> tensordict.TensorDict:
        """
        Simulate a simple RD map with random noise and a few random peaks.
        Returns:
            ddm: np.ndarray, shape = (2,H,W) if complex, else (1,H,W)
        """
        # Create an empty DDM
        channels = 2 if self.complex_ddm else 1
        dwell_time_s = self.rng.uniform(*self.dwell_time_s_interval)
        num_pulses = self.draw_num_pusles()
        pri = dwell_time_s / num_pulses
        num_samples_dwell = dwell_time_s * self.sampling_rate_hz 
        ddm = np.zeros(int( num_samples_dwell), dtype=np.complex64)
        pulse =   self.generate_pulse(pulse_duration_s=self.duty_cycle*pri)# Get a scaled pulse (placeholder)
        # Add random noise
        ddm += self.rng.normal(0, 0.1, size=ddm.shape)
        #inject original pulse to the DDM at regular intervals based on PRI
        for i in range(num_pulses):
            offset = int(i * pri * self.sampling_rate_hz)
            ddm[offset:offset+len(pulse)] += pulse
        # Add a few random peaks to simulate targets
        num_peaks = self.rng.integers(self.num_targets[0], self.num_targets[1] + 1)
        for i in range(num_peaks):
            target_range ,target_doppler , target_power = self.draw_random_target()
            self.inject_target_to_dwell(ddm, num_pulses, target_range, target_doppler, target_power, pulse, num_samples_dwell)
        ddm = ddm.reshape((num_pulses, int(num_samples_dwell/num_pulses)))
        '''
        import plotly.express as px
        px.imshow(np.abs(ddm[:, ::100]), aspect="auto")
        '''
        return ddm  
    
    def inject_target_to_dwell(self, ddm,num_pulses, target_range, target_doppler, target_power, pulse, num_samples_dwell):
        # Calculate the time delay and Doppler shift for the target
        pri_n_samples = int(num_samples_dwell / num_pulses)
        random_teta_offset = self.rng.uniform(0, 2 * np.pi)  # Random initial phase offset
        time_delay_s = 2 * target_range / self.speed_of_light_mps  # Round-trip time delay
        doppler_frequency_hz = 2 * target_doppler / self.wavelength  # Doppler shift in Hz 
        delta_teta = 2 * np.pi * doppler_frequency_hz   # Doppler phase shift
        # Calculate the sample index for the time delay
        sample_index_start = int(time_delay_s * self.sampling_rate_hz)
        sample_index_end = sample_index_start + len(pulse)
        for pulse_n in range(num_pulses):
            shifted_scaled_pulse = target_power * pulse * np.exp(1j * delta_teta * pulse_n+random_teta_offset)  # Apply Doppler shift
            ddm[pulse_n*pri_n_samples+sample_index_start:pulse_n*pri_n_samples+sample_index_end] += shifted_scaled_pulse
        
    def draw_random_target(self):
        target_range = self.rng.uniform(*self.target_range_interval)
        target_doppler = self.rng.uniform(*self.target_doppler_interval)  # Example Doppler range
        target_rcs = self.rng.uniform(*self.rcs_m2)  # Example RCS range
        target_power = self.p_transmit *(self.antenna_gain**2) *(self.wavelength)**2 *target_rcs  / (4*np.pi)**3*target_range# Convert RCS to SNR using a simple radar equation model (this is a placeholder)
        return target_range, target_doppler, target_power

if __name__ == "__main__":
    generator = RDGenerator(complex_ddm=True, delay_bins=32, doppler_bins=32, seed=42)
    sample_ddm = generator.simulate_ddm()
    print("Sample DDM shape:", sample_ddm.shape)
     
