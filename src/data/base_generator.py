import numpy as np
from abc import ABC, abstractmethod
import tensordict

class BaseDDMGenerator(ABC):
    """
    Base class for Delay-Doppler Map (DDM) generators.
    Subclasses implement domain-specific simulation.
    Compatible with Hydra and IterableDataset.
    """

    def __init__(
        self,
        complex_ddm: bool = True,
        delay_bins: int = 32,
        doppler_bins: int = 32,
        seed: int = None,
    ):
        '''

        Args:
            complex_ddm: If True, output DDMs have 2 channels [real, imag]. If False, output has 1 channel.
            delay_bins: Number of bins in the delay dimension.
            doppler_bins: Number of bins in the Doppler dimension.
            seed: Random seed for reproducibility.
        '''
        self.complex_ddm = complex_ddm
        self.delay_bins = delay_bins
        self.doppler_bins = doppler_bins
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def simulate_ddm(self) -> tensordict.TensorDict:
        """
        Generate one DDM sample.
        Must be implemented by subclasses.

        Returns:
            ddm: np.ndarray, shape = (2,H,W) if complex, else (1,H,W)
        """
        pass

    def __call__(self) -> tensordict.TensorDict:
        """
        Callable interface, generates one DDM sample.
        """
        ddm = self.simulate_ddm()

        # Ensure correct output shape
        if self.complex_ddm:
            if ddm.shape[0] != 2:
                raise ValueError("Complex DDM must have 2 channels [real, imag]")
        else:
            if ddm.shape[0] != 1:
                raise ValueError("Real DDM must have 1 channel")

        return ddm