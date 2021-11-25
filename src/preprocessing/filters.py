import numpy as np

from typing import Union


# Homomorphic filter class
class HomomorphicFilter:
    """Homomorphic filter implemented with diferents filters and an option to an external filter.

    High-frequency filters implemented:
        butterworth
        gaussian
    Attributes:
        a, b: Floats used on emphasis filter:
            H = a + b*H

        .
    """
    def get_filter(self, shape: tuple, gl: Union[int, float] = 0.5, gh: Union[int, float] = 2, c: float = 0.5,
                   d0: int = 30) -> np.ndarray:
        p = shape[0] / 2
        q = shape[1] / 2
        u, v = np.meshgrid(range(shape[0]), range(shape[1]), sparse=False, indexing='ij')
        d = ((u - p) ** 2 + (v - q) ** 2).astype(float)
        h = (gh - gl) * (1 - (np.exp(-c * (d / d0) ** 2))) + gl

        return np.fft.fftshift(h)

    def filter(self, img, **filter_params):
        """
        Method to apply homormophic filter on an image
        Attributes:
            I: Single channel image
            filter_params: Parameters to be used on filters:
                butterworth:
                    filter_params[0]: Cutoff frequency
                    filter_params[1]: Order of filter
                gaussian:
                    filter_params[0]: Cutoff frequency
            filter: Choose of the filter, options:
                butterworth
                gaussian
                external
            H: Used to pass external filter
        """

        # Se obtiene el filtro
        h = self.get_filter(img.shape, **filter_params)

        # Take the image to log domain and then to frequency domain
        img_log = np.log1p(np.array(img.copy(), dtype=float))
        img_fft = np.fft.fft2(img_log)
        img_fft_filt = np.multiply(h, img_fft)
        img_filt = np.fft.ifft2(img_fft_filt)
        img = np.exp(np.real(img_filt)) - 1
        return np.uint8(img)
