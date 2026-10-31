"""Funciones filtros notch"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

Pair = tuple[int, int]
Padding = tuple[int, int]


def pad_reflect(image: np.ndarray, padding: Padding) -> np.ndarray:
    """Agrega padding por reflexión en el orden (vertical, horizontal)."""
    py, px = padding
    return np.pad(image, ((py, py), (px, px)), mode="reflect")


def crop_padding(image: np.ndarray, padding: Padding) -> np.ndarray:
    """Elimina el padding y recupera el tamaño original."""
    py, px = padding
    y_end = image.shape[0] - py if py else image.shape[0]
    x_end = image.shape[1] - px if px else image.shape[1]
    return image[py:y_end, px:x_end]


def frequency_coordinates(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Crea coordenadas enteras con la frecuencia cero en el centro."""
    height, width = shape
    v = np.arange(-height // 2, height - height // 2)
    u = np.arange(-width // 2, width - width // 2)
    return np.meshgrid(u, v)


def notch_reject(
    shape: tuple[int, int],
    pairs: Iterable[Pair],
    kind: str = "ideal",
    width: float = 1.0,
) -> np.ndarray:
    """Construye una máscara notch ideal o gaussiana con pares simétricos."""
    if width <= 0:
        raise ValueError("width debe ser mayor que cero")
    if kind not in {"ideal", "gaussian"}:
        raise ValueError("kind debe ser 'ideal' o 'gaussian'")

    coordinates_u, coordinates_v = frequency_coordinates(shape)
    mask = np.ones(shape, dtype=np.float64)
    for u0, v0 in pairs:
        for center_u, center_v in ((u0, v0), (-u0, -v0)):
            distance_squared = (
                (coordinates_u - center_u) ** 2
                + (coordinates_v - center_v) ** 2
            )
            if kind == "ideal":
                mask[distance_squared <= width**2] = 0.0
            else:
                mask *= 1.0 - np.exp(-distance_squared / (2.0 * width**2))
    return mask


def scaled_notch_pairs(
    original_shape: tuple[int, int],
    padding: Padding,
    pairs: Sequence[Pair],
) -> list[Pair]:
    """Escala los centros al tamaño del espectro después del padding."""
    py, px = padding
    original_height, original_width = original_shape
    height = original_height + 2 * py
    width = original_width + 2 * px
    return [
        (int(round(u * width / original_width)), int(round(v * height / original_height)))
        for u, v in pairs
    ]


def apply_notch(
    image: np.ndarray,
    pairs: Sequence[Pair],
    kind: str = "gaussian",
    width: float = 2.0,
    padding: Padding = (32, 32),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Aplica FFT, máscara notch e IFFT, y recorta el padding."""
    padded = pad_reflect(image, padding)
    spectrum_before = np.fft.fftshift(np.fft.fft2(padded))
    scaled_pairs = scaled_notch_pairs(image.shape, padding, pairs)
    mask = notch_reject(padded.shape, scaled_pairs, kind=kind, width=width)
    spectrum_after = mask * spectrum_before
    filtered_padded = np.real(
        np.fft.ifft2(np.fft.ifftshift(spectrum_after))
    )
    return (
        crop_padding(filtered_padded, padding),
        spectrum_before,
        mask,
        spectrum_after,
    )


def log_spectrum(spectrum: np.ndarray) -> np.ndarray:
    """Calcula la magnitud logarítmica de un espectro complejo."""
    return np.log1p(np.abs(spectrum))
