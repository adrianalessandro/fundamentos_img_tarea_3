"""Operaciones de degradacion y restauracion de imagenes"""

from __future__ import annotations

import numpy as np


def psf_to_frequency(psf: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Convierte una PSF espacial centrada en su origen de convolución."""
    if (
        psf.ndim != 2
        or psf.shape[0] > shape[0]
        or psf.shape[1] > shape[1]
        or psf.sum() <= 0
    ):
        raise ValueError("La PSF debe ser 2D, positiva y caber en la imagen")
    padded = np.zeros(shape, dtype=np.float64)
    height, width = psf.shape
    padded[:height, :width] = psf / psf.sum()
    centered = np.roll(padded, -(height // 2), axis=0)
    centered = np.roll(centered, -(width // 2), axis=1)
    return np.fft.fft2(centered)


def apply_blur(image: np.ndarray, frequency_response: np.ndarray) -> np.ndarray:
    """Aplica desenfoque circular mediante multiplicación en frecuencia."""
    return np.real(np.fft.ifft2(frequency_response * np.fft.fft2(image)))


def apply_adjoint(image: np.ndarray, frequency_response: np.ndarray) -> np.ndarray:
    """Aplica el operador adjunto del desenfoque."""
    return np.real(
        np.fft.ifft2(np.conj(frequency_response) * np.fft.fft2(image))
    )


def wiener_deconvolution(
    observation: np.ndarray,
    frequency_response: np.ndarray,
    regularization: float,
) -> np.ndarray:
    """Calcula la restauración de Wiener con regularización positiva."""
    if regularization <= 0:
        raise ValueError("regularization debe ser positivo")
    observation_frequency = np.fft.fft2(observation)
    restored_frequency = (
        np.conj(frequency_response)
        * observation_frequency
        / (np.abs(frequency_response) ** 2 + regularization)
    )
    return np.real(np.fft.ifft2(restored_frequency))


def rmse(first: np.ndarray, second: np.ndarray) -> float:
    """Calcula el error cuadrático medio."""
    return float(np.sqrt(np.mean((first - second) ** 2)))


def perturb_psf(psf: np.ndarray) -> np.ndarray:
    """Introduce un error de centrado de un píxel en la PSF."""
    return np.roll(psf, 1, axis=1)


def tikhonov_cost(
    estimate: np.ndarray,
    observation: np.ndarray,
    frequency_response: np.ndarray,
    regularization: float,
) -> float:
    """Evalúa el funcional de Tikhonov de norma L2."""
    residual = apply_blur(estimate, frequency_response) - observation
    return float(
        0.5 * np.sum(residual**2)
        + 0.5 * regularization * np.sum(estimate**2)
    )


def tikhonov_gradient_descent(
    observation: np.ndarray,
    frequency_response: np.ndarray,
    regularization: float,
    step: float,
    reference: np.ndarray,
    max_iter: int = 300,
    tolerance: float = 1e-7,
    initial: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resuelve Tikhonov mediante descenso de gradiente."""
    if regularization < 0 or step <= 0 or max_iter <= 0:
        raise ValueError(
            "regularization debe ser no negativa, step positivo e iteraciones positivas"
        )
    estimate = (
        np.zeros_like(observation)
        if initial is None
        else np.array(initial, dtype=np.float64, copy=True)
    )
    costs = [
        tikhonov_cost(
            estimate, observation, frequency_response, regularization
        )
    ]
    errors = [rmse(estimate, reference)]
    for _ in range(max_iter):
        residual = apply_blur(estimate, frequency_response) - observation
        gradient = (
            apply_adjoint(residual, frequency_response)
            + regularization * estimate
        )
        next_estimate = estimate - step * gradient
        next_cost = tikhonov_cost(
            next_estimate, observation, frequency_response, regularization
        )
        costs.append(next_cost)
        errors.append(rmse(next_estimate, reference))
        relative_change = abs(costs[-2] - costs[-1]) / max(
            abs(costs[-2]), 1e-12
        )
        estimate = next_estimate
        if relative_change < tolerance:
            break
    return estimate, np.asarray(costs), np.asarray(errors)


def closed_form_tikhonov(
    observation: np.ndarray,
    frequency_response: np.ndarray,
    regularization: float,
) -> np.ndarray:
    """Calcula la solución cerrada de Tikhonov en frecuencia."""
    if regularization < 0:
        raise ValueError("regularization debe ser no negativa")
    observation_frequency = np.fft.fft2(observation)
    estimate_frequency = (
        np.conj(frequency_response)
        * observation_frequency
        / (np.abs(frequency_response) ** 2 + regularization)
    )
    return np.real(np.fft.ifft2(estimate_frequency))
