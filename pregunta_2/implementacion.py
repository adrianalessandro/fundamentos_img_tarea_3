"""Ejecucion reproducible (orquestador)"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from funciones import (
    apply_blur,
    closed_form_tikhonov,
    perturb_psf,
    psf_to_frequency,
    rmse,
    tikhonov_gradient_descent,
    wiener_deconvolution,
)

ROOT = Path(__file__).resolve().parent.parent
IMAGE_PATH = ROOT / "img" / "cameraman.jpg"
PSF_PATH = ROOT / "img" / "psf.png"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
SEED = 2714
NOISE_LEVELS = (0.01, 0.03)
K_VALUES = np.logspace(-6, -1, 31)
LAMBDA_VALUES = (1e-5, 1e-3, 1e-2)
TAU_VALUES = (1.0, 2.2)


def cargar_datos() -> tuple[np.ndarray, np.ndarray]:
    image = np.asarray(Image.open(IMAGE_PATH).convert("L"), dtype=np.float64) / 255.0
    psf = np.asarray(Image.open(PSF_PATH).convert("L"), dtype=np.float64)
    psf /= psf.sum()
    return image, psf


def guardar_figura(figure: plt.Figure, name: str) -> None:
    figure.savefig(OUTPUT_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(figure)


def mostrar_imagen(
    axis: plt.Axes,
    image: np.ndarray,
    title: str,
    *,
    error: bool = False,
) -> None:
    if error:
        axis.imshow(image, vmin=0, vmax=0.25, cmap="magma")
    else:
        axis.imshow(image, vmin=0, vmax=1, cmap="gray")
    axis.set_title(title)
    axis.axis("off")


def guardar_entrada(image: np.ndarray, psf: np.ndarray) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(9, 4))
    mostrar_imagen(axes[0], image, "Imagen ideal")
    axes[1].imshow(psf, vmin=0, vmax=psf.max(), cmap="gray")
    axes[1].set_title("PSF normalizada")
    axes[1].axis("off")
    figure.tight_layout()
    guardar_figura(figure, "01_entrada_y_psf.png")


def guardar_respuesta_frecuencia(
    frequency_response: np.ndarray,
    shape: tuple[int, int],
) -> None:
    magnitude = np.abs(np.fft.fftshift(frequency_response))
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    axes[0].imshow(
        np.fft.fftshift(np.log1p(np.abs(frequency_response))), cmap="magma"
    )
    axes[0].set_title("log(1 + |H|), centrado")
    axes[1].plot(np.fft.fftshift(magnitude)[shape[0] // 2])
    
    axes[1].set_title("Corte central de |H|")
    axes[1].set_xlabel("frecuencia horizontal")
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    guardar_figura(figure, "02_respuesta_frecuencia.png")


def crear_observaciones(
    image: np.ndarray,
    frequency_response: np.ndarray,
) -> dict[float, np.ndarray]:
    rng = np.random.default_rng(SEED)
    observations = {}
    for noise_level in NOISE_LEVELS:
        noise = rng.normal(0.0, noise_level, size=image.shape)
        observations[noise_level] = apply_blur(image, frequency_response) + noise
    return observations


def guardar_observaciones(
    image: np.ndarray,
    observations: dict[float, np.ndarray],
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(13, 4))
    mostrar_imagen(axes[0], image, "x ideal")
    mostrar_imagen(axes[1], observations[0.01], "y, sigma=0.01")
    mostrar_imagen(axes[2], observations[0.03], "y, sigma=0.03")
    figure.tight_layout()
    guardar_figura(figure, "03_observaciones_degradadas.png")


def ejecutar_wiener(
    image: np.ndarray,
    observations: dict[float, np.ndarray],
    frequency_response: np.ndarray,
) -> tuple[dict[float, np.ndarray], dict[float, tuple[float, float]]]:
    errors = {}
    best = {}
    for noise_level, observation in observations.items():
        error_values = np.array(
            [
                rmse(
                    wiener_deconvolution(
                        observation, frequency_response, regularization
                    ),
                    image,
                )
                for regularization in K_VALUES
            ]
        )
        errors[noise_level] = error_values
        best_index = int(np.argmin(error_values))
        best[noise_level] = (
            float(K_VALUES[best_index]),
            float(error_values[best_index]),
        )
        print(
            f"sigma={noise_level:.2f}: K óptimo={K_VALUES[best_index]:.3e}, "
            f"RMSE={error_values[best_index]:.6f}"
        )
    return errors, best


def guardar_wiener_rmse(
    errors: dict[float, np.ndarray],
    best: dict[float, tuple[float, float]],
) -> None:
    figure, axis = plt.subplots(figsize=(7, 4))
    for noise_level in NOISE_LEVELS:
        axis.semilogx(
            K_VALUES,
            errors[noise_level],
            "o-",
            label=f"sigma={noise_level}",
        )
        best_k, best_error = best[noise_level]
        axis.scatter([best_k], [best_error], s=55)
    axis.set_xlabel("K")
    axis.set_ylabel("RMSE")
    axis.set_title("Wiener: RMSE versus K")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    guardar_figura(figure, "04_wiener_rmse.png")


def guardar_reconstrucciones_wiener(
    image: np.ndarray,
    observations: dict[float, np.ndarray],
    frequency_response: np.ndarray,
    best: dict[float, tuple[float, float]],
) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(13, 8))
    for row, noise_level in enumerate(NOISE_LEVELS):
        best_k, best_error = best[noise_level]
        reconstruction = wiener_deconvolution(
            observations[noise_level], frequency_response, best_k
        )
        mostrar_imagen(
            axes[row, 0],
            observations[noise_level],
            f"Degradada, sigma={noise_level}",
        )
        mostrar_imagen(
            axes[row, 1],
            np.clip(reconstruction, 0, 1),
            f"Wiener, K={best_k:.2e}, RMSE={best_error:.4f}",
        )
        mostrar_imagen(
            axes[row, 2],
            np.abs(image - reconstruction),
            "Error absoluto",
            error=True,
        )
    figure.tight_layout()
    guardar_figura(figure, "05_reconstrucciones_wiener.png")


def guardar_sensibilidad_psf(
    image: np.ndarray,
    observation: np.ndarray,
    frequency_response: np.ndarray,
    psf: np.ndarray,
    regularization: float,
) -> None:
    wrong_response = psf_to_frequency(perturb_psf(psf), image.shape)
    correct = wiener_deconvolution(observation, frequency_response, regularization)
    wrong = wiener_deconvolution(observation, wrong_response, regularization)
    print(f"PSF correcta: {psf.shape}, PSF incorrecta: {perturb_psf(psf).shape}")
    print(f"RMSE correcta={rmse(correct, image):.6f}")
    print(f"RMSE incorrecta={rmse(wrong, image):.6f}")

    figure, axes = plt.subplots(1, 3, figsize=(13, 4))
    mostrar_imagen(axes[0], np.clip(correct, 0, 1), "PSF correcta")
    mostrar_imagen(axes[1], np.clip(wrong, 0, 1), "PSF incorrecta")
    mostrar_imagen(
        axes[2], np.abs(correct - wrong), "Diferencia entre reconstrucciones", error=True
    )
    figure.tight_layout()
    guardar_figura(figure, "06_sensibilidad_psf.png")

    crop = np.s_[80:140, 80:140]
    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    mostrar_imagen(axes[0], image[crop], "Ideal (recorte)")
    mostrar_imagen(axes[1], np.clip(correct[crop], 0, 1), "PSF correcta")
    mostrar_imagen(axes[2], np.clip(wrong[crop], 0, 1), "PSF incorrecta")
    figure.tight_layout()
    guardar_figura(figure, "07_zoom_sensibilidad_psf.png")


def ejecutar_tikhonov(
    image: np.ndarray,
    observation: np.ndarray,
    frequency_response: np.ndarray,
    regularization_values: tuple[float, ...],
) -> dict[tuple[float, float], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    runs = {}
    for regularization in regularization_values:
        for step in TAU_VALUES:
            result = tikhonov_gradient_descent(
                observation,
                frequency_response,
                regularization,
                step,
                image,
                max_iter=120,
                tolerance=1e-9,
            )
            runs[(regularization, step)] = result
            estimate, costs, errors = result
            print(
                f"lambda={regularization:.0e}, tau={step:.1f}: "
                f"iteraciones={len(costs)-1}, costo={costs[-1]:.4e}, "
                f"RMSE={errors[-1]:.6f}, costo finito={np.all(np.isfinite(costs))}"
            )
    return runs


def guardar_tikhonov_runs(
    runs: dict[tuple[float, float], tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 4))
    regularization_values = sorted({regularization for regularization, _ in runs})
    for regularization in regularization_values:
        for step in TAU_VALUES:
            _, costs, errors = runs[(regularization, step)]
            label = f"lambda={regularization:.0e}, tau={step}"
            axes[0].plot(costs, label=label)
            axes[1].plot(errors, label=label)
    axes[0].set_title("Costo por iteración")
    axes[1].set_title("RMSE por iteración")
    for axis in axes:
        axis.set_xlabel("iteración")
        axis.set_yscale("log")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.tight_layout()
    guardar_figura(figure, "08_tikhonov_convergencia.png")


def guardar_comparacion_tikhonov(
    image: np.ndarray,
    observation: np.ndarray,
    frequency_response: np.ndarray,
    regularization: float,
) -> None:
    comparison = {}
    for current_lambda in (regularization, 0.0):
        closed = closed_form_tikhonov(
            observation, frequency_response, current_lambda
        )
        iterative, costs, _ = tikhonov_gradient_descent(
            observation,
            frequency_response,
            current_lambda,
            1.0,
            image,
            max_iter=500,
            tolerance=1e-9,
        )
        comparison[current_lambda] = (closed, iterative)
        print(
            f"lambda={current_lambda:.3e}: "
            f"RMSE cerrada={rmse(closed, image):.6f}, "
            f"RMSE iterativa={rmse(iterative, image):.6f}, "
            f"RMSE entre soluciones={rmse(closed, iterative):.6e}"
        )

    closed, iterative = comparison[regularization]
    crop = np.s_[80:140, 80:140]
    figure, axes = plt.subplots(2, 3, figsize=(13, 8))
    mostrar_imagen(axes[0, 0], observation, "Degradada")
    mostrar_imagen(axes[0, 1], np.clip(closed, 0, 1), "Cerrada, lambda=K")
    mostrar_imagen(axes[0, 2], np.clip(iterative, 0, 1), "Iterativa, lambda=K")
    mostrar_imagen(axes[1, 0], image[crop], "Ideal (recorte)")
    mostrar_imagen(axes[1, 1], np.clip(closed[crop], 0, 1), "Cerrada (recorte)")
    mostrar_imagen(axes[1, 2], np.clip(iterative[crop], 0, 1), "Iterativa (recorte)")
    figure.tight_layout()
    guardar_figura(figure, "09_comparacion_tikhonov.png")


def ejecutar_pregunta_2() -> None:
    """Ejecuta los experimentos del notebook y guarda todas sus figuras."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    image, psf = cargar_datos()
    frequency_response = psf_to_frequency(psf, image.shape)
    print(f"Imagen ideal: {image.shape}; rango [{image.min():.3f}, {image.max():.3f}]")
    print(
        f"PSF original: {psf.shape}; suma={psf.sum():.6f}; "
        f"rango [{psf.min():.3f}, {psf.max():.3f}]"
    )
    print(
        f"H: forma={frequency_response.shape}; "
        f"|H| minimo={np.abs(np.fft.fftshift(frequency_response)).min():.6e}; "
        f"|H| maximo={np.abs(frequency_response).max():.6f}"
    )
    guardar_entrada(image, psf)
    guardar_respuesta_frecuencia(frequency_response, image.shape)
    observations = crear_observaciones(image, frequency_response)
    for noise_level, observation in observations.items():
        print(
            f"sigma={noise_level:.2f}: observacion en "
            f"[{observation.min():.3f}, {observation.max():.3f}]"
        )
    guardar_observaciones(image, observations)
    errors, best = ejecutar_wiener(image, observations, frequency_response)
    guardar_wiener_rmse(errors, best)
    guardar_reconstrucciones_wiener(image, observations, frequency_response, best)
    sigma = 0.03
    guardar_sensibilidad_psf(
        image, observations[sigma], frequency_response, psf, best[sigma][0]
    )
    tikhonov_lambdas = tuple(
        sorted(set(LAMBDA_VALUES) | {best[sigma][0]})
    )
    runs = ejecutar_tikhonov(
        image,
        observations[sigma],
        frequency_response,
        tikhonov_lambdas,
    )
    guardar_tikhonov_runs(runs)
    guardar_comparacion_tikhonov(
        image, observations[sigma], frequency_response, best[sigma][0]
    )
    print(f"Salidas guardadas en: {OUTPUT_DIR}")
