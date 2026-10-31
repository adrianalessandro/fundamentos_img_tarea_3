"""Ejecucion reproducible (orquestador)"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from funciones import (
    apply_notch,
    log_spectrum,
    scaled_notch_pairs,
)

ROOT = Path(__file__).resolve().parent.parent
IMAGE_PATH = ROOT / "img" / "P1.CR2"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
NOTCH_PAIRS = [(1058, 0)]
DEFAULT_PADDING = (32, 32)
WIDTHS = (1.0, 2.0, 3.0, 5.0)


def cargar_imagen(path: Path = IMAGE_PATH) -> np.ndarray:
    """Carga la imagen como escala de grises normalizada en [0, 1]."""
    return np.asarray(Image.open(path).convert("L"), dtype=np.float64) / 255.0


def guardar_figura(figura: plt.Figure, name: str) -> None:
    """Guarda una figura en output con resolución adecuada para el reporte."""
    figura.savefig(OUTPUT_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(figura)


def _mostrar_imagen(
    axis: plt.Axes,
    image: np.ndarray,
    title: str,
    *,
    difference: bool = False,
    spectrum: bool = False,
    spectrum_vmax: float | None = None,
) -> None:
    if difference:
        axis.imshow(image, vmin=-0.18, vmax=0.18, cmap="coolwarm")
    elif spectrum:
        axis.imshow(image, vmin=0, vmax=spectrum_vmax)
    else:
        axis.imshow(image, vmin=0, vmax=1, cmap="gray")
    axis.set_title(title)
    axis.axis("off")


def guardar_imagen_entrada(image: np.ndarray) -> None:
    figura, axis = plt.subplots(figsize=(5, 5))
    _mostrar_imagen(axis, image, "Imagen de entrada en escala de grises")
    guardar_figura(figura, "01_imagen_entrada.png")


def guardar_espectro(
    image: np.ndarray,
    pairs: Iterable[tuple[int, int]],
) -> None:
    spectrum = log_spectrum(np.fft.fftshift(np.fft.fft2(image)))
    figura, axis = plt.subplots(figsize=(8, 6))
    axis.imshow(spectrum)
    center_y, center_x = np.array(spectrum.shape) // 2
    for u0, v0 in pairs:
        for u, v in ((u0, v0), (-u0, -v0)):
            axis.plot(center_x + u, center_y + v, "rx", ms=10, mew=2)
            axis.text(
                center_x + u + 3,
                center_y + v + 3,
                f"({u},{v})",
                color="yellow",
                fontsize=9,
            )
    axis.set_title("Espectro logarítmico y centros seleccionados")
    axis.set_xlabel("u (horizontal)")
    axis.set_ylabel("v (vertical)")
    guardar_figura(figura, "02_espectro_notches.png")


def calcular_resultados(
    image: np.ndarray,
    pairs: list[tuple[int, int]],
) -> dict[tuple[str, float], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    results = {}
    for kind in ("ideal", "gaussian"):
        for width in WIDTHS:
            filtered, spectrum, mask, spectrum_after = apply_notch(
                image, pairs, kind=kind, width=width, padding=DEFAULT_PADDING
            )
            results[(kind, width)] = (
                filtered,
                spectrum,
                mask,
                spectrum_after,
                image - filtered,
            )
    return results


def guardar_comparacion_anchos(
    results: dict[tuple[str, float], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
) -> None:
    figura, axes = plt.subplots(2, 4, figsize=(16, 8))
    for row, kind in enumerate(("ideal", "gaussian")):
        for col, width in enumerate(WIDTHS):
            filtered = results[(kind, width)][0]
            _mostrar_imagen(axes[row, col], filtered, f"{kind}, width={width:g}")
    figura.suptitle("Imagen filtrada para distintos anchos")
    figura.tight_layout()
    guardar_figura(figura, "03_comparacion_anchos.png")


def guardar_casos_principales(
    image: np.ndarray,
    results: dict[tuple[str, float], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
) -> None:
    spectrum_values = [
        log_spectrum(results[(kind, 2.0)][1])
        for kind in ("ideal", "gaussian")
    ]
    spectrum_after_values = [
        log_spectrum(results[(kind, 2.0)][3])
        for kind in ("ideal", "gaussian")
    ]
    spectrum_vmax = max(
        spectrum.max()
        for spectrum in spectrum_values + spectrum_after_values
    )
    figura, axes = plt.subplots(2, 4, figsize=(16, 8))
    for row, (kind, width) in enumerate((("ideal", 2.0), ("gaussian", 2.0))):
        filtered, spectrum, _, spectrum_after, difference = results[(kind, width)]
        panels = (
            (image, "entrada", False, False),
            (log_spectrum(spectrum), "espectro antes", False, True),
            (log_spectrum(spectrum_after), "espectro después", False, True),
            (difference, "diferencia", True, False),
        )
        for col, (data, title, is_difference, is_spectrum) in enumerate(panels):
            _mostrar_imagen(
                axes[row, col],
                data,
                f"{kind}, width={width:g}: {title}",
                difference=is_difference,
                spectrum=is_spectrum,
                spectrum_vmax=spectrum_vmax,
            )
    figura.tight_layout()
    guardar_figura(figura, "04_casos_principales.png")


def ejecutar_comparacion_padding(image: np.ndarray, pairs: list[tuple[int, int]]) -> None:
    padding_cases = ((0, 0), (32, 32), (64, 64))
    padding_results = {}
    for padding in padding_cases:
        filtered, _, _, _ = apply_notch(
            image, pairs, kind="gaussian", width=2.0, padding=padding
        )
        padding_results[padding] = filtered
        difference = image - filtered
        print(
            f"padding={padding}: std(diferencia)={np.std(difference):.5f}, "
            f"media(abs(diferencia))={np.mean(np.abs(difference)):.5f}"
        )

    figura, axes = plt.subplots(1, 3, figsize=(15, 5))
    for axis, padding in zip(axes, padding_cases):
        _mostrar_imagen(
            axis,
            padding_results[padding],
            f"Gaussiano, width=2, padding={padding[0]}",
        )
    figura.tight_layout()
    guardar_figura(figura, "05_comparacion_padding.png")

    figura, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis, padding in zip(axes, padding_cases):
        _mostrar_imagen(
            axis,
            padding_results[padding][:60, :60],
            f"Esquina, padding={padding[0]}",
        )
    figura.tight_layout()
    guardar_figura(figura, "06_zoom_padding.png")


def ejecutar_caso_final(image: np.ndarray, pairs: list[tuple[int, int]]) -> None:
    final_image, spectrum, mask, spectrum_after = apply_notch(
        image, pairs, kind="gaussian", width=2.0, padding=DEFAULT_PADDING
    )
    py, px = DEFAULT_PADDING
    padded_shape = (image.shape[0] + 2 * py, image.shape[1] + 2 * px)
    center_y, center_x = np.array(padded_shape) // 2
    u0, v0 = scaled_notch_pairs(image.shape, DEFAULT_PADDING, pairs)[0]
    row, col = center_y + v0, center_x + u0
    complex_before = spectrum[row, col]
    mask_value = mask[row, col]
    complex_after = spectrum_after[row, col]
    print(f"Frecuencia relativa consultada: (u,v)=({u0},{v0})")
    print(f"F antes = {complex_before.real:.6e} + {complex_before.imag:.6e}j")
    print(f"H aplicado = {mask_value:.6e}")
    print(f"F después = {complex_after.real:.6e} + {complex_after.imag:.6e}j")
    print(
        "Comprobación |F después - H*F antes| = "
        f"{abs(complex_after - mask_value * complex_before):.3e}"
    )

    figura, axes = plt.subplots(1, 3, figsize=(14, 4))
    _mostrar_imagen(axes[0], image, "Entrada")
    _mostrar_imagen(axes[1], final_image, "Resultado final")
    _mostrar_imagen(axes[2], image - final_image, "Diferencia", difference=True)
    figura.tight_layout()
    guardar_figura(figura, "07_resultado_final.png")


def ejecutar_pregunta_1() -> None:
    """Ejecuta todos los experimentos del notebook y guarda sus figuras."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    image = cargar_imagen()
    pairs = NOTCH_PAIRS
    print(f"Tamaño original: {image.shape}; rango: [{image.min():.3f}, {image.max():.3f}]")
    scaled_pairs = scaled_notch_pairs(image.shape, DEFAULT_PADDING, pairs)
    print(
        f"Pares de notches: {scaled_pairs}; padding elegido: "
        f"{DEFAULT_PADDING} pixeles por borde"
    )
    guardar_imagen_entrada(image)
    guardar_espectro(image, NOTCH_PAIRS)
    results = calcular_resultados(image, pairs)
    for (kind, width), (_, _, _, _, difference) in results.items():
        print(
            f"{kind}, width={width:g}: std(diferencia)={np.std(difference):.5f}, "
            f"media(abs(diferencia))={np.mean(np.abs(difference)):.5f}"
        )
    guardar_comparacion_anchos(results)
    guardar_casos_principales(image, results)
    ejecutar_comparacion_padding(image, pairs)
    ejecutar_caso_final(image, pairs)
    print(f"Salidas guardadas en: {OUTPUT_DIR}")
