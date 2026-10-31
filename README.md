# Fundamentos de procesamiento de imágenes - Tarea 3

Ejecutar desde la raiz, las imagenes de input estan en la carpeta img/.

## Pregunta 1: eliminación de ruido periódico

Consolidado en:
[`pregunta_1/`](./pregunta_1/).

- [`main_p1.py`](./pregunta_1/main_p1.py): script principal de ejecución.
- [`implementacion.py`](./pregunta_1/implementacion.py): carga, experimentos,
  visualizaciones y exportación de resultados.
- [`funciones.py`](./pregunta_1/funciones.py): padding, coordenadas de
  frecuencia, máscaras notch y operaciones FFT.
- [`pregunta_1.md`](./pregunta_1.md): respuestas a las preguntas guiadas.

Para ejecutar la pregunta 1 desde la raíz del repositorio:

```bash
python pregunta_1/main_p1.py
```

Las imágenes se guardan en
[`pregunta_1/output/`](./pregunta_1/output/).

## Pregunta 2: restauración de imágenes

Consolidado en:
[`pregunta_2/`](./pregunta_2/).

- [`main_p2.py`](./pregunta_2/main_p2.py): script principal de ejecución.
- [`implementacion.py`](./pregunta_2/implementacion.py): experimentos,
  métricas, visualizaciones y exportación de resultados.
- [`funciones.py`](./pregunta_2/funciones.py): conversión de PSF a frecuencia,
  desenfoque, Wiener y Tikhonov.
- [`pregunta_2.md`](./pregunta_2.md): respuestas a las preguntas guiadas.

Para ejecutar la pregunta 2 desde la raíz del repositorio:

```bash
python pregunta_2/main_p2.py
```

Las imágenes se guardan en
[`pregunta_2/output/`](./pregunta_2/output/).