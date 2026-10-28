# El hombre del tiempo: ¿qué tiempo hace en una foto?

**Algoritmos y Arquitecturas para Procesado de Imágenes** · Grado en Ciencia de Datos e Inteligencia Artificial · Curso 2026–2027
**Proyecto 14 · Tema 3: Imágenes como datos visuales**

**Autores:** NOMBRE_1 y NOMBRE_2

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TU_USUARIO/aapi-p14-hombre-del-tiempo/blob/main/notebook/hombre_del_tiempo.ipynb)

---

## 1. El reto

Dada una fotografía, ¿es posible decir qué tiempo hace? Clasificamos imágenes en cuatro condiciones (**nublado, lluvia, sol y amanecer**) comparando descriptores clásicos con embeddings de redes preentrenadas, anotando de forma automática con CLIP *zero-shot* y usando un índice FAISS para resolver la clasificación. Como prueba final, comprobamos si el sistema acierta con **10 fotos propias hechas con el móvil**, fuera del dataset.

## 2. Solución

| Bloque | Qué hacemos |
|---|---|
| **A. Características** | Descriptores clásicos: momentos e histogramas de color en HSV y Lab y entropía local. Embeddings profundos preentrenados: CLIP ViT-B/32 y DINOv2-small (o ResNet50). Tabla comparativa y visualización t-SNE/UMAP. |
| **B. Anotación semántica** | Etiquetado zero-shot con CLIP y al menos 2 versiones de prompts (p. ej. `"a rainy street"`). Se evalúan con precisión, matriz de confusión y 5 fallos comentados. Se prueban también clases nuevas sin reentrenar (niebla, nieve). |
| **C. Índice y reto** | Clasificación de las 4 clases por kNN con FAISS, con validación cruzada. Se mide la métrica del reto y el tiempo medio de consulta frente a la búsqueda exhaustiva. |
| **D. Demo** | Vídeo de 1–2 min con el funcionamiento en Colab. |

No se entrena ninguna red desde cero: los modelos preentrenados se usan solo como extractores.

## 3. Dataset

- **Nombre:** Multi-class Weather Dataset
- **Fuente original:** Kaggle, [pratik2901/multiclass-weather-dataset](https://www.kaggle.com/datasets/pratik2901/multiclass-weather-dataset)
- **Contenido:** 1.125 fotografías en 4 clases (nublado, lluvia, sol, amanecer). Se usa el dataset completo.
- **Licencia:** la indicada por el autor en Kaggle (LICENCIA_A_CONFIRMAR).
- **Descarga (OneDrive):** [ENLACE_ONEDRIVE](ENLACE_ONEDRIVE) · Tamaño: XX MB

El cuaderno descarga y descomprime los datos automáticamente desde OneDrive en su primera celda, sin pasos manuales. Las imágenes y los ficheros pesados (como los embeddings `.npy`) no están en este repositorio, sino en OneDrive.

## 4. Resultados principales

> Rellenar con los resultados finales (validación cruzada estratificada, semilla fija).

| Descriptor / modelo | Exactitud (CV) | F1 macro | Tiempo medio de consulta |
|---|---|---|---|
| Color HSV (histograma + momentos) | – | – | – |
| Color Lab (histograma + momentos) | – | – | – |
| Entropía local | – | – | – |
| CLIP ViT-B/32 (embedding + kNN) | – | – | – |
| DINOv2-small (embedding + kNN) | – | – | – |
| CLIP zero-shot (prompt v1) | – | – | – |
| CLIP zero-shot (prompt v2) | – | – | – |

**Índice:** FAISS frente a búsqueda exhaustiva: – ms/consulta frente a – ms/consulta.
**Reto de las fotos del móvil:** X/10 aciertos (detalle en [`fotos_movil/`](fotos_movil/)).
**Conclusión:** qué descriptor gana y por qué (completar).

## 5. Cómo reproducir los resultados

1. Abrir el cuaderno con el botón **Open in Colab** de arriba.
2. Seleccionar un entorno con GPU (*Entorno de ejecución → Cambiar tipo de entorno → GPU T4*).
3. Ejecutar todas las celdas de principio a fin (*Entorno de ejecución → Ejecutar todas*). La primera celda instala las dependencias y descarga el dataset desde OneDrive.

Las semillas están fijadas, de modo que los resultados son reproducibles. Las librerías usadas están en [`requirements.txt`](requirements.txt).

## 6. Estructura del repositorio

```
.
├── README.md
├── requirements.txt
├── notebook/        # cuaderno principal de Colab (.ipynb), con las salidas visibles
├── docs/            # guía del proyecto, memoria y diapositivas en PDF
├── demo/            # enlace al vídeo de la demo
└── fotos_movil/     # las 10 fotos propias del reto y sus predicciones
```

## 7. Entregables

- Cuaderno de Colab ejecutable de principio a fin: [`notebook/`](notebook/)
- Diapositivas (PDF): [`docs/`](docs/)
- Vídeo de la demo: [`demo/`](demo/)

## 8. Referencias

- Dataset: Multi-class Weather Dataset (Kaggle), pratik2901.
- Radford et al., *Learning Transferable Visual Models From Natural Language Supervision* (CLIP), 2021.
- Oquab et al., *DINOv2: Learning Robust Visual Features without Supervision*, 2023.
- Johnson et al., *Billion-scale similarity search with GPUs* (FAISS), 2019.
