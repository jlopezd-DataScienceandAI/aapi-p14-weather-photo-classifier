# El hombre del tiempo: ¿qué tiempo hace en una foto?

**Algoritmos y Arquitecturas para Procesado de Imágenes** · Grado en Ciencia de Datos e Inteligencia Artificial · Curso 2026–2027
**Proyecto 14 · Tema 3: Imágenes como datos visuales**

**Autores:** NOMBRE_1 y NOMBRE_2

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jlopezd-DataScienceandAI/aapi-p14-weather-photo-classifier/blob/main/notebook/weather_classifier.ipynb)

---

## 1. El reto

Dada una fotografía, ¿es posible decir qué tiempo hace? Clasificamos imágenes en cuatro condiciones (**nublado, lluvia, sol y amanecer**) comparando descriptores clásicos con embeddings de redes preentrenadas, anotando de forma automática con CLIP *zero-shot* y usando un índice FAISS para resolver la clasificación. Como prueba final, comprobamos si el sistema acierta con **10 fotos propias hechas con el móvil**, fuera del dataset.

## 2. Solución

| Bloque | Qué hacemos |
|---|---|
| **A. Características** | Cinco descriptores clásicos (histogramas y momentos de color en HSV y Lab, entropía local, textura con GLCM + LBP y un conjunto combinado) frente a tres embeddings preentrenados: CLIP ViT-B/32, DINOv2-small y ResNet50. Tabla comparativa con validación cruzada y visualización t-SNE de cuatro espacios de características (los clásicos combinados, CLIP, DINOv2-small y ResNet50). |
| **B. Anotación semántica** | Etiquetado *zero-shot* con CLIP y cuatro versiones de prompts (el nombre de la clase, una plantilla, una palabra natural y cinco escenas por clase, como `"a rainy street"`). Se evalúan con exactitud, F1, intervalo bootstrap y matrices de confusión, con 5 fallos comentados con hipótesis y prueba. Se añaden dos clases nuevas sin reentrenar (niebla y nieve) y se estudia si CLIP distingue amaneceres de atardeceres. |
| **C. Índice y reto** | Clasificación de las 4 clases por kNN sobre un índice FAISS, con validación cruzada anidada para elegir `k` y la votación. Tiempo medio de consulta de FAISS frente a la búsqueda exhaustiva (NumPy y scikit-learn), y simulación con 100 000 vectores para ver cuándo un índice aproximado compensa. |
| **Reto del móvil** | Diez fotos propias, fuera del dataset, clasificadas con cinco sistemas (kNN con FAISS, regresión logística y *zero-shot*), con análisis de los fallos y de cinco fotos extra que no encajan en las cuatro clases. |
| **D. Demo** | Vídeo de 1–2 min con el funcionamiento en Colab. |

No se entrena ninguna red desde cero: los modelos preentrenados se usan solo como extractores.

## 3. Dataset

- **Nombre:** Multi-class Weather Dataset
- **Fuente original:** Kaggle, [pratik2901/multiclass-weather-dataset](https://www.kaggle.com/datasets/pratik2901/multiclass-weather-dataset)
- **Contenido:** 1.125 fotografías en 4 clases: Cloudy (300), Rain (215), Shine (253) y Sunrise (357). Se usa el dataset completo.
- **Licencia:** la indicada por el autor en la página de Kaggle (enlace arriba).
- **Descarga (OneDrive):** [multiclass-weather-dataset.zip](https://upm365-my.sharepoint.com/:u:/g/personal/daniel_barahonamor_alumnos_upm_es/IQCw1c6SsmcrQ7rzzOwaT8GKAeJezfnfHky_qXpRPKh22ns) · Tamaño: 95,8 MB

El cuaderno descarga y descomprime los datos automáticamente desde OneDrive en su primera celda, sin pasos manuales. Las imágenes y los ficheros pesados (como los embeddings `.npy`) no están en este repositorio, sino en OneDrive. Las 15 fotos del móvil sí están aquí, en [`fotos_movil/`](fotos_movil/), porque pesan poco.

## 4. Resultados principales

Todos los resultados usan validación cruzada de 5 particiones agrupada por casi-duplicados (el dataset tiene 121 pares) y semilla fija.

**Descriptores y clasificadores (bloque A).** Los kNN son de scikit-learn con `k = 5`. Los tiempos son los milisegundos por imagen para calcular el descriptor: los de los clásicos son de CPU (se calculan todos a la vez) y los de las redes, de la GPU T4 de Colab.

| Descriptor | Exactitud kNN | F1 macro kNN | Exactitud regresión logística | F1 macro regresión logística | ms por imagen |
|---|---|---|---|---|---|
| **CLIP ViT-B/32** | 0,950 | 0,951 | **0,980** | **0,980** | 7,4 |
| ResNet50 | 0,948 | 0,947 | 0,969 | 0,969 | 5,7 |
| DINOv2-small | 0,927 | 0,929 | 0,962 | 0,962 | 6,4 |
| Clásicos combinados | 0,880 | 0,873 | 0,929 | 0,927 | 57,5 |
| Color HSV (histograma + momentos) | 0,819 | 0,798 | 0,823 | 0,810 | 57,5 |
| Color Lab (histograma + momentos) | 0,805 | 0,783 | 0,802 | 0,785 | 57,5 |
| Textura (GLCM + LBP) | 0,666 | 0,665 | 0,714 | 0,721 | 57,5 |
| Entropía local | 0,446 | 0,447 | 0,472 | 0,465 | 57,5 |

**Anotación *zero-shot* con CLIP (bloque B).** No usa ninguna etiqueta: se evalúa contra las etiquetas reales de las 1.125 fotos. El intervalo es el bootstrap del 95 % del F1 macro.

| Versión de prompt | Exactitud | F1 macro | Intervalo del F1 |
|---|---|---|---|
| v1 · el nombre de la clase (`Shine`) | 0,832 | 0,805 | 0,781–0,828 |
| v2 · plantilla (`a photo of shine`) | 0,687 | 0,628 | 0,599–0,657 |
| v3 · palabra natural (`a photo of a sunny day`) | 0,948 | 0,948 | 0,935–0,961 |
| **v4 · cinco escenas por clase** | **0,962** | **0,962** | 0,950–0,973 |

**kNN sobre un índice FAISS (bloque C).** Validación cruzada anidada: la `k` y la votación se eligen dentro de cada partición de entrenamiento.

| Descriptor | Exactitud | F1 macro | Intervalo del F1 |
|---|---|---|---|
| **ResNet50 + FAISS** | 0,953 | 0,952 | 0,939–0,963 |
| CLIP ViT-B/32 + FAISS | 0,950 | 0,951 | 0,938–0,963 |
| DINOv2-small + FAISS | 0,932 | 0,933 | 0,919–0,946 |
| Clásicos combinados + FAISS | 0,890 | 0,884 | 0,865–0,901 |

**Índice frente a búsqueda exhaustiva** (1.125 fotos, tiempo medio por consulta): NumPy 0,0125 ms · scikit-learn 0,0164 ms · FAISS Flat 0,0188 ms · FAISS IVF 0,0189 ms · FAISS HNSW 0,0548 ms. Con tan pocas fotos el índice no acelera nada, y el coste real es calcular el embedding (7,4 ms). Con 100 000 vectores sintéticos, HNSW es 12 veces más rápido que FAISS exacto y 22 veces más que NumPy, con un *recall*@10 de 0,996.

**Reto de las fotos del móvil** (10 fotos fuera del dataset, detalle en [`fotos_movil/`](fotos_movil/)):

| Sistema | Aciertos |
|---|---|
| **CLIP *zero-shot* (v4)** | **10/10** |
| CLIP + regresión logística | 8/10 |
| ResNet50 + regresión logística | 7/10 |
| CLIP + kNN con FAISS | 6/10 |
| ResNet50 + kNN con FAISS (el clasificador final del bloque C) | 5/10 |

Con 10 fotos los intervalos son muy anchos (el del 10/10 va de 0,69 a 1,00), así que las diferencias entre sistemas son orientativas.

**Conclusión: CLIP es el mejor descriptor y su *zero-shot* es el que mejor generaliza.** Dentro del dataset, CLIP gana con kNN y con regresión logística (0,980), por delante de ResNet50, DINOv2-small y los descriptores clásicos, que se confunden sobre todo entre nublado y lluvia. Fuera del dataset, el *zero-shot* con cinco escenas por clase acierta las 10 fotos del móvil, mientras que el kNN con ResNet50, que había ganado en la validación cruzada, solo acierta 5: mira la escena entera (playas, edificios) y no el cielo. El índice FAISS no decide el resultado con un dataset de este tamaño.

## 5. Cómo reproducir los resultados

1. Abrir el cuaderno con el botón **Open in Colab** de arriba.
2. Seleccionar un entorno con GPU (*Entorno de ejecución → Cambiar tipo de entorno → GPU T4*) antes de ejecutar nada.
3. Ejecutar todas las celdas de principio a fin (*Entorno de ejecución → Ejecutar todas*). La primera celda instala las dependencias y descarga el dataset desde OneDrive, y la sección del reto descarga las fotos del móvil de este repositorio, que por eso tiene que ser público.

Las semillas están fijadas, de modo que los resultados son reproducibles; los tiempos varían según la máquina. Las librerías usadas están en [`requirements.txt`](requirements.txt).

## 6. Estructura del repositorio

```
.
├── README.md
├── requirements.txt
├── notebook/        # cuaderno principal de Colab (.ipynb), con las salidas visibles
├── docs/            # guía del proyecto y diapositivas en PDF
├── demo/            # enlace al vídeo de la demo
└── fotos_movil/     # las 15 fotos propias: 10 del reto y 5 extra
```

## 7. Entregables

- Cuaderno de Colab ejecutable de principio a fin: [`notebook/weather_classifier.ipynb`](notebook/weather_classifier.ipynb)
- Diapositivas (PDF): [`docs/`](docs/)
- Vídeo de la demo: [`demo/`](demo/)

## 8. Referencias

- Dataset: Multi-class Weather Dataset (Kaggle), pratik2901.
- Radford et al., *Learning Transferable Visual Models From Natural Language Supervision* (CLIP), 2021.
- Oquab et al., *DINOv2: Learning Robust Visual Features without Supervision*, 2023.
- He et al., *Deep Residual Learning for Image Recognition* (ResNet), 2016.
- Johnson et al., *Billion-scale similarity search with GPUs* (FAISS), 2019.
