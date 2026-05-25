"""
MÓDULO 02 — Extracción de Embeddings
Modelo: gaunernst/vit_small_patch8_gap_112.cosface_ms1mv3
Librería: timm (Hugging Face)

Responsabilidad: cargar el modelo y extraer el vector de 512 dims
a partir de una imagen de rostro recortada.
"""

import timm
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np


# ── Constantes del modelo ──────────────────────────────────────
MODELO_ID  = "hf_hub:gaunernst/vit_small_patch8_gap_112.cosface_ms1mv3"
INPUT_SIZE = 112       # el modelo requiere exactamente 112x112 px
EMBED_DIM  = 512       # dimensión del vector de salida


def cargar_modelo():
    """
    Descarga (primera vez) y carga el modelo desde Hugging Face Hub.
    Retorna el modelo en modo evaluación listo para inferencia.
    """
    print(f"[modelo_hf] Cargando modelo: {MODELO_ID}")
    modelo = timm.create_model(MODELO_ID, pretrained=True).eval()
    print(f"[modelo_hf] Modelo cargado — output: {EMBED_DIM} dims")
    return modelo


def preprocesar_imagen(imagen_bgr: np.ndarray) -> torch.Tensor:
    """
    Convierte un recorte de rostro BGR (OpenCV) al tensor que espera el modelo.

    Pasos:
      1. BGR → RGB
      2. Redimensionar a 112x112
      3. Normalizar al rango [-1, 1]  (requerido por CosFace/MS1MV3)
      4. Convertir a tensor (1, 3, 112, 112)

    Args:
        imagen_bgr: array numpy HxWx3 en formato BGR (salida de OpenCV)

    Returns:
        tensor de forma (1, 3, 112, 112)
    """
    imagen_rgb = imagen_bgr[:, :, ::-1].copy()          # BGR → RGB
    pil_img    = Image.fromarray(imagen_rgb).resize(
        (INPUT_SIZE, INPUT_SIZE), Image.BILINEAR
    )
    arr    = np.array(pil_img).astype(np.float32) / 255.0
    arr    = (arr - 0.5) / 0.5                           # normalización [-1, 1]
    tensor = torch.tensor(arr).permute(2, 0, 1).unsqueeze(0)  # (1, 3, 112, 112)
    return tensor


def extraer_embedding(modelo, imagen_bgr: np.ndarray) -> np.ndarray:
    """
    Extrae el vector de características (embedding) de un rostro.

    Args:
        modelo:      modelo cargado con cargar_modelo()
        imagen_bgr:  recorte del rostro en formato BGR (numpy array)

    Returns:
        vector numpy de forma (512,) normalizado L2
    """
    tensor = preprocesar_imagen(imagen_bgr)
    with torch.no_grad():
        embedding = modelo(tensor)                      # (1, 512)
        embedding = F.normalize(embedding, dim=1)       # normalización L2
    return embedding.numpy().flatten()                  # (512,)
