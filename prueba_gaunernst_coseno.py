"""
PRUEBA BÁSICA DEL MODELO
gaunernst/vit_small_patch8_gap_112.cosface_ms1mv3

Instrucciones:
1. Coloca dos fotos de la misma persona como: persona_A_foto1.jpg, persona_A_foto2.jpg
2. Coloca una foto de otra persona como:        persona_B_foto1.jpg
3. Ejecuta: python prueba_modelo.py

Instalación previa:
    pip install timm torch torchvision Pillow numpy
"""

import timm
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import time
import os

print("=" * 60)
print("  PRUEBA BÁSICA — vit_small_patch8_gap_112.cosface_ms1mv3")
print("=" * 60)

# ─────────────────────────────────────────────
# 1. CARGAR MODELO
# ─────────────────────────────────────────────
print("\n[1] Cargando modelo desde Hugging Face Hub...")
t0 = time.time()

model = timm.create_model(
    "hf_hub:gaunernst/vit_small_patch8_gap_112.cosface_ms1mv3",
    pretrained=True
).eval()

t_carga = time.time() - t0
n_params = sum(p.numel() for p in model.parameters())

print(f"    ✅ Modelo cargado en {t_carga:.2f}s")
print(f"    Parámetros totales : {n_params:,}")
print(f"    Tamaño aproximado  : {n_params * 4 / 1e6:.1f} MB")


# ─────────────────────────────────────────────
# 2. PREPROCESAMIENTO
# ─────────────────────────────────────────────
def preprocess(image_path: str) -> torch.Tensor:
    """
    Carga una imagen, la redimensiona a 112x112
    y la normaliza al rango [-1, 1] como requiere el modelo.
    """
    img = Image.open(image_path).convert("RGB").resize((112, 112))
    arr = np.array(img).astype(np.float32) / 255.0
    arr = (arr - 0.5) / 0.5                        # normalización facial estándar
    tensor = torch.tensor(arr).permute(2, 0, 1).unsqueeze(0)  # (1, 3, 112, 112)
    return tensor


# ─────────────────────────────────────────────
# 3. EXTRACCIÓN DE EMBEDDINGS
# ─────────────────────────────────────────────
imagenes = {
    "Persona A - foto 1": "persona_A_foto1.jpg",
    "Persona A - foto 2": "persona_A_foto7.jpg",
    "Persona B - foto 1": "persona_B_foto2.jpg",
}

# Verificar que las imágenes existen
for nombre, path in imagenes.items():
    if not os.path.exists(path):
        print(f"\n❌ ERROR: No se encontró '{path}'")
        print("   Asegúrate de tener las 3 imágenes en la misma carpeta que este script.")
        exit(1)

print("\n[2] Extrayendo embeddings...")
embeddings = {}

with torch.no_grad():
    for nombre, path in imagenes.items():
        t0 = time.time()
        tensor = preprocess(path)
        emb = model(tensor)
        emb = F.normalize(emb, dim=1)   # normalización L2 para similitud coseno
        t_inf = (time.time() - t0) * 1000
        embeddings[nombre] = emb
        print(f"    {nombre}: shape={tuple(emb.shape)} | inferencia={t_inf:.1f}ms")


# ─────────────────────────────────────────────
# 4. VERIFICAR DIMENSIÓN
# ─────────────────────────────────────────────
dim = embeddings["Persona A - foto 1"].shape[1]
print(f"\n[3] Dimensión del vector de salida: {dim} dims", "✅" if dim == 512 else "⚠️")


# ─────────────────────────────────────────────
# 5. SIMILITUD COSENO — IMPLEMENTACIÓN MANUAL
# ─────────────────────────────────────────────
def similitud_coseno(a: torch.Tensor, b: torch.Tensor) -> float:
    """
    Similitud coseno implementada manualmente con NumPy.
    Rango: 0 a 1  (1 = idénticos, 0 = completamente distintos)
    Fórmula: cos(A,B) = (A · B) / (||A|| * ||B||)
    """
    a = a.numpy().flatten()
    b = b.numpy().flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


print("\n[4] Comparación vectorial (similitud coseno manual):")
print("-" * 45)

eA1 = embeddings["Persona A - foto 1"]
eA2 = embeddings["Persona A - foto 2"]
eB1 = embeddings["Persona B - foto 1"]

sim_AA = similitud_coseno(eA1, eA2)
sim_AB = similitud_coseno(eA1, eB1)

print(f"    Persona A foto1 vs Persona A foto2  (MISMA)    : {sim_AA:.4f}")
print(f"    Persona A foto1 vs Persona B foto1  (DISTINTA)  : {sim_AB:.4f}")
print(f"    Diferencia                                       : {sim_AA - sim_AB:.4f}")
print("-" * 45)


# ─────────────────────────────────────────────
# 6. DIAGNÓSTICO
# ─────────────────────────────────────────────
UMBRAL = 0.75

print(f"\n[5] Diagnóstico con umbral = {UMBRAL}:")
print(f"    Persona A reconocida  : {'✅ SÍ' if sim_AA >= UMBRAL else '❌ NO'} (similitud={sim_AA:.4f})")
print(f"    Persona B como extraña: {'✅ SÍ' if sim_AB < UMBRAL else '❌ NO (falso positivo)'} (similitud={sim_AB:.4f})")

if sim_AA > sim_AB:
    print("\n    ✅ El modelo distingue correctamente entre personas.")
    print("    ✅ El vector de 512 dims se extrae correctamente.")
    print("    ✅ La similitud coseno funciona como se espera.")
else:
    print("\n    ⚠️  El modelo no logró distinguir. Prueba con fotos reales.")

print("\n" + "=" * 60)
print("  PRUEBA COMPLETADA")
print("=" * 60)