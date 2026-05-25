"""
PRUEBA BÁSICA DEL MODELO
biometric-ai-lab/Face_Recognition

Instrucciones:
1. Coloca dos fotos de la misma persona: persona_A_foto1.jpg, persona_A_foto2.jpg
2. Coloca una foto de otra persona:       persona_B_foto1.jpg
3. Ejecuta: python prueba_biometric_coseno.py

Instalación previa:
    pip install torch torchvision Pillow numpy huggingface_hub opencv-python
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from huggingface_hub import hf_hub_download
from PIL import Image
import numpy as np
import time
import os

print("=" * 60)
print("  PRUEBA BÁSICA — biometric-ai-lab/Face_Recognition")
print("  (Acceso directo al backbone, sin wrapper)")
print("=" * 60)


# ─────────────────────────────────────────────
# 1. ARQUITECTURA DEL MODELO
# ─────────────────────────────────────────────
class FaceRecognitionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.wide_resnet101_2(weights=None)
        self.backbone.fc = nn.Identity()
        self.embed = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        features = self.backbone(x)
        embedding = self.embed(features)
        return F.normalize(embedding, p=2, dim=1)


# ─────────────────────────────────────────────
# 2. DESCARGAR Y CARGAR PESOS
# ─────────────────────────────────────────────
print("\n[1] Descargando pesos desde Hugging Face Hub...")
t0 = time.time()

try:
    model_path = hf_hub_download(
        repo_id="biometric-ai-lab/Face_Recognition",
        filename="pytorch_model.bin"
    )
    print(f"    Archivo: {model_path}")
except Exception as e:
    print(f"    ❌ Error al descargar: {e}")
    exit(1)

# Inspeccionar claves del checkpoint
print("\n    Inspeccionando estructura del checkpoint...")
checkpoint = torch.load(model_path, map_location="cpu")

if isinstance(checkpoint, dict):
    print(f"    Claves en el checkpoint: {list(checkpoint.keys())}")
else:
    print(f"    El checkpoint es de tipo: {type(checkpoint)}")

# Extraer el state_dict correcto usando 'model_state_dict'
if "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
    print("    ✅ Usando clave 'model_state_dict'")
    # Mostrar info extra del checkpoint si existe
    if "val_accuracy" in checkpoint:
        print(f"    Val accuracy reportada: {checkpoint['val_accuracy']}")
    if "config" in checkpoint:
        print(f"    Config: {checkpoint['config']}")
elif "model" in checkpoint:
    state_dict = checkpoint["model"]
    print("    ✅ Usando clave 'model'")
elif isinstance(checkpoint, dict) and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
    state_dict = checkpoint
    print("    ✅ Checkpoint es state_dict directo")
else:
    print("    ❌ Estructura desconocida. Claves encontradas:")
    for k, v in checkpoint.items():
        print(f"       - {k}: {type(v)}")
    exit(1)

model = FaceRecognitionModel()
model.load_state_dict(state_dict)
model.eval()

t_carga = time.time() - t0
n_params = sum(p.numel() for p in model.parameters())

print(f"\n    ✅ Modelo cargado en {t_carga:.2f}s")
print(f"    Parámetros totales : {n_params:,}")
print(f"    Tamaño aproximado  : {n_params * 4 / 1e6:.1f} MB")


# ─────────────────────────────────────────────
# 3. PREPROCESAMIENTO
# ─────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

def preprocess(image_path: str) -> torch.Tensor:
    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0)      # (1, 3, 224, 224)


# ─────────────────────────────────────────────
# 4. EXTRACCIÓN DE EMBEDDINGS
# ─────────────────────────────────────────────
imagenes = {
    "Persona A - foto 1": "persona_A_foto1.jpg",
    "Persona A - foto 2": "persona_A_foto2.jpg",
    "Persona B - foto 1": "persona_B_foto4.jpg",
}

for nombre, path in imagenes.items():
    if not os.path.exists(path):
        print(f"\n❌ No se encontró '{path}'")
        print("   Coloca las 3 imágenes en la misma carpeta que este script.")
        exit(1)

print("\n[2] Extrayendo embeddings...")
embeddings = {}

with torch.no_grad():
    for nombre, path in imagenes.items():
        t0 = time.time()
        tensor = preprocess(path)
        emb = model(tensor)
        t_inf = (time.time() - t0) * 1000
        embeddings[nombre] = emb
        print(f"    {nombre}: shape={tuple(emb.shape)} | inferencia={t_inf:.1f}ms")


# ─────────────────────────────────────────────
# 5. VERIFICAR DIMENSIÓN
# ─────────────────────────────────────────────
dim = embeddings["Persona A - foto 1"].shape[1]
print(f"\n[3] Dimensión del vector de salida: {dim} dims", "✅" if dim == 512 else "⚠️")


# ─────────────────────────────────────────────
# 6. SIMILITUD COSENO — IMPLEMENTACIÓN MANUAL
# ─────────────────────────────────────────────
def similitud_coseno(a: torch.Tensor, b: torch.Tensor) -> float:
    """
    Similitud coseno implementada manualmente con NumPy.
    Fórmula: cos(A,B) = (A · B) / (||A|| * ||B||)
    Rango: 0 a 1  (1 = idénticos)
    """
    a = a.numpy().flatten()
    b = b.numpy().flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


print("\n[4] Comparación vectorial (similitud coseno manual):")
print("-" * 50)

eA1 = embeddings["Persona A - foto 1"]
eA2 = embeddings["Persona A - foto 2"]
eB1 = embeddings["Persona B - foto 1"]

sim_AA = similitud_coseno(eA1, eA2)
sim_AB = similitud_coseno(eA1, eB1)

print(f"    Persona A foto1 vs Persona A foto2  (MISMA)    : {sim_AA:.4f}")
print(f"    Persona A foto1 vs Persona B foto1  (DISTINTA)  : {sim_AB:.4f}")
print(f"    Diferencia                                       : {sim_AA - sim_AB:.4f}")
print("-" * 50)


# ─────────────────────────────────────────────
# 7. DIAGNÓSTICO
# ─────────────────────────────────────────────
UMBRAL = 0.45

print(f"\n[5] Diagnóstico con umbral = {UMBRAL}:")
print(f"    Persona A reconocida  : {'✅ SÍ' if sim_AA >= UMBRAL else '❌ NO'} (similitud={sim_AA:.4f})")
print(f"    Persona B como extraña: {'✅ SÍ' if sim_AB < UMBRAL else '❌ NO (falso positivo)'} (similitud={sim_AB:.4f})")

if sim_AA > sim_AB:
    print("\n    ✅ El modelo distingue entre personas correctamente.")
    print("    ✅ Vector de 512 dims extraído sin usar el wrapper.")
else:
    print("\n    ⚠️  No logró distinguir. Prueba con fotos más claras.")

print("\n" + "=" * 60)
print("  PRUEBA COMPLETADA")
print("=" * 60)