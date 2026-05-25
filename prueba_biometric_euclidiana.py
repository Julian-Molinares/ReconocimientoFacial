"""
PRUEBA BÁSICA DEL MODELO — DISTANCIA EUCLIDIANA
biometric-ai-lab/Face_Recognition

Instrucciones:
1. Coloca dos fotos de la misma persona: persona_A_foto1.jpg, persona_A_foto2.jpg
2. Coloca una foto de otra persona:       persona_B_foto1.jpg
3. Ejecuta: python prueba_biometric_euclidiana.py

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
print("  Métrica: DISTANCIA EUCLIDIANA (implementación manual)")
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
        return embedding   # sin normalizar L2 para distancia euclidiana


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

checkpoint = torch.load(model_path, map_location="cpu")

if "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
    print("    ✅ Usando clave 'model_state_dict'")
    if "val_accuracy" in checkpoint:
        print(f"    Val accuracy reportada: {checkpoint['val_accuracy']}")
elif "model" in checkpoint:
    state_dict = checkpoint["model"]
else:
    state_dict = checkpoint

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
    return transform(img).unsqueeze(0)


# ─────────────────────────────────────────────
# 4. EXTRACCIÓN DE EMBEDDINGS
# ─────────────────────────────────────────────
imagenes = {
    "Persona A - foto 1": "persona_A_foto15.jpg",
    "Persona A - foto 2": "persona_A_foto16.jpg",
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
# 6. DISTANCIA EUCLIDIANA — IMPLEMENTACIÓN MANUAL
#    Fórmula: d(A,B) = sqrt( sum( (Ai - Bi)^2 ) )
#    Rango: 0 a inf  —  0 = idénticos, mayor = más distintos
# ─────────────────────────────────────────────
def distancia_euclidiana(a: torch.Tensor, b: torch.Tensor) -> float:
    """
    Distancia Euclidiana implementada manualmente con NumPy.
    Fórmula: d(A,B) = sqrt( sum_i( (A_i - B_i)^2 ) )
    Rango: 0 a inf  (0 = idénticos, mayor = más distintos)
    """
    a = a.numpy().flatten()
    b = b.numpy().flatten()
    diferencia = a - b
    return float(np.sqrt(np.sum(diferencia ** 2)))


print("\n[4] Comparación vectorial (distancia euclidiana manual):")
print("-" * 50)
print("    NOTA: valores MENORES = personas MÁS parecidas")
print("-" * 50)

eA1 = embeddings["Persona A - foto 1"]
eA2 = embeddings["Persona A - foto 2"]
eB1 = embeddings["Persona B - foto 1"]

dist_AA = distancia_euclidiana(eA1, eA2)
dist_AB = distancia_euclidiana(eA1, eB1)

print(f"    Persona A foto1 vs Persona A foto2  (MISMA)    : {dist_AA:.4f}")
print(f"    Persona A foto1 vs Persona B foto1  (DISTINTA)  : {dist_AB:.4f}")
print(f"    Diferencia                                       : {dist_AB - dist_AA:.4f}")
print("-" * 50)


# ─────────────────────────────────────────────
# 7. EXPERIMENTO DE UMBRALES
#    Se generan automáticamente basados en los
#    valores reales obtenidos del modelo
# ─────────────────────────────────────────────

# Umbral óptimo = punto medio entre ambas distancias
umbral_optimo = (dist_AA + dist_AB) / 2

# Generar 5 umbrales alrededor de los valores reales
paso = (dist_AB - dist_AA) / 4
umbrales = [
    round(dist_AA + paso * 1, 4),
    round(dist_AA + paso * 2, 4),   # ← umbral óptimo (punto medio)
    round(dist_AA + paso * 3, 4),
    round(dist_AA * 0.9, 4),        # umbral muy estricto (bajo la dist AA)
    round(dist_AB * 1.1, 4),        # umbral muy permisivo (sobre la dist AB)
]

print(f"\n[5] Experimento de umbrales (distancia euclidiana):")
print(f"    {'Umbral':<10} {'A vs A (misma)':<22} {'A vs B (distinta)':<22} {'Correcto?'}")
print("    " + "-" * 60)

for umbral in umbrales:
    # Misma persona: distancia debe ser MENOR al umbral
    aa_ok  = dist_AA < umbral
    # Distinta persona: distancia debe ser MAYOR O IGUAL al umbral
    ab_ok  = dist_AB >= umbral

    aa_str = "✅ reconocida"   if aa_ok  else "❌ rechazada"
    ab_str = "✅ extraña"      if ab_ok  else "❌ falso positivo"
    res    = "✅" if (aa_ok and ab_ok) else "❌"

    print(f"    {umbral:<10} {aa_str:<22} {ab_str:<22} {res}")

print(f"\n    Valores obtenidos:")
print(f"      dist(A,A) = {dist_AA:.4f}  ← debe ser MENOR al umbral elegido")
print(f"      dist(A,B) = {dist_AB:.4f}  ← debe ser MAYOR al umbral elegido")
print(f"\n    Umbral óptimo sugerido: {umbral_optimo:.4f}")

if dist_AA < dist_AB:
    print("\n    ✅ El modelo distingue entre personas correctamente.")
    print("    ✅ Vector de 512 dims extraído sin usar el wrapper.")
else:
    print("\n    ⚠️  Las distancias son muy similares.")
    print("    Prueba con fotos más distintas entre personas.")

print("\n" + "=" * 60)
print("  PRUEBA COMPLETADA")
print("=" * 60)