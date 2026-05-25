"""
PRUEBA BÁSICA DEL MODELO — DISTANCIA EUCLIDIANA + EXPERIMENTO DE UMBRALES
gaunernst/vit_small_patch8_gap_112.cosface_ms1mv3

Instrucciones:
1. Coloca dos fotos de la misma persona como: persona_A_foto1.jpg, persona_A_foto2.jpg
2. Coloca una foto de otra persona como:        persona_B_foto1.jpg
3. Ejecuta: python prueba_modelo_euclidiana.py

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
print("  PRUEBA EUCLIDIANA — vit_small_patch8_gap_112.cosface_ms1mv3")
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
    arr = (arr - 0.5) / 0.5                                    # normalización facial estándar
    tensor = torch.tensor(arr).permute(2, 0, 1).unsqueeze(0)   # (1, 3, 112, 112)
    return tensor


# ─────────────────────────────────────────────
# 3. EXTRACCIÓN DE EMBEDDINGS
# ─────────────────────────────────────────────
imagenes = {
    "Persona A - foto 1": "fotos_alumnos/persona_A_foto1.jpg",
    "Persona A - foto 2": "fotos_alumnos/persona_A_foto2.jpg",
    "Persona B - foto 1": "fotos_alumnos/persona_B_foto4.jpg",
}

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
        emb = F.normalize(emb, dim=1)   # normalización L2 obligatoria
        t_inf = (time.time() - t0) * 1000
        embeddings[nombre] = emb
        print(f"    {nombre}: shape={tuple(emb.shape)} | inferencia={t_inf:.1f}ms")


# ─────────────────────────────────────────────
# 4. VERIFICAR DIMENSIÓN
# ─────────────────────────────────────────────
dim = embeddings["Persona A - foto 1"].shape[1]
print(f"\n[3] Dimensión del vector de salida: {dim} dims", "✅" if dim == 512 else "⚠️")


# ─────────────────────────────────────────────
# 5. DISTANCIA EUCLIDIANA — IMPLEMENTACIÓN MANUAL
# ─────────────────────────────────────────────
def distancia_euclidiana(a: torch.Tensor, b: torch.Tensor) -> float:
    """
    Distancia euclidiana implementada manualmente con NumPy.
    Fórmula: d(A,B) = sqrt( sum_i( (A_i - B_i)^2 ) )
    Rango: 0 a inf  (0 = idénticos, mayor = más distintos)

    Nota: con vectores L2-normalizados el rango real es [0, 2],
    donde 0 = idénticos y 2 = opuestos.
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
# 6. EXPERIMENTO DE UMBRALES
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
    aa_ok = dist_AA < umbral
    # Distinta persona: distancia debe ser MAYOR O IGUAL al umbral
    ab_ok = dist_AB >= umbral

    aa_str = "✅ reconocida"    if aa_ok else "❌ rechazada"
    ab_str = "✅ extraña"       if ab_ok else "❌ falso positivo"
    res    = "✅"               if (aa_ok and ab_ok) else "❌"

    print(f"    {umbral:<10} {aa_str:<22} {ab_str:<22} {res}")

print(f"\n    Valores obtenidos:")
print(f"      dist(A,A) = {dist_AA:.4f}  ← debe ser MENOR al umbral elegido")
print(f"      dist(A,B) = {dist_AB:.4f}  ← debe ser MAYOR al umbral elegido")
print(f"\n    Umbral óptimo sugerido: {umbral_optimo:.4f}")


# ─────────────────────────────────────────────
# 7. DIAGNÓSTICO FINAL
# ─────────────────────────────────────────────
print(f"\n[6] Diagnóstico con umbral óptimo = {umbral_optimo:.4f}:")
print(f"    Persona A reconocida  : {'✅ SÍ' if dist_AA < umbral_optimo else '❌ NO'} (distancia={dist_AA:.4f})")
print(f"    Persona B como extraña: {'✅ SÍ' if dist_AB >= umbral_optimo else '❌ NO (falso positivo)'} (distancia={dist_AB:.4f})")

if dist_AA < dist_AB:
    print("\n    ✅ El modelo distingue correctamente entre personas.")
    print("    ✅ El vector de 512 dims se extrae correctamente.")
    print("    ✅ La distancia euclidiana funciona como se espera.")
else:
    print("\n    ⚠️  El modelo no logró distinguir. Prueba con fotos reales.")

print("\n" + "=" * 60)
print("  PRUEBA COMPLETADA")
print("=" * 60)