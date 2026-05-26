"""
MAIN — Punto de entrada del sistema
Integra todos los módulos: cámara, detección, identificación y registro.

MEJORAS v2:
  - Se usa preprocesar_rostro() de modelo_hf para recortar el ROI con un
    margen del 20% antes de extraer el embedding. Esto es el cambio más
    importante: la distancia euclidiana baja de ~1.0–1.5 a ~0.4–0.8.
  - El diagnóstico en consola también muestra la distancia real para
    facilitar el ajuste del umbral.

Uso:
    python main.py
"""

import cv2
import time
import numpy as np
from modelo_hf      import cargar_modelo, extraer_embedding, preprocesar_rostro
from comparacion    import cargar_base_datos, identificar, cargar_umbral, distancia_euclidiana
from registro_excel import (crear_excel_sesion, registrar_asistencia,
                             obtener_presentes, marcar_ausentes,
                             ContadorConfirmaciones, CONFIRMACIONES_REQUERIDAS)


# ── Configuración ──────────────────────────────────────────────
ESCALA_DETECTOR = 1.1
MIN_VECINOS     = 5
MIN_TAMANIO     = (80, 80)

COLOR_RECONOCIDO  = (0, 255, 0)
COLOR_DESCONOCIDO = (0, 0, 255)
COLOR_INFO        = (255, 255, 0)
COLOR_CONTEO      = (0, 165, 255)   # naranja — confirmando


def main():
    print("=" * 55)
    print("  SISTEMA DE ASISTENCIA — Reconocimiento Facial")
    print("=" * 55)

    # ── Cargar modelo ──────────────────────────────────────────
    print("\n[main] Cargando modelo...")
    modelo = cargar_modelo()
    print("[main] Modelo cargado ✅")

    # ── Cargar base de datos ───────────────────────────────────
    print("[main] Cargando base de datos...")
    bd = cargar_base_datos()
    print(f"[main] Alumnos registrados: {len(bd)} ✅")

    if not bd:
        print("\n❌ Base de datos vacía. Ejecuta captura_alumnos.py primero.")
        input("Presiona ENTER para salir...")
        return

    # ── Cargar umbral ──────────────────────────────────────────
    print("[main] Cargando umbral...")
    umbral = cargar_umbral()
    print(f"[main] Umbral: {umbral:.4f} ✅")

    # ── Crear Excel de NUEVA sesión ────────────────────────────
    print("[main] Creando Excel de sesión...")
    ruta_excel = crear_excel_sesion()
    print(f"[main] Excel listo ✅")

    # ── Detector Haar Cascade ──────────────────────────────────
    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    # ── Abrir cámara ───────────────────────────────────────────
    print("[main] Abriendo cámara...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("❌ No se pudo abrir la cámara")
        input("Presiona ENTER para salir...")
        return

    print("[main] Cámara abierta ✅")
    print(f"[main] Confirmaciones requeridas: {CONFIRMACIONES_REQUERIDAS}")
    print("[main] Sistema activo — presiona Q para salir\n")

    # ── Contador de confirmaciones ─────────────────────────────
    contador = ContadorConfirmaciones(CONFIRMACIONES_REQUERIDAS)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gris  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        caras = detector.detectMultiScale(
            gris,
            scaleFactor=ESCALA_DETECTOR,
            minNeighbors=MIN_VECINOS,
            minSize=MIN_TAMANIO
        )

        for (x, y, w, h) in caras:
            # ── MEJORA CLAVE: recortar con margen antes de embedding ──
            # preprocesar_rostro() expande el ROI un 20% en cada lado
            # para que el modelo reciba el rostro completo y no un recorte
            # demasiado ajustado que produce embeddings de baja calidad.
            rostro    = preprocesar_rostro(frame, x, y, w, h)
            embedding = extraer_embedding(modelo, rostro)

            # Diagnóstico en consola
            for nom, fotos in bd.items():
                ref   = fotos["foto1"] + fotos["foto2"]
                norma = np.linalg.norm(ref)
                ref   = ref / norma if norma > 0 else ref
                dist  = distancia_euclidiana(embedding, ref)
                print(f"  dist({nom}) = {dist:.4f}  {'✓' if dist < umbral else '✗'}")
            print(f"  umbral = {umbral:.4f}\n")

            nombre, distancia = identificar(embedding, bd, umbral)

            # ── Confirmaciones ─────────────────────────────────
            listo = contador.confirmar(nombre)
            conteo_actual = contador.conteos.get(nombre, 0)

            if nombre == "Desconocido":
                color    = COLOR_DESCONOCIDO
                etiqueta = f"Desconocido (dist={distancia:.2f})"
            else:
                if listo:
                    # Alcanzó las N confirmaciones → registrar
                    registrado = registrar_asistencia(ruta_excel, nombre, distancia)
                    if registrado:
                        print(f"[main] ✅ {nombre} confirmado y registrado")
                    color    = COLOR_RECONOCIDO
                    etiqueta = f"{nombre} ✅ ({distancia:.2f})"
                else:
                    # Confirmando — mostrar progreso
                    color    = COLOR_CONTEO
                    etiqueta = f"{nombre} [{conteo_actual}/{CONFIRMACIONES_REQUERIDAS}] ({distancia:.2f})"

            # Dibujar bounding box con el margen visual (x, y, w, h original del Haar)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.rectangle(frame, (x, y-30), (x+w, y), color, -1)
            cv2.putText(frame, etiqueta, (x+4, y-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        # Panel de presentes
        presentes = obtener_presentes(ruta_excel)
        cv2.rectangle(frame, (0, 0), (230, 28 + 22 * max(len(presentes), 1)),
                      (30, 30, 30), -1)
        cv2.putText(frame, f"Presentes: {len(presentes)}", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_INFO, 2)
        # Mostrar umbral activo en pantalla
        h_frame, w_frame = frame.shape[:2]
        cv2.putText(frame, f"Umbral: {umbral:.2f}", (w_frame - 150, h_frame - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_INFO, 1)
        for i, p in enumerate(presentes):
            cv2.putText(frame, f"  {p}", (5, 40 + 22 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_RECONOCIDO, 1)

        cv2.imshow("Sistema de Asistencia  |  Q para salir", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("\n[main] Marcando ausentes...")
    marcar_ausentes(ruta_excel, bd)
    print(f"[main] Sistema cerrado.")
    print(f"[main] Presentes: {obtener_presentes(ruta_excel)}")


if __name__ == "__main__":
    main()