#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, subprocess, re, csv, json
from pathlib import Path
import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from PIL import Image
from tqdm import tqdm

def dcm_to_png(dcm_path: Path, png_path: Path):
    ds = pydicom.dcmread(str(dcm_path), force=True)
    arr = ds.pixel_array.astype(np.float32)

    # VOI LUT (si existe)
    try:
        arr = apply_voi_lut(arr, ds)
    except Exception:
        pass

    # Rescale Slope/Intercept (si existen)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    inter = float(getattr(ds, "RescaleIntercept", 0.0))
    arr = arr * slope + inter

    # Windowing (si existe)
    def window(img, center, width):
        low, high = center - width / 2.0, center + width / 2.0
        img = np.clip(img, low, high)
        img = (img - low) / (high - low + 1e-5)
        return img

    wc = getattr(ds, "WindowCenter", None)
    ww = getattr(ds, "WindowWidth", None)
    if isinstance(wc, pydicom.multival.MultiValue): wc = float(wc[0])
    if isinstance(ww, pydicom.multival.MultiValue): ww = float(ww[0])

    if wc is not None and ww is not None and ww > 1:
        arr = window(arr, float(wc), float(ww))
    else:
        # Normaliza robusto por percentiles
        p1, p99 = np.percentile(arr, [1,99])
        arr = np.clip(arr, p1, p99)
        arr = (arr - p1) / (p99 - p1 + 1e-5)

    arr = (arr * 255.0).astype(np.uint8)
    Image.fromarray(arr).save(str(png_path))

def analyze_with_medgemma(pyfile: Path, image_path: Path, model: str, device: str, dtype: str, extra_args=None):
    cmd = [
        "python", str(pyfile),
        "--image", str(image_path),
        "--model", model,
        "--device", device
    ]
    if dtype:
        cmd += ["--dtype", dtype]
    if extra_args:
        cmd += extra_args
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out = res.stdout.strip()
    err = res.stderr.strip()
    return out, err, res.returncode

# Traducción muy básica EN->ES para términos frecuentes
TRANSLATE = [
    (r"\bNormal chest radiograph\b", "Radiografía de tórax normal"),
    (r"\bNormal cardiomediastinal silhouette\b", "Silueta cardiomediastínica normal"),
    (r"\bClear lung fields bilaterally\b", "Campos pulmonares limpios bilateralmente"),
    (r"\bNo pleural effusion\b", "Sin derrame pleural"),
    (r"\bNo pneumothorax\b", "Sin neumotórax"),
    (r"\bno acute cardiopulmonary process\b", "sin proceso cardiopulmonar agudo"),
    (r"\bpleural effusion\b", "derrame pleural"),
    (r"\bpneumothorax\b", "neumotórax"),
    (r"\bpneumonia\b", "neumonía"),
    (r"\bconsolidation\b", "consolidación"),
    (r"\bopacity\b", "opacidad"),
    (r"\bnodule\b", "nódulo"),
    (r"\bmass\b", "masa"),
    (r"\bfracture\b", "fractura"),
    (r"\bcardiomegaly\b", "cardiomegalia"),
    (r"\batelectasis\b", "atelectasia"),
    (r"\bedema\b", "edema"),
    (r"\binfiltrate\b", "infiltrado"),
    (r"\blesion\b", "lesión"),
]

POSITIVE_KWS = r"(pneumonia|consolidation|opacity|nodule|mass|fracture|cardiomegaly|atelectasis|edema|infiltrate|effusion|pneumothorax|lesion|collapse|airspace|infection)"
NEGATION = r"(no|without|sin|without evidence of|no evidence of|not seen)"
NEGATED_PATTERN = re.compile(rf"{NEGATION}\s+(?:\w+\s+){{0,3}}{POSITIVE_KWS}", re.I)
POSITIVE_PATTERN = re.compile(POSITIVE_KWS, re.I)
NORMAL_PHRASES = re.compile(r"(normal chest radiograph|no acute cardiopulmonary process|no acute disease)", re.I)

def infer_alert(text: str) -> bool:
    # Si explícitamente dice normal y no hay positivos no negados -> NO alerta
    if NORMAL_PHRASES.search(text) and not POSITIVE_PATTERN.search(text):
        return False
    # Si hay menciones positivas y no todas están negadas -> alerta
    positives = POSITIVE_PATTERN.findall(text)
    if positives:
        # Si todas están negadas en contexto, entonces no alerta
        # (heurística simple)
        negated_hits = NEGATED_PATTERN.findall(text)
        if len(negated_hits) >= len(positives):
            return False
        return True
    # Por defecto, si no hay nada claro, no alerta
    return False

def to_spanish(text: str) -> str:
    out = text
    for pat, rep in TRANSLATE:
        out = re.sub(pat, rep, out, flags=re.I)
    return out

def main():
    ap = argparse.ArgumentParser(description="Convierte DICOMs a PNG, analiza con Med-Gemma y genera reportes ES + CSV.")
    ap.add_argument("--dcm_dir", required=True, help="Carpeta con .dcm")
    ap.add_argument("--out_dir", required=True, help="Carpeta de salida")
    ap.add_argument("--repo_root", default=str(Path(__file__).resolve().parents[1]), help="Raíz del repo (donde está analyze_cxr.py)")
    ap.add_argument("--model", default="google/med-gemma-2-2b-it")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, help="Args extra para analyze_cxr.py")
    args = ap.parse_args()

    dcm_dir = Path(args.dcm_dir)
    out_dir = Path(args.out_dir)
    png_dir = out_dir / "png"
    rep_dir = out_dir / "reportes"
    csv_path = out_dir / "resumen.csv"
    for p in (png_dir, rep_dir):
        p.mkdir(parents=True, exist_ok=True)

    analyzer = Path(args.repo_root) / "analyze_cxr.py"
    if not analyzer.exists():
        raise FileNotFoundError(f"No se encontró analyze_cxr.py en {analyzer}")

    rows = []
    dcm_list = sorted([p for p in dcm_dir.rglob("*.dcm")])
    if not dcm_list:
        print(f"[WARN] No se encontraron DICOMs en {dcm_dir}")
        return

    for dcm in tqdm(dcm_list, desc="Procesando DICOMs"):
        stem = dcm.stem
        png_path = png_dir / f"{stem}.png"

        # 1) Convertir
        try:
            dcm_to_png(dcm, png_path)
        except Exception as e:
            print(f"[ERROR] Falló conversión: {dcm} -> {e}")
            continue

        # 2) Analizar con Med-Gemma
        out, err, code = analyze_with_medgemma(analyzer, png_path, args.model, args.device, args.dtype, args.extra)
        if code != 0:
            print(f"[ERROR] analyze_cxr.py falló en {png_path.name}:\n{err[:400]}")
            continue

        # 3) Heurística: ALERTA si hallazgos positivos no negados
        alerta = infer_alert(out)

        # 4) Versión en español (simple)
        out_es = to_spanish(out)

        # 5) Guardar TXT por imagen
        header = "ALERTA: SE DETECTÓ ANOMALÍA\n" if alerta else "SIN ALERTA: No se detecta anomalía relevante\n"
        body = f"Imagen: {png_path.name}\n\n=== REPORTE (ES) ===\n{out_es}\n\n=== ORIGINAL (EN) ===\n{out}\n"
        (rep_dir / f"{stem}.txt").write_text(header + body, encoding="utf-8")

        rows.append({
            "dicom": str(dcm),
            "png": str(png_path),
            "alerta": "SI" if alerta else "NO",
            "reporte_txt": str(rep_dir / f"{stem}.txt")
        })

    # 6) CSV resumen
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["dicom", "png", "alerta", "reporte_txt"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"\n✅ Listo. PNGs: {png_dir}")
    print(f"📝 Reportes: {rep_dir}")
    print(f"📊 Resumen: {csv_path}")

if __name__ == "__main__":
    main()
