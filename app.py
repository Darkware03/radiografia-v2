import os
from io import BytesIO
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl
from PIL import Image
import requests
import torch
from transformers import pipeline
import sys, torch, transformers
print(">>> PYTHON:", sys.executable)
print(">>> TORCH:", torch.__version__, "| CUDA available:", torch.cuda.is_available(), "| compiled CUDA:", torch.version.cuda)
print(">>> TRANSFORMERS:", transformers.__version__)

# ========= Configuración del modelo (carga una sola vez) =========
USE_GPU = torch.cuda.is_available()
DTYPE = torch.bfloat16 if USE_GPU else torch.float32
DEVICE = 0 if USE_GPU else -1

# Opcional: silenciar warning de symlinks en Windows
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# Cargar pipeline global (se carga al arrancar el server)
pipe = pipeline(
    task="image-text-to-text",
    model="google/medgemma-4b-it",
    torch_dtype=DTYPE,
    device=DEVICE,
)

# ========= FastAPI =========
app = FastAPI(
    title="MedGemma CXR API",
    description="API para análisis de radiografías de tórax con MedGemma 4B (multimodal).",
    version="1.0.0",
)

class AnalyzeResponse(BaseModel):
    findings: Optional[str]
    impression: Optional[str]
    full_text: str
    device: str

def load_image_from_url(url: str) -> Image.Image:
    try:
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30, stream=True)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"No se pudo descargar/abrir la imagen: {e}")

def load_image_from_file(file: UploadFile) -> Image.Image:
    try:
        data = file.file.read()
        return Image.open(BytesIO(data)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Archivo inválido: {e}")

def build_messages(image: Image.Image, instruction: Optional[str] = None):
    user_text = instruction or (
        "Analiza la radiografía de tórax. "
        "Responde en dos secciones: 1) Findings y 2) Impression. "
        "No inventes hallazgos."
    )
    return [
        {"role": "system", "content": [
            {"type": "text", "text": "Eres un radiólogo experto. Responde claro y conciso."}
        ]},
        {"role": "user", "content": [
            {"type": "text", "text": user_text},
            {"type": "image", "image": image}
        ]}
    ]

def split_report(text: str):
    """Intenta separar Findings / Impression de forma simple."""
    findings, impression = None, None
    t = text.replace("\r", "")
    # Buscas encabezados típicos
    markers = ["Findings", "Impression", "FINDINGS", "IMPRESSION"]
    if any(m in t for m in markers):
        # Muy naive, pero suele servir con el prompt dado
        lower = t.lower()
        f_idx = lower.find("findings")
        i_idx = lower.find("impression")

        if f_idx != -1 and i_idx != -1:
            if f_idx < i_idx:
                findings = t[f_idx:i_idx].split(":", 1)[-1].strip()
                impression = t[i_idx:].split(":", 1)[-1].strip()
            else:
                impression = t[i_idx:f_idx].split(":", 1)[-1].strip()
                findings = t[f_idx:].split(":", 1)[-1].strip()
        elif f_idx != -1:
            findings = t[f_idx:].split(":", 1)[-1].strip()
        elif i_idx != -1:
            impression = t[i_idx:].split(":", 1)[-1].strip()

    return findings, impression

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    image_url: Optional[HttpUrl] = Form(None),
    instruction: Optional[str] = Form(None),
    max_new_tokens: int = Form(160),
    do_sample: bool = Form(False),
    file: UploadFile = File(None),
):
    """
    Envía una radiografía por URL o como archivo (multipart/form-data).
    - image_url: URL directa a la imagen (JPG/PNG).
    - file: archivo subido (campo 'file').
    - instruction: prompt opcional para afinar la respuesta.
    - max_new_tokens: tokens de salida (160 por defecto).
    - do_sample: True para sampling; False = greedy (más estable).
    """

    if not image_url and not file:
        raise HTTPException(status_code=400, detail="Debes enviar 'image_url' o 'file'.")

    # Cargar imagen
    if image_url:
        image = load_image_from_url(str(image_url))
    else:
        image = load_image_from_file(file)

    # Construir mensajes
    messages = build_messages(image, instruction)

    # Generar texto
    out = pipe(
        text=messages,
        max_new_tokens=int(max_new_tokens),
        do_sample=do_sample,
        temperature=0.2 if do_sample else None,
        top_p=0.9 if do_sample else None,
    )

    # Normalizar salida (puede venir como lista de mensajes)
    res = out[0].get("generated_text", "")
    if isinstance(res, list):
        res = res[-1].get("content", "")

    findings, impression = split_report(res)

    return JSONResponse(
        content=AnalyzeResponse(
            findings=findings,
            impression=impression,
            full_text=res,
            device="cuda" if USE_GPU else "cpu"
        ).model_dump()
    )
