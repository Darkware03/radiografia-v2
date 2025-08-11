from transformers import pipeline
from PIL import Image
import requests, torch

# Elegir device y dtype automáticamente
use_gpu = torch.cuda.is_available()
device = 0 if use_gpu else -1
dtype = torch.bfloat16 if use_gpu else torch.float32

# Crea el pipeline: MedGemma 4B multimodal (texto + imagen)
pipe = pipeline(
    task="image-text-to-text",
    model="google/medgemma-4b-it",
    torch_dtype=dtype,
    device=device
)

# Radiografía pública (PA chest X-ray)
img_url = "https://upload.wikimedia.org/wikipedia/commons/c/c8/Chest_Xray_PA_3-8-2010.png"
image = Image.open(requests.get(img_url, headers={"User-Agent":"Mozilla/5.0"}, stream=True).raw).convert("RGB")

# Mensaje estilo reporte radiológico breve
messages = [
    {"role": "system", "content": [
        {"type": "text", "text": "Eres un radiólogo experto. Responde claro y conciso."}
    ]},
    {"role": "user", "content": [
        {"type": "text", "text": "Analiza la radiografía de tórax. Da 1) Findings y 2) Impression. No inventes hallazgos."},
        {"type": "image", "image": image}
    ]}
]

# Generar reporte
out = pipe(text=messages, max_new_tokens=220)

# Manejar posibles formatos de salida
res = out[0].get("generated_text", "")
if isinstance(res, list):
    # Formato mensajes
    res = res[-1].get("content", "")
print("\n=== REPORTE ===\n")
print(res)
