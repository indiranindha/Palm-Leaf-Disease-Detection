from fastapi import FastAPI, UploadFile, File
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import io

from src.models.convnext import build_convnext

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
CONFIDENCE_THRESHOLD = 0.65
NUM_CLASSES = 9

CLASS_NAMES = [
    "Black Scorch",
    "Fusarium Wilt",
    "Healthy sample",
    "Leaf Spots",
    "Magnesium Deficiency",
    "Manganese Deficiency",
    "Parlatoria Blanchardii",
    "Potassium Deficiency",
    "Rachis Blight"
]

# --------------------------------------------------
# APP
# --------------------------------------------------
app = FastAPI(title="Palm Leaf Disease Detection API")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------
# MODEL
# --------------------------------------------------
model = build_convnext(
    num_classes=NUM_CLASSES,
    pretrained=False,
    freeze_backbone=False
)

model.load_state_dict(
    torch.load("checkpoints/best_model.pth", map_location=device)
)

model.to(device)
model.eval()

# --------------------------------------------------
# TRANSFORM
# --------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# --------------------------------------------------
# ROUTES
# --------------------------------------------------
@app.get("/")
def health_check():
    return {"status": "API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image)
        probs = F.softmax(logits, dim=1)

        confidence, pred_idx = torch.max(probs, dim=1)

    confidence = confidence.item()
    pred_idx = pred_idx.item()

    if confidence < CONFIDENCE_THRESHOLD:
        return {
            "prediction": "Bukan Kelapa Sawit",
            "confidence": round(confidence, 4)
        }

    return {
        "prediction": CLASS_NAMES[pred_idx],
        "confidence": round(confidence, 4)
    }
