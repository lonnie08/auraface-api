from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import mediapipe as mp
import numpy as np
import base64
import math
import os
import sys
import io
import traceback
import logging
import torch
from torchvision import transforms
from PIL import Image
from typing import Any, Dict, List, Tuple, Literal

import httpx
from pydantic import BaseModel, ValidationError
from dotenv import load_dotenv

from beauty_model import BeautyNet

# Force logs to stderr so Railway captures them
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

app = FastAPI(title="AuraFace API")

load_dotenv()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== MediaPipe ==========
USE_LEGACY_MP = hasattr(mp, "solutions")

if USE_LEGACY_MP:
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    logging.info("MediaPipe: using legacy solutions API")
else:
    from mediapipe.tasks.python import vision, BaseOptions
    TASK_MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")
    face_landmarker = vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=TASK_MODEL_PATH),
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    )
    logging.info("MediaPipe: using tasks API (face_landmarker.task)")

# ========== BeautyNet ==========
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "weights", "model.pth")
device = torch.device("cpu")

beauty_model = None
beauty_normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def load_beauty_model():
    global beauty_model
    if not os.path.exists(WEIGHTS_PATH):
        logging.warning("Beauty model weights not found at %s", WEIGHTS_PATH)
        return
    try:
        model = BeautyNet(n_bins=5, backbone="efficientnet_b0", pretrained=False)
        state_dict = torch.load(WEIGHTS_PATH, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        beauty_model = model
        logging.info("BeautyNet loaded from %s", WEIGHTS_PATH)
    except Exception:
        logging.error("Failed to load BeautyNet:\n%s", traceback.format_exc())


load_beauty_model()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

# OpenAI API (ChatGPT) configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

ROAST_TIMEOUT_MS = max(1, int(os.getenv("ROAST_TIMEOUT_MS", "12000")))

ALLOWED_ROAST_MODES = {"gentle", "brutal"}
ROAST_SYSTEM_PROMPT = (
    "You write short face-analysis roasts for an entertainment app. "
    "Keep it witty and concise (2-4 sentences). "
    "Never include hate speech, protected-class insults, sexual abuse content, self-harm encouragement, "
    "violence threats, or illegal instructions. "
    "Avoid slurs and explicit sexual content."
)


class RoastFaceData(BaseModel):
    face_shape: str
    scores: Dict[str, float]
    proportions: Dict[str, float]
    characteristics: List[str]


class RoastRequest(BaseModel):
    mode: Literal["gentle", "brutal"]
    face_data: RoastFaceData
    lang: str = "en"
    provider: Literal["deepseek", "openai"] = "openai"  # Default to OpenAI


class RoastResponse(BaseModel):
    comment: str


LANGUAGE_NAMES = {
    "en": "English",
    "zh": "Simplified Chinese",
}

def build_roast_user_prompt(payload: RoastRequest) -> str:
    mode_instructions = {
        "gentle": "Tone: playful, warm, non-humiliating. Light teasing only.",
        "brutal": "Tone: sharper sarcasm and bolder teasing, but still safe and non-abusive.",
    }

    lang = payload.lang if payload.lang in LANGUAGE_NAMES else "en"
    lang_name = LANGUAGE_NAMES[lang]

    face = payload.face_data
    return (
        f"{mode_instructions[payload.mode]}\n"
        f"IMPORTANT: You MUST write your entire response in {lang_name} language. Do NOT write in any other language.\n"
        "Write one roast comment in plain text only (no bullet points, no markdown).\n"
        f"Face shape: {face.face_shape}\n"
        f"Scores: {face.scores}\n"
        f"Proportions: {face.proportions}\n"
        f"Characteristics: {face.characteristics}\n"
    )


def extract_deepseek_text(data: Dict[str, Any]) -> str:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("Missing choices in DeepSeek response")

    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        raise ValueError("Missing message in DeepSeek response")

    content = message.get("content", "")
    if isinstance(content, str):
        text = content.strip()
    elif isinstance(content, list):
        text_parts: List[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                value = part.get("text")
                if isinstance(value, str):
                    text_parts.append(value)
        text = "".join(text_parts).strip()
    else:
        text = ""

    if not text:
        raise ValueError("Empty content in DeepSeek response")

    return text


def validate_roast_payload(payload: RoastRequest) -> None:
    face = payload.face_data

    if payload.mode not in ALLOWED_ROAST_MODES:
        raise HTTPException(400, "Invalid mode; must be 'gentle' or 'brutal'")

    if not face.face_shape or not face.face_shape.strip():
        raise HTTPException(400, "face_data.face_shape is required")

    required_scores = {"eyebrows", "eyes", "lips", "nose"}
    if not required_scores.issubset(set(face.scores.keys())):
        raise HTTPException(400, "face_data.scores must include eyebrows, eyes, lips, and nose")

    if not isinstance(face.characteristics, list):
        raise HTTPException(400, "face_data.characteristics must be a list")


async def fetch_deepseek_roast(payload: RoastRequest) -> str:
    if not DEEPSEEK_API_KEY:
        raise HTTPException(503, "Roast service is not configured")

    url = f"{DEEPSEEK_BASE_URL}/chat/completions"
    body = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": ROAST_SYSTEM_PROMPT},
            {"role": "user", "content": build_roast_user_prompt(payload)},
        ],
        "temperature": 0.9,
        "max_tokens": 180,
    }

    try:
        async with httpx.AsyncClient(timeout=ROAST_TIMEOUT_MS / 1000.0) as client:
            response = await client.post(
                url,
                headers={
                    "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=body,
            )
    except httpx.TimeoutException:
        logging.warning("/api/roast upstream timeout mode=%s", payload.mode)
        raise HTTPException(504, "Roast generation timed out")
    except httpx.HTTPError as exc:
        logging.error("/api/roast upstream request failed mode=%s err=%s", payload.mode, exc.__class__.__name__)
        raise HTTPException(502, "Roast provider request failed")

    if response.status_code != 200:
        logging.warning(
            "/api/roast upstream bad status mode=%s status=%s",
            payload.mode,
            response.status_code,
        )
        raise HTTPException(502, "Roast provider returned an invalid response")

    try:
        data = response.json()
        return extract_deepseek_text(data)
    except ValueError as exc:
        logging.error("/api/roast invalid provider payload mode=%s err=%s", payload.mode, str(exc))
        raise HTTPException(502, "Roast provider returned an invalid response")


# ========== OpenAI (ChatGPT) Functions ==========

async def fetch_openai_response(prompt: str, system_prompt: str = "You are a helpful assistant.", max_tokens: int = 200, temperature: float = 0.9) -> str:
    """Call OpenAI API (ChatGPT)"""
    if not OPENAI_API_KEY:
        raise HTTPException(503, "OpenAI service is not configured")

    url = f"{OPENAI_BASE_URL}/chat/completions"
    body = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    try:
        async with httpx.AsyncClient(timeout=ROAST_TIMEOUT_MS / 1000.0) as client:
            response = await client.post(
                url,
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=body,
            )
    except httpx.TimeoutException:
        logging.warning("OpenAI API timeout")
        raise HTTPException(504, "OpenAI request timed out")
    except httpx.HTTPError as exc:
        logging.error("OpenAI API request failed: %s", exc.__class__.__name__)
        raise HTTPException(502, "OpenAI request failed")

    if response.status_code != 200:
        logging.warning("OpenAI API bad status: %s", response.status_code)
        raise HTTPException(502, "OpenAI returned an invalid response")

    try:
        data = response.json()
        return extract_deepseek_text(data)
    except ValueError as exc:
        logging.error("OpenAI invalid payload: %s", str(exc))
        raise HTTPException(502, "OpenAI returned an invalid response")


# ========== Translation ==========

class TranslateRequest(BaseModel):
    text: str
    source_lang: str = "auto"
    target_lang: str = "en"


class TranslateResponse(BaseModel):
    translated_text: str
    detected_lang: str = "en"


SUPPORTED_LANGUAGES = {
    "en": "English",
    "zh": "Chinese",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "ja": "Japanese",
    "ko": "Korean",
    "ar": "Arabic",
    "hi": "Hindi",
}


async def translate_text(text: str, target_lang: str = "en", source_lang: str = "auto") -> tuple[str, str]:
    """Translate text using OpenAI API"""

    if not OPENAI_API_KEY:
        # Fallback: return original text if no API key
        raise HTTPException(503, "Translation service requires OpenAI API key")

    target_name = SUPPORTED_LANGUAGES.get(target_lang, "English")
    source_name = "the source language" if source_lang == "auto" else SUPPORTED_LANGUAGES.get(source_lang, "the source language")

    system_prompt = f"You are a professional translator. Translate the following text from {source_name} to {target_name}. Only output the translated text, nothing else."

    try:
        translated = await fetch_openai_response(
            prompt=text,
            system_prompt=system_prompt,
            max_tokens=1000,
            temperature=0.3
        )
        # Simple language detection
        detected = target_lang  # Default to target, actual detection would require more logic
        return translated, detected
    except Exception as e:
        logging.error("Translation failed: %s", str(e))
        raise HTTPException(500, f"Translation failed: {str(e)}")



def euclidean(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def determine_face_shape(landmarks: List[Tuple[int, int]], h: int, w: int) -> Tuple[str, Dict[str, float]]:
    top = landmarks[10]
    chin = landmarks[152]
    left_cheek = landmarks[234]
    right_cheek = landmarks[454]
    left_jaw = landmarks[172]
    right_jaw = landmarks[397]
    left_temple = landmarks[67]
    right_temple = landmarks[297]

    face_length = euclidean(top, chin)
    cheek_width = euclidean(left_cheek, right_cheek)
    jaw_width = euclidean(left_jaw, right_jaw)
    forehead_width = euclidean(left_temple, right_temple)

    if cheek_width == 0:
        cheek_width = 1.0

    l2w = face_length / cheek_width          # typical 1.0–1.5
    fr = forehead_width / cheek_width         # typical 0.40–0.65
    jr = jaw_width / cheek_width              # typical 0.70–0.90

    # Score each shape based on how well the ratios match its profile
    def gauss(val, ideal, sigma):
        return math.exp(-((val - ideal) ** 2) / (2 * sigma ** 2))

    scores = {
        # Oval: moderate length, balanced forehead & jaw
        "oval": gauss(l2w, 1.25, 0.15) * gauss(fr, 0.53, 0.08) * gauss(jr, 0.80, 0.06),
        # Round: short face, wide jaw
        "round": gauss(l2w, 1.05, 0.12) * gauss(jr, 0.87, 0.05),
        # Square: short face, very wide jaw
        "square": gauss(l2w, 1.08, 0.12) * gauss(jr, 0.90, 0.04),
        # Heart: wider forehead relative to jaw, moderate-long face
        "heart": gauss(l2w, 1.25, 0.18) * gauss(fr - jr, -0.20, 0.08),
        # Diamond: narrow forehead AND narrow jaw (cheekbones widest)
        "diamond": gauss(fr, 0.45, 0.06) * gauss(jr, 0.73, 0.06),
        # Oblong: long face
        "oblong": gauss(l2w, 1.45, 0.12) * gauss(fr, 0.50, 0.10),
    }

    # Normalize to sum to 1
    total = sum(scores.values())
    if total > 0:
        scores = {k: round(v / total, 4) for k, v in scores.items()}
    else:
        scores = {k: round(1.0 / 6, 4) for k in scores}

    # Fix rounding so it sums to exactly 1
    diff = round(1.0 - sum(scores.values()), 4)
    max_key = max(scores, key=scores.get)
    scores[max_key] = round(scores[max_key] + diff, 4)

    shape = max_key.capitalize()
    return shape, scores


def calculate_feature_scores(landmarks: List[Tuple[int, int]]) -> Dict[str, float]:
    face_width = euclidean(landmarks[234], landmarks[454])

    eye_dist = euclidean(landmarks[33], landmarks[263])
    eye_ratio = eye_dist / face_width if face_width > 0 else 0.48
    eye_score = max(4.0, min(9.8, 10 - abs(0.48 - eye_ratio) * 25))

    lip_width = euclidean(landmarks[61], landmarks[291])
    lip_height = euclidean(landmarks[13], landmarks[14])
    lip_ratio = lip_width / lip_height if lip_height > 0 else 2.0
    lip_score = max(5.0, min(9.5, 10 - abs(2.0 - lip_ratio) * 3))

    nose_width = euclidean(landmarks[55], landmarks[285])
    nose_score = max(6.0, min(9.2, 10 - abs(0.35 - nose_width / face_width) * 30))

    brow_angle = abs(math.degrees(math.atan2(
        landmarks[300][1] - landmarks[70][1],
        landmarks[300][0] - landmarks[70][0]
    )))
    brow_score = max(5.5, min(9.0, 10 - abs(15 - brow_angle) * 0.3))

    return {
        "eyebrows": round(brow_score, 1),
        "eyes": round(eye_score, 1),
        "lips": round(lip_score, 1),
        "nose": round(nose_score, 1)
    }


def get_characteristics(face_shape: str) -> List[str]:
    base = ["Balanced proportions", "Symmetrical features"]
    mapping = {
        "Round": ["Apple cheeks", "Soft jawline", "Full cheeks"],
        "Oval": ["High cheekbones", "Balanced forehead", "Gentle chin"],
        "Square": ["Strong jawline", "Square chin", "Angular features"],
        "Heart": ["Wide forehead", "Pointed chin", "Heart-shaped hairline"],
        "Diamond": ["Prominent cheekbones", "Narrow forehead", "Narrow chin"],
        "Oblong": ["High forehead", "Long face", "Straight jawline"]
    }
    return base + mapping.get(face_shape, [])


def get_measurements(landmarks: List[Tuple[int, int]]) -> Dict[str, float]:
    return {
        "face_width_px": round(euclidean(landmarks[234], landmarks[454]), 1),
        "face_height_px": round(euclidean(landmarks[10], landmarks[152]), 1),
        "forehead_width_px": round(euclidean(landmarks[67], landmarks[297]), 1),
        "jaw_width_px": round(euclidean(landmarks[172], landmarks[397]), 1),
    }


def get_proportions(measurements: Dict) -> Dict[str, float]:
    fw = measurements["face_width_px"]
    fh = measurements["face_height_px"]
    return {
        "face_ratio": round(fh / fw, 3) if fw > 0 else 0,
        "forehead_to_cheek_ratio": round(measurements["forehead_width_px"] / fw, 3) if fw > 0 else 0,
        "jaw_to_cheek_ratio": round(measurements["jaw_width_px"] / fw, 3) if fw > 0 else 0
    }


def get_style_recommendations(face_shape: str) -> List[str]:
    recs = {
        "Round": ["Add angles and definition", "Create vertical lines", "Layered hair with volume"],
        "Oval": ["Almost any style works", "Try soft layers or bangs", "Rectangular glasses"],
        "Square": ["Soften the jawline", "Add volume at sides", "Oval or round frames"],
        "Heart": ["Balance wide forehead", "Chin-length bobs", "Side parts"],
        "Diamond": ["Highlight cheekbones", "Add width at forehead and jaw", "Long layers"],
        "Oblong": ["Create width at sides", "Bangs and layers", "Oversized frames"]
    }
    return recs.get(face_shape, recs["Oval"])


@torch.no_grad()
def predict_beauty(rgb_image: np.ndarray, landmarks_raw) -> Dict:
    if beauty_model is None:
        return {}

    pil_image = Image.fromarray(rgb_image).resize((224, 224))
    # Bypass numpy entirely — use PIL raw bytes to build tensor
    raw = bytearray(pil_image.tobytes())  # H*W*3 RGB bytes
    img_tensor = torch.frombuffer(raw, dtype=torch.uint8).reshape(224, 224, 3)
    img_tensor = img_tensor.permute(2, 0, 1).float() / 255.0
    img_tensor = beauty_normalize(img_tensor).unsqueeze(0).to(device)

    lms = [[lm.x, lm.y, lm.z] for lm in landmarks_raw]
    lms_tensor = torch.FloatTensor([lms]).to(device)

    dist, _ = beauty_model(img_tensor, lms_tensor)
    bins = torch.arange(1, 6, dtype=torch.float32, device=device)
    score_raw = (dist * bins).sum(dim=1).item()

    score_100 = round((score_raw - 1) / 4 * 100, 1)
    # Apply a boost to make scores more user-friendly (typical range becomes 45-85)
    score_100 = round(score_100 * 0.85 + 15, 1)
    score_100 = max(0, min(100, score_100))

    distribution = [round(v, 4) for v in dist.squeeze().tolist()]

    return {
        "beauty_score": score_100,
        "beauty_score_raw": round(score_raw, 3),
        "beauty_distribution": distribution,
    }


# ========== API ==========

@app.post("/api/analyze")
async def analyze_face(file: UploadFile = File(...)):
    try:
        content_type = file.content_type or ""
        if not content_type.startswith("image/"):
            raise HTTPException(400, "Only images allowed")

        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
        rgb_image = np.array(pil_img, dtype=np.uint8).copy()

        if USE_LEGACY_MP:
            results = face_mesh.process(rgb_image)
            if not results.multi_face_landmarks:
                raise HTTPException(400, "No face detected")
            landmarks_list = results.multi_face_landmarks[0].landmark
        else:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            results = face_landmarker.detect(mp_image)
            if not results.face_landmarks:
                raise HTTPException(400, "No face detected")
            landmarks_list = results.face_landmarks[0]

        h, w, _ = rgb_image.shape
        landmarks_px = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks_list]

        face_shape, probabilities = determine_face_shape(landmarks_px, h, w)
        scores = calculate_feature_scores(landmarks_px)
        characteristics = get_characteristics(face_shape)
        measurements = get_measurements(landmarks_px)
        proportions = get_proportions(measurements)
        recommendations = get_style_recommendations(face_shape)

        beauty_result = predict_beauty(rgb_image, landmarks_list)

        # Draw annotated image
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        annotated = bgr_image.copy()
        if USE_LEGACY_MP:
            mp_drawing.draw_landmarks(
                image=annotated,
                landmark_list=results.multi_face_landmarks[0],
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
        else:
            for lm in landmarks_list:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(annotated, (cx, cy), 1, (0, 255, 0), -1)

        _, buffer = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        annotated_b64 = base64.b64encode(buffer).decode("utf-8")

        result = {
            "success": True,
            "face_shape": face_shape,
            "face_shape_confidence": round(probabilities.get(face_shape.lower(), 0.5) * 100, 1),
            "shape_probabilities": probabilities,
            "scores": scores,
            "characteristics": characteristics,
            "measurements": measurements,
            "proportions": proportions,
            "style_recommendations": recommendations,
            "annotated_image": f"data:image/jpeg;base64,{annotated_b64}",
            **beauty_result,
        }

        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception:
        logging.error("analyze_face error:\n%s", traceback.format_exc())
        raise HTTPException(500, "Internal analysis error")


@app.post("/api/roast", response_model=RoastResponse)
async def roast_face(payload_raw: Dict[str, Any]):
    try:
        payload = RoastRequest.model_validate(payload_raw)
    except ValidationError:
        raise HTTPException(400, "Invalid roast payload")

    validate_roast_payload(payload)

    # Choose provider based on request
    if payload.provider == "openai":
        comment = await fetch_openai_response(
            prompt=build_roast_user_prompt(payload),
            system_prompt=ROAST_SYSTEM_PROMPT,
            max_tokens=180,
            temperature=0.9
        )
    else:
        comment = await fetch_deepseek_roast(payload)

    return RoastResponse(comment=comment)


@app.post("/api/translate", response_model=TranslateResponse)
async def translate(payload: TranslateRequest):
    """Translate text between languages"""
    if not payload.text or not payload.text.strip():
        raise HTTPException(400, "Text is required")

    try:
        translated, detected = await translate_text(
            text=payload.text,
            target_lang=payload.target_lang,
            source_lang=payload.source_lang
        )
        return TranslateResponse(
            translated_text=translated,
            detected_lang=detected
        )
    except HTTPException:
        raise
    except Exception as e:
        logging.error("/api/translate error: %s", str(e))
        raise HTTPException(500, "Translation failed")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "beauty_model_loaded": beauty_model is not None,
        "weights_path": WEIGHTS_PATH,
        "weights_exists": os.path.exists(WEIGHTS_PATH),
    }


if __name__ == "__main__":
    logging.info("AuraFace API starting...")
    logging.info("Docs: http://localhost:8000/docs")

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
