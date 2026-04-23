import json
import base64
import os
import io
from openai import OpenAI
from PIL import Image, ImageOps
import cv2
import numpy as np


# -------------------------------
# LAZY INIT OPENAI CLIENT
# -------------------------------
client = None

def get_client():
    global client
    if client is None:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return client


# -------------------------------
# DESKEW FUNCTION
# -------------------------------
def deskew_image(pil_image):

    img = np.array(pil_image.convert("L"))
    img = cv2.bitwise_not(img)

    thresh = cv2.threshold(
        img, 0, 255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )[1]

    coords = np.column_stack(np.where(thresh > 0))

    if len(coords) == 0:
        return pil_image

    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # Safety check
    if abs(angle) > 15:
        return pil_image

    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    rotated = cv2.warpAffine(
        np.array(pil_image),
        M,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

    return Image.fromarray(rotated)


# -------------------------------
# IMAGE PREPROCESSING
# -------------------------------
def preprocess_image(image_bytes):

    image_stream = io.BytesIO(image_bytes)

    try:
        pil_image = Image.open(image_stream)
        pil_image.verify()
        image_stream.seek(0)
        pil_image = Image.open(image_stream)
    except Exception:
        raise ValueError("Invalid or corrupted image file")

    pil_image = ImageOps.exif_transpose(pil_image)

    # Only deskew if clearly portrait
    if pil_image.height > pil_image.width * 1.2:
        pil_image = deskew_image(pil_image)

    # Resize (cost optimization)
    max_width = 1200
    if pil_image.width > max_width:
        ratio = max_width / pil_image.width
        new_height = int(pil_image.height * ratio)
        pil_image = pil_image.resize((max_width, new_height))

    buffer = io.BytesIO()
    pil_image.convert("RGB").save(buffer, format="JPEG", quality=90)

    return buffer.getvalue()


# -------------------------------
# OPENAI HMR EXTRACTION
# -------------------------------
def extract_hmr_data(image_base64):

    response = get_client().chat.completions.create(
        model="gpt-4o",
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": """
                You are a Hour Meter Reading data extractor.

                Only extract values clearly visible in the image.
                Do NOT guess or infer missing digits.

                Extract only:
                - Hours (format: 2345.23)

                {
                "hours": null
                }
                Rules:
                - If the value is clearly visible → return it as a string (example: "2345.23")
                - If the value is NOT clearly visible → return null
                - Do NOT return empty string ""
                - Output must be valid JSON only. No extra text.

                """
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract HMR details."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        },
                    },
                ],
            }
        ],
    )

    # ✅ DEBUG LOG 
    print("MODEL OUTPUT:", response.choices[0].message.content)


    try:
        return json.loads(response.choices[0].message.content)
    except Exception:
        return {"hours": None}


# -------------------------------
# UNIVERSAL INPUT HANDLER (KEY FIX)
# -------------------------------
def extract_image_bytes(event):

    if "body" not in event:
        raise ValueError("No request body")

    raw_body = event["body"]

    # ----------------------------------
    # CASE 1: API Gateway / Postman Binary
    # ----------------------------------
    if event.get("isBase64Encoded", False):
        try:
            return base64.b64decode(raw_body)
        except Exception:
            raise ValueError("Invalid base64 body")

    body = raw_body

    # ----------------------------------
    # CASE 2: JSON input
    # ----------------------------------
    if isinstance(body, str):
        try:
            parsed = json.loads(body)

            if isinstance(parsed, dict):
                body = parsed.get("image") or parsed.get("body") or body

        except:
            pass

    # ----------------------------------
    # CASE 3: Data URI
    # ----------------------------------
    if isinstance(body, str) and body.startswith("data:image"):
        body = body.split(",", 1)[1]

    # ----------------------------------
    # CASE 4: Base64 string
    # ----------------------------------
    if isinstance(body, str):
        try:
            return base64.b64decode(body)
        except Exception:
            # fallback → treat as raw bytes
            return body.encode()

    # ----------------------------------
    # CASE 5: Already bytes
    # ----------------------------------
    if isinstance(body, (bytes, bytearray)):
        return body

    raise ValueError("Unsupported input format")

def is_valid_hours(val):
    if val is None:
        return False

    val = str(val).strip()

    if val == "":
        return False

    try:
        float(val)
        return True
    except:
        return False


# -------------------------------
# LAMBDA HANDLER
# -------------------------------
def lambda_handler(event, context):

    try:

        # ✅ UNIVERSAL INPUT HANDLING
        image_bytes = extract_image_bytes(event)

        # Size check (10MB)
        if len(image_bytes) > 10 * 1024 * 1024:
            return {
                "statusCode": 400,
                "body": json.dumps({
                    "error": "Image too large. Max 10MB allowed."
                })
            }

        # ✅ PREPROCESS
        processed_bytes = preprocess_image(image_bytes)
        base64_image = base64.b64encode(processed_bytes).decode("utf-8")

        # ✅ MODEL CALL
        result = extract_hmr_data(base64_image)
        hours = result.get("hours")

        # ✅ VALIDATION
        if not is_valid_hours(hours):
            return {
                "statusCode": 400,
                "body": json.dumps({
                    "success": False,
                    "message": "Hour meter reading not clear. Please take a clearer picture."
                })
            }
                
        # ✅ NORMALIZATION
        val = float(str(hours).strip())

        if val.is_integer():
            hours = str(int(val))
        else:
            hours = str(round(val, 2))

        result["hours"] = hours

        return {
            "statusCode": 200,
            "body": json.dumps({
                "success": True,
                "message": "HMR extracted successfully.",
                "data": result
            })
        }

    except ValueError as ve:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": str(ve)})
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }