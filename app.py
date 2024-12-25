from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from io import BytesIO
from PIL import Image

app = FastAPI()

# loadmodel
model = tf.keras.models.load_model("mobilenetv2_model_5.h5")

# load class names from class.txt
with open("classes.txt", "r") as file:
    CLASS_NAMES = [line.strip() for line in file.readlines()]

#  Inference   
def predict_image(img: Image.Image):
    # Resize array
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Inference
    predictions = model.predict(img_array)

    # best confidents
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]

    # name from CLASS_NAMES
    class_name = CLASS_NAMES[predicted_class]

    return {
        "class_id": int(predicted_class),
        "class_name": class_name,
        "confidence": float(confidence)
    }

# Endpoint upload Inference
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # check type
        if not file.content_type.startswith("image/"):
            raise ValueError("Uploaded file is not an image")

        # read file upload
        image_bytes = await file.read()
        img = Image.open(BytesIO(image_bytes))

        # check image can open
        if img.mode != "RGB":
            img = img.convert("RGB")  # change to RGB

        # Inference
        result = predict_image(img)

        # send result with JSON format
        return {"success": True, "result": result}

    except ValueError as ve:
        return {"success": False, "error": str(ve)}

    except Exception as e:
        return {"success": False, "error": f"Unexpected error: {str(e)}"}
