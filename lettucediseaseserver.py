from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import tensorflow as tf
from PIL import Image
import io

# Initialize FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load TFLite Model
TFLITE_MODEL_PATH = "lettuce_lite_model.tflite"

print(tf.__version__)
print(np.__version__)

try:
    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
    interpreter.allocate_tensors()
    print("✅ TFLite Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading TFLite model: {e}")
    interpreter = None

# Get input and output details
input_details = interpreter.get_input_details() if interpreter else None
output_details = interpreter.get_output_details() if interpreter else None

# Class Names (same as before)
CLASS_NAMES = [
    'Fungal', 'healthy', 'Viral'
]

# Preprocess Image
def preprocess_image(image_data):
    """Convert image to RGB, resize, normalize, and prepare for TFLite model."""
    img = Image.open(io.BytesIO(image_data)).convert("RGB")
    img = img.resize((224, 224))  # Resize to model input size
    img_array = np.array(img).astype("float32") / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Prediction Endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if interpreter is None:
        return JSONResponse(status_code=500, content={"error": "TFLite model not loaded properly."})

    try:
        # Read and preprocess the image
        image_data = await file.read()
        img_array = preprocess_image(image_data)

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])

        # Get prediction class
        class_index = np.argmax(predictions)
        confidence = float(np.max(predictions))

        response = {
            "prediction": CLASS_NAMES[class_index],
            "confidence": confidence
        }
        return JSONResponse(content=response)

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Error during prediction: {str(e)}"})

# Root Endpoint with API status
@app.get("/")
async def root():
    html_content = """
    <html>
        <head>
            <title>API Status</title>
        </head>
        <body style="text-align: center; padding: 50px;">
            <h1 style="color: green;">✅ API is running with TFLite!</h1>
            <p>Use the <code>/predict</code> endpoint to classify plant diseases.</p>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)