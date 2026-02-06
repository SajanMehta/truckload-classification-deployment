from fastapi import FastAPI, UploadFile, File, HTTPException
from contextlib import asynccontextmanager
from PIL import Image
from io import BytesIO
from typing import Union
#import tensorflow.keras as keras
import keras
import tensorflow as tf
import mlflow
import os
import pandas as pd


#link = "https://tisstorageproduction.blob.core.windows.net/tis-blobstorage-prod/98868187-187-0.jpg"

#model = keras.models.load_model("models/my_model.keras", compile=False)

MLFLOW_TRACKING_USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME")
MLFLOW_TRACKING_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
CURRENT_MODEL_NAME = os.getenv("CURRENT_MODEL_NAME")
CURRENT_MODEL_VERSION = os.getenv("CURRENT_MODEL_VERSION")

URI = f"https://{MLFLOW_TRACKING_USERNAME}:{MLFLOW_TRACKING_PASSWORD}@{MLFLOW_TRACKING_URI}"


mlflow.set_tracking_uri(URI)

model_uri = f"models:/{CURRENT_MODEL_NAME}/{CURRENT_MODEL_VERSION}"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup tasks (load model)
    app.state.model = mlflow.tensorflow.load_model(model_uri)
    #app.state.model = keras.models.load_model("models/my_model.keras", compile=False)
    yield
    # Shutdown tasks
    del app.state.model

app = FastAPI(lifespan=lifespan)

def preprocessing(image: Image.Image, target_size: tuple) -> Image.Image:
    image_resized = image.resize(target_size)
    normalized = tf.keras.preprocessing.image.img_to_array(image_resized) / 255.0
    expanded = tf.expand_dims(normalized, axis=0)
    return expanded

def model_pipeline(image: Image.Image) -> int:
    model = app.state.model
    h, w = model.input_shape[1:3]
    preprocessed_image = preprocessing(image, (w, h))
    output = model.predict(preprocessed_image)
    predicted_class = tf.argmax(output, axis=1).numpy()[0]
    return int(predicted_class)

label_map = {
    0: "Empty Trailer",
    1: "Quarter Full",
    2: "3-Quarter Full",
    3: "Full",
    4: "Not A Trailer"
}

@app.get("/health")
def health():
    model = app.state.model
    return {"ok": True, "model_loaded": model is not None, "model_input_shape": getattr(model, "input_shape", None)}

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

# Link method
# @app.get("/predict/")
# def predict_image():
#     response = requests.get(link)
#     response.raise_for_status()
#     img = Image.open(BytesIO(response.content))
#     preprocessed_image = preprocessing(img, model.input_shape[1:3])
#     output = model.predict(preprocessed_image)
#     predicted_class = tf.argmax(output, axis=1).numpy()[0]
#     return {"predicted_class": int(predicted_class), "model_input_shape": model.input_shape}

@app.post("/predict-csv")
async def predict_csv(file: UploadFile = File(...)):
    # Handling poor upload
    if file.content_type != "text/csv":
        raise HTTPException(status_code=415, detail=f"Unsupported content type: {file.content_type}")
    
    # Read the file
    content = await file.read()

    # Complete import and read in the content to pandas    
    try:
        df= pd.read_csv(BytesIO(content))


    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process CSV file: {str(e)}")
    
    trailers = {}

    for _, row in df.iterrows():
        link = row["manifest_image"]
        response = requests.get(link)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        predicted_class = model_pipeline(img)
        if predicted_class in trailers.keys() and trailers[predicted_class]>0:
            trailers[predicted_class] += 1
        else:
            trailers[predicted_class] = 1

    no_distinct_trailer_classes = len(trailers.keys())
    no_trailers = sum(trailers.values())

    if no_distinct_trailer_classes >= 2 and no_trailers >= 4:
        return {"manifest_prediction": "This manifest is predicted to have the correct items", "links": df["manifest_image"].tolist()}
    else:
        return {"manifest_prediction": "This manifest is predicted to have incorrect items"}
    
@app.post("/predict-image")
async def predict_image(image: UploadFile = File(...)):
    # Handling poor upload
    if image.content_type is None or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail=f"Unsupported content type: {image.content_type}")

    content = await image.read()

    try:
        pil = Image.open(BytesIO(content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process image: {str(e)}")

    predicted_class = model_pipeline(pil)
    return {"predicted_class": label_map[int(predicted_class)]}