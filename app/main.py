from fastapi import FastAPI, UploadFile, File, status, HTTPException
from PIL import Image
from io import BytesIO
import numpy as np
from app import ml
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "This is a CNN Classification model API for classifying chest X-rays into one of the four categories: COVID-19, Normal, Pneumonia-Bacterial or Pneumonia-Viral. Returns class probabilities."}


@app.post("/predict-image")
async def predict_api(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg")
    
    if not extension:
        raise HTTPException(status_code = status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                            detail="Image must be in jepg format!")

    try:
        image = np.asarray(Image.open(BytesIO(await file.read())))
        prediction = ml.predict_image(image)

    except Exception as e:
        raise HTTPException(status_code = status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                            detail="Uploaded file is not an image!")

    return prediction
    