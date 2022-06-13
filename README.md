# Image Classification API

---

## Brief Description

This is an Image Classification model for classifying chest X-ray images into the following categories/classes:

- COVID-19
- Normal
- Pneumonia-Bacterial
- Pneumonia-Viral

The [model](https://github.com/pmahajan11/Image-Classification-API-UbuntuVM/blob/17c89a1fcd3d4837365de93450cde13cfc3f1d84/Image%20Classification%20Model.ipynb) takes a chest X-ray image in jpeg format as input and returns class probabilities.
The EfficientNetB3 CNN model pretrained on the imagenet data-set is used and fine-tuned on the 
[Curated Dataset for COVID-19 Posterior-Anterior Chest Radiography Images (X-Rays)](https://data.mendeley.com/datasets/9xkhgts2s6/3) data-set.

The final model was deployed as an API reated using [FastAPI](https://fastapi.tiangolo.com/) framework.

---

## How to make an api call

Refer the following example code:

```
import requests

url = 'http://164.90.149.249/predict-image/'
path = 'path/to/image.jpeg'

files = {"file": open(path, 'rb')}
response = requests.post(url, files=files)
print(response.json())
```

---

Note: This project is made for educational purposes only. Do not use for real life inference.

---

## References

- [YouTube](https://youtu.be/0sOvCWFmrtA)
- [Kaggle](https://www.kaggle.com/code/gpiosenka/pneumonia-f1-score-86)
