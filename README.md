# Image Classification [API](http://164.90.149.249)

---

## Brief Description

This is an Image Classification API for classifying chest X-ray images into the following categories/classes:

- COVID-19
- Normal
- Pneumonia-Bacterial
- Pneumonia-Viral

The API takes a chest X-ray image in jpeg format as input and returns class probabilities. Created using [FastAPI](https://fastapi.tiangolo.com/) framework.
The EfficientNetB3 CNN model pretrained on the imagenet data-set is used and fine-tuned on the 
[Curated Dataset for COVID-19 Posterior-Anterior Chest Radiography Images (X-Rays)](https://data.mendeley.com/datasets/9xkhgts2s6/3) data-set.

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

Also, refer [docs](http://164.90.149.249/docs) for more info.

---

Note: This project is made for educational purposes only. Do not use for real life inference.

---

## References

- [YouTube](https://youtu.be/0sOvCWFmrtA)
- [Kaggle](https://www.kaggle.com/code/gpiosenka/pneumonia-f1-score-86)
