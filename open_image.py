import requests
import numpy as np
import cv2
from PIL import Image
from io import BytesIO

def load_image_from_drive(file_id):
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = requests.get(url)
    
    if response.status_code == 200:
        img = Image.open(BytesIO(response.content))
        img_cv2 = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        return img_cv2
    else:
        raise Exception(f"Gagal download gambar. Status code: {response.status_code}")

file_id = "1jm60AOL0v41uk4mqr-LksMjSGXyAbPMK"
img = load_image_from_drive(file_id)

cv2.imshow("Gambar dari Google Drive", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
