from ultralytics import YOLO
import numpy as np
import streamlit as st
from PIL import Image
import cv2
import io

def pred(img):
    model = YOLO("best.pt")
    img = img.copy()
    img = np.array(img)
    results = model.predict(img, stream=True,)

    for result in results:
        boxes = result.boxes.cpu().numpy()
        number_faces = len(boxes)
        for box in boxes:
            r = box.xyxy[0].astype(int)
            cv2.rectangle(img, r[:2], r[2:], (255, 255, 255), 3)
    return img, number_faces


def main():
    def load_image():
        uploded_file = st.file_uploader(label="## Inson yuzini aniqlash uchun rasm yuklang",
                                        type=["png", 'jpg', 'jpeg'])
        if uploded_file is not None:
            image_data = uploded_file.getvalue()
            st.image(image_data)
            return Image.open(io.BytesIO(image_data))
        else:
            return None

    img = load_image()
    result = st.button("Yuzni aniqlash")
    if result:
        prediction, num_faces = pred(img)
        st.write("Natija!")
        st.image(prediction)
        st.write(f"Rasmdan topilgan yuzlar soni : {num_faces}")

def infinity_run():
    try:
        main()
    except Exception as e:
        main()

if __name__=="__main__":
    infinity_run()



