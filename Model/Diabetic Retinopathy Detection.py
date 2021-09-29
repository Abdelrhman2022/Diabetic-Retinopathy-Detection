from tensorflow import keras
import keras.preprocessing.image as image
import numpy as np

#https://drive.google.com/file/d/1NG-UuGP_q3S5CYuopvNLBeuaytsC8osy/view

# Recreate the exact same model, including its weights and the optimizer
model = keras.models.load_model('/content/drive/MyDrive/data/LastModel.h5')

def get_predictions(img):
    test_img = image.load_img(img, target_size = (128, 128, 3))
    test_img = image.img_to_array(test_img)
    test_img = np.expand_dims(test_img, axis = 0)
    pred = model.predict(test_img)
    return  np.argmax(pred)

img ='/content/drive/MyDrive/data/1008_right.jpeg'
get_predictions(img)
