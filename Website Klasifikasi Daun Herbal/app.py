import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import skimage
from keras_preprocessing.image import load_img, img_to_array
from flask import Flask, render_template, request, send_from_directory

app = Flask(__name__)

dir_path = os.path.dirname(os.path.realpath(__file__))
UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"

# Load model
cnn_model = tf.keras.models.load_model(STATIC_FOLDER + "/models/" + "savemodel.h5")


# Preprocess an image
def classify(model,image):
    class_names = ['Belimbing Wuluh', 'Jambu Biji', 'Jeruk Nipis', 'Kemangi', 'Lidah Buaya', 'Nangka', 'Pandan', 'Pepaya', 'Seledri', 'Sirih']
    img = load_img(image, target_size = (108,108))
    image_array = img_to_array(img)
    image_array = np.expand_dims(image_array, axis = 0)
    images = np.vstack([image_array])
    classes = model.predict(images, batch_size = 64)
    list_index = [0,1,2,3,4,5,6,7,8,9]
    x = classes

    for i in range(10):
        for j in range(10):
            if x[0][list_index[i]] > x[0][list_index[j]]:
                temp = list_index[i]
                list_index[i] = list_index[j]
                list_index[j] = temp
    
    for i in range(1):
        label = class_names[list_index[i]]
        classified_prob = round(classes[0][list_index[i]] * 100,2)
    return label, classified_prob


# home page
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/classify")
def dataset():
    return render_template("classify.html")

@app.route("/classify", methods=["POST", "GET"])
def upload_file():

    if request.method == "GET":
        return render_template("index.html")
    else:
        try:
            file = request.files["image"]
            upload_image_path = os.path.join(UPLOAD_FOLDER, file.filename)
            print(upload_image_path)
            file.save(upload_image_path)
        except FileNotFoundError:
            return render_template("index.html")    
        label,prob = classify(cnn_model, upload_image_path)
        
    return render_template(
        "classify.html", image_file_name=file.filename, label=label, prob=prob
    )

@app.route("/classify/<filename>")
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == "__main__":
    app.debug = True
    app.run(debug=True)
    app.debug = True