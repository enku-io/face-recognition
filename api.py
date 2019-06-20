from flask import Flask
from flask import Flask, abort, request,jsonify
import json
import os
from predict import load_image,align_image,create_model,distance
import numpy as np
from align import AlignDlib
from config import ROOT, ALLOWED_EXTENSIONS,UPLOAD_FOLDER,ANCHOR_FILE,TEST_FILE


app = Flask(__name__)
THRESHOLD = 0.10

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/save',methods=["POST"])
def save():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({"message":"'No selected file'"})
        file = request.files['file']

        if file.filename == '':
            return jsonify({"message":"'No selected file'"})
        if file and allowed_file(file.filename):
            file.save(os.path.join(UPLOAD_FOLDER, ANCHOR_FILE))
            return jsonify({"message":"File saved"})

@app.route('/unlock',methods=["POST"])
def unlock():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({"message":"'No selected file'"})
        file = request.files['file']

        if file.filename == '':
            return jsonify({"message":"'No selected file'"})
        if file and allowed_file(file.filename):
            file.save(os.path.join(UPLOAD_FOLDER, TEST_FILE))
            nn4_small2_pretrained = create_model()
            nn4_small2_pretrained.load_weights(os.path.join(ROOT, 'weights/nn4.small2.v1.h5'))
            img1 = load_image(os.path.join(UPLOAD_FOLDER,ANCHOR_FILE))
            img2 = load_image(os.path.join(UPLOAD_FOLDER,TEST_FILE))
            img1 = align_image(img1)
            img2 = align_image(img2)
            embed1 = nn4_small2_pretrained.predict(np.expand_dims(img1, axis=0))[0]
            embed2 = nn4_small2_pretrained.predict(np.expand_dims(img2, axis=0))[0]
            dist = distance(embed1, embed2)
            if dist < THRESHOLD:
                return jsonify({"message":"You have successfully logged in" })
            else:
                return jsonify({"message": "Sorry, Your face doesn't match with our database"})

if __name__ == '__main__':
    # nn4_small2_pretrained = create_model()
    # nn4_small2_pretrained.load_weights(os.path.join(ROOT, 'weights/nn4.small2.v1.h5'))
    app.run(host='0.0.0.0', port=5000, debug=True)