from flask import Flask, url_for, request, redirect, render_template, jsonify, flash
from markupsafe import escape
from werkzeug.utils import secure_filename
from PIL import Image


import os
import json
import requests
import numpy as np


IMAGE_SIZE = (150, 150)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

HOST = 'http://localhost'
PORT = 8501
MODEL = 'skin-cancer-detector'


#diseases = ['bkl', 'nv', 'df', 'mel', 'vasc', 'bcc', 'akiec']
diseases = ['nv', 'mel', 'akiec', 'vasc', 'df', 'bcc', 'bkl']

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS







app = Flask(__name__)
# ensuring that the image is not cached
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0





@app.route("/")
def home():
	return redirect(url_for("detector"))



@app.route("/detector", methods = ["GET", "POST"])
def detector():
	if request.method == "GET":
		return render_template("skin-cancer-detector.html", results = [], filepath=url_for('static', filename='image2.jpg'), filename='here will be your picture')

	elif request.method == "POST":
		# check if the post request has the file part
		if 'file' not in request.files:
			#flash('No file part')
			return redirect(request.url)
		file = request.files['file']
		# if user does not select file, browser also
		# submit an empty part without filename
		if file.filename == '':
			#flash('No selected file')
			return redirect(request.url)
		if not allowed_file(file.filename):
			#flash('Only png, jpg and jpeg files please')
			return redirect(request.url)

		if file and allowed_file(file.filename):
			filename = secure_filename(file.filename)
			filepath = os.path.join('static', 'image.jpg')
			file.save(filepath)


		image = Image.open(request.files["file"].stream)

		# resize the image to the size required by the model
		image = image.resize(IMAGE_SIZE)

		# transform the image into a numpy array with values between 0 and 1 
		# and add a batch dimension
		image = np.array(image)[np.newaxis] / 255.
		
		# the data needed for the request
		server_url = f'{HOST}:{PORT}/v1/models/{MODEL}:predict'
		headers = {"content-type": "application/json"}
		data = {"signature_name": "serving_default",
				"instances": image.tolist()}

		# get a response from the model
		response = requests.post(server_url, json = data, headers = headers)
		response.raise_for_status()

		# use the response to determine the top 3 classes
		probabilities = response.json()['predictions'][0] 
		top_3_indices = sorted(range(7), reverse=True, key=lambda i: probabilities[i])[:3]
		results = [ (diseases[i], round(probabilities[i]*100, 1)) for i in top_3_indices]

		return render_template("skin-cancer-detector.html", results = results, filepath=url_for('static', filename='image.jpg'), filename=filename)




if __name__ == "__main__":
	
	
	app.run(host = '0.0.0.0', port = 5100)

