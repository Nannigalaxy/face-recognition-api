import flask
from flask import request, jsonify, make_response, abort
import urllib.request
import warnings
warnings.filterwarnings("ignore")
from model import Facenet
import numpy as np
from model import functions
import cv2

app = flask.Flask(__name__)

# needs few seconds to load model
model = Facenet.loadModel()
print("model loaded")



def get_embedding(model,img_path):
	if type(img_path) == list:
		img_targ = img_path
	else:
		img_targ = [img_path]
	
	input_shape = (160, 160)
	pred_avg = 0

	for targ in img_targ:

		img_face = functions.detectFace(targ, input_shape)
		img_targ_rep = model.predict(img_face)[0,:]
		pred_avg+=np.array(img_targ_rep)

	img_targ_rep = pred_avg/len(img_targ)
	embedding = img_targ_rep.tolist()

	return embedding



@app.route('/')
@app.route('/face-embedding',methods=['GET'])
def home():
    return '''<h1>Face Embedding API</h1>'''



@app.route('/face-embedding/url', methods=['POST'])
def face_embedding_url():
    if not request.json or not 'url' in request.json:
        abort(400)

    img_url = request.json['url']
    req_img = urllib.request.urlopen(img_url)

    img_res = req_img.read()
    img = np.asarray(bytearray(img_res), dtype="uint8")
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)

    vector = get_embedding(model,img)

    # response to request
    response = {'embedding':{
        'url': img_url,
        'vector': vector
    }}
    return jsonify(response), 201



@app.route('/face-embedding/img', methods=['POST'])
def face_embedding_img():
    if not request.json or not 'img' in request.json:
        abort(400)

    img = request.json['img']
    vector = get_embedding_url(model,img)

    # response to request
    response = {'embedding':{
        'url': img_url,
        'vector': vector
    }}
    return jsonify(response), 201


@app.route('/face-embedding/array', methods=['POST'])
def face_embedding_array():
    if not request.json or not 'array' in request.json:
        abort(400)
    img_arr = request.json['array']

    if type(img_arr) == list:
        img_targ = img_arr
    else:
        img_targ = [img_arr]
    imgs = []
    for image in img_arr:
	    img = np.asarray(image, dtype="uint8")
	    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
	    imgs.append(img)
    vector = get_embedding(model,imgs)

    response = {'embedding': {
        'vector': vector
    }}
    return jsonify(response), 201


if __name__=='__main__':
	app.run(threaded=True)
