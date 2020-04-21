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


def get_embedding(model,req_img):
	
	img_res = req_img.read()
	img = np.asarray(bytearray(img_res), dtype="uint8")
	img = cv2.imdecode(img, cv2.IMREAD_COLOR)

	input_shape = (160, 160)

	img_face = functions.detectFace(img, input_shape)
	img_targ_rep = model.predict(img_face)[0,:]

	# converting numpy array to list
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
    vector = get_embedding(model,req_img)

    # response as json
    embedding = {
        'url': img_url,
        'vector': vector
    }
    return jsonify({'face_embedding': embedding}), 201


# @app.route('/face-embedding/mongodb/gridfs', methods=['POST'])
# def face_embedding_mongo():
#     if not request.json or not 'raw_gridfs_file' in request.json:
#         abort(400)

#     grid_fs = request.json['raw_gridfs_file']
#     vector = get_embedding(model,grid_fs)

#     embedding = {
#         'id': grid_fs._id,
#         'name': grid_fs.filename,
#         'vector': vector
#     }
#     return jsonify({'face_embedding': embedding}), 201


if __name__=='__main__':
	app.run(threaded=True)
