from flask import Flask

UPLOAD_FOLDER = 'static/uploads/'
IMAGES_FOLDER = 'object_detection/test_images/'

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['IMAGES_FOLDER'] = IMAGES_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 240 * 1024 * 1024