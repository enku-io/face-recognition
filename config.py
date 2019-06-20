import os

ROOT = os.path.dirname(os.path.abspath(__file__))
ALLOWED_EXTENSIONS = set([ 'png', 'jpg', 'jpeg'])
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)),"images")
ANCHOR_FILE = "anchor.jpeg"
TEST_FILE = "test.jpeg"