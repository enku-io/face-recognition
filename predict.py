from model import create_model
from align import AlignDlib
from IdentityMetadata import IdentityMetadata
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.svm import LinearSVC
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os.path
import cv2
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
alignment = AlignDlib((os.path.join(ROOT, 'models/landmarks.dat')))

def load_image(path):
    img = cv2.imread(path, 1)
    # OpenCV loads images with color channels
    # in BGR order. So we need to reverse them
    return img[...,::-1]

def align_image(img):
    return alignment.align(96, img, alignment.getLargestFaceBoundingBox(img), landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)

def distance(emb1, emb2):
    return np.sum(np.square(emb1 - emb2))

def load_metadata(path):
    metadata = []
    for i in os.listdir(path):
        for f in os.listdir(os.path.join(path, i)):
            metadata.append(IdentityMetadata(path, i, f))
    return np.array(metadata)

def findCosineSimilarity(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

if __name__ == '__main__':
    print("inside predict")
    nn4_small2_pretrained = create_model()
    nn4_small2_pretrained.load_weights(os.path.join(ROOT, 'weights/nn4.small2.v1.h5'))
    alignment = AlignDlib((os.path.join(ROOT, 'models/landmarks.dat')))
    img1 = load_image(os.path.join(ROOT,'images/beke.jpeg'))
    img2 = load_image(os.path.join(ROOT,'images/test.jpeg'))
    img1 = align_image(img1)
    img2 = align_image(img2)
    img1 = (img1 / 255.).astype(np.float32)
    img2 = (img2 / 255.).astype(np.float32)
    embed1 = nn4_small2_pretrained.predict(np.expand_dims(img1, axis=0))[0]
    embed2 = nn4_small2_pretrained.predict(np.expand_dims(img2, axis=0))[0]
    print(distance(embed1,embed2))

