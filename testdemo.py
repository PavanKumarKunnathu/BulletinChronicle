from mtcnn.mtcnn import MTCNN
import tensorflow as tf
import cv2
import numpy as np
from numpy import load
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from keras.models import load_model

from scipy.spatial.distance import cosine
import warnings
print("hii")
# model = tf.keras.models.load_model('facenet_keras.hdf5')

detector=MTCNN()
def extract_face(image,resize=(224,224)):
  image=cv2.imread(image)
  faces=detector.detect_faces(image)
  x1,y1,width,height=faces[0]['box']
  x2,y2=x1+width,y1+height

  face_boundary=image[y1:y2,x1:x2]
  face_image=cv2.resize(face_boundary,resize)
  return face_image
def get_embeddings(faces):
  face=np.asarray(faces,'float32')
  face=preprocess_input(face,version=2)
  model=VGGFace(model='vgg16',include_top=False,input_shape=(224,224,3),pooling='avg')
  return model.predict(face)
def get_similarity(faces):
  embeddings=get_embeddings(faces)
  score=cosine(embeddings[0],embeddings[1])
  return score
print("om")
faces=[extract_face('C:/Users/HP/OneDrive/Pictures/MainProject/users/pavankumar.jpg'),extract_face('C:/Users/HP/OneDrive/Pictures/MainProject/users/pavank.jpg')]

x=get_similarity(faces)
if x<=0.3:
    print("face matched",x)
else:
    print("face not matched",x)
