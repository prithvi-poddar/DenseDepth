import os
import glob
import argparse
import time

# Kerasa / TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
from keras.models import load_model
from layers import BilinearUpSampling2D
from loss import depth_loss_function
from utils import predict, load_images, display_images
from matplotlib import pyplot as plt

# Argument Parser
parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
parser.add_argument('--model', default='kitti.h5', type=str, help='Trained Keras model file.')
parser.add_argument('--input', default='examples/*.jpg', type=str, help='Input filename or folder.')
args = parser.parse_args()

# Custom object needed for inference and training
custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': depth_loss_function}

print('Loading model...')

# Load model into GPU / CPU
model = load_model(args.model, custom_objects=custom_objects, compile=False)

print('\nModel loaded ({0}).'.format(args.model))

# Input images
import os
import glob

files = glob.glob('examples/*') 
for f in files:
    os.remove(f)
files = glob.glob('depth_maps/*') 
for f in files:
    os.remove(f)
    
    
import cv2
import numpy as np

cap = cv2.VideoCapture(0)
count = 0

while True:
    count+=1
    ret, frame = cap.read()
    frame = cv2.resize(frame, (640, 480)) 
    if ret:
        cv2.imwrite("examples/frame%d.jpg" % count, frame)
        inputs = load_images( glob.glob(args.input) )
        #print('\nLoaded ({0}) images of size {1}.'.format(inputs.shape[0], inputs.shape[1:]))
        
        # Compute results
        outputs = predict(model, inputs)
        # Display results
        viz = display_images(outputs.copy())
        plt.figure(figsize=(10,5))
        plt.imsave("depth_maps/depth%d.jpg"%count,viz)
        plt.show()
        os.remove("examples/frame%d.jpg" % count)
        
    else:
        break
   
cv2.destroyAllWindows()
        
        
        
        
        
        
