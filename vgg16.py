from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, VGG16	
import numpy as np
model = VGG16(weights='imagenet', include_top=False, pooling='max')

path = 'elefante.jpg'

img = image.load_img(path, target_size=(780,438)) 
x = image.img_to_array(img)

x = np.expand_dims(x,axis=0)
x = preprocess_input(x)

features = model.predict(x) 

print(features)


