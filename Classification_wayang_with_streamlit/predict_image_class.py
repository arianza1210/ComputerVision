import streamlit as st
from PIL import Image,ImageOps
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from tensorflow.keras import preprocessing
from  tensorflow.keras.models import load_model
from tensorflow.keras.activations import softmax
import os
import h5py
st.header('Image Class Prediction')

def main () :
    file_uploaded = st.file_uploader('Choose the file',type=['jpg','png','jpeg'])
    if file_uploaded is not None:
        image = Image.open(file_uploaded)
        figure=plt.figure()
        plt.imshow(image)
        plt.axis('off')
        result= predict_class(image)
        st.write(result)
        st.pyplot(figure)

def predict_class(image):
    classifier_model=tf.keras.models.load_model(r'models/my_model.hdf5')
    shape=((224,224,3))
    model = tf.keras.Sequential([hub.KerasLayer(classifier_model, input_shape=shape)])
    test_image= image.resize((224,224))
    test_image=preprocessing.image.img_to_array(test_image)
    test_image=test_image/255.0
    test_image=np.expand_dims(test_image,axis=0)
    class_names=['wayang_beber',
                'wayang_gedog',
                'wayang_golek',
                'wayang_krucil',
                'wayang_kulit',
                'wayang_suluh']
    predictions= model.predict(test_image)
    scores =tf.nn.softmax(predictions[0])
    scores=scores.numpy()
    image_class=class_names[np.argmax(scores)]
    result= 'The Predicted Result Of The Image You Entered Is : {}'.format(image_class)
    return result

if __name__=="__main__":
    main()

