import numpy as np
from PIL import Image
from statistics import mode
import streamlit as st
from streamlit import session_state as ss
from streamlit_drawable_canvas import st_canvas

import cv2
import tensorflow as tf
import keras
from keras.models import load_model #type: ignore

tf.experimental.numpy.experimental_enable_numpy_behavior()

st.set_page_config(
    layout="wide",
    initial_sidebar_state='collapsed'
)

st.title('Handwritten Digit and Character Recognizer')

# Load the model
if 'model' not in ss:
    ss.model = load_model('model.keras')

# Store the class labels
if 'class_labels' not in ss:
    ss.class_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E',
                'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
                'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
                'o', 'p', 'q', 'r', 's', 't', 'u','v', 'w', 'x', 'y', 'z', 'T', 'U',
                'V', 'W','X', 'Y', 'Z']


def normalize_image(image): # Function to normalize the image values
    new_image = tf.cast(image/255., tf.float32)
    return new_image

def get_bbox(img): # Function to get bounding box coordinates
    bbox_coords = []

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    thresh = cv2.threshold(gray,128,255,cv2.THRESH_BINARY)[1]

    contours = cv2.findContours(255-thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    for cntr in contours:
        x,y,w,h = cv2.boundingRect(cntr)
        if (len(bbox_coords) == 0) or (w*h>=500):
            bbox_coords.append([x,y,w,h])
        elif (len(bbox_coords)>0) and (w*h<500):
             prev_y = bbox_coords[-1][1]
             bbox_coords[-1][0] = min(x,bbox_coords[-1][0])
             bbox_coords[-1][1] = min(y,bbox_coords[-1][1])
             bbox_coords[-1][2] += w
             bbox_coords[-1][3] = prev_y+bbox_coords[-1][3] - y
    
    return bbox_coords

def display_bbox(img, bbox_coords): # Function to display image with bounding boxes
        result = img.copy()
        for coords in bbox_coords:
            x,y,w,h = coords
            cv2.rectangle(result, (x, y), (x+w, y+h), (0, 0, 255), 2)
        return result

def get_charac(img, coord): # Function to create a new image with just the character
     x,y,w,h = coord
     charac = img[y:y+h,x:x+w] # Crop the image to get only the character
     new_image = np.full((900, 1200, 3), 255) # Create a white background
     new_image[350:350+charac.shape[0],500:500+charac.shape[1],:] = charac # Overlay the cropped character on the white background
     new_image = new_image.astype(np.uint8)
     return new_image

def rotate_image(image, angle): # Function to rotate the image
  # Rotate the image
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  rotated = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)

  char_coord = get_bbox(rotated)[2] # Get bounding box for rotated character
  rot_img = get_charac(rotated, char_coord) # Replace the background of the rotated character
  return rot_img

def preprocess_image(img): # Function to prepare the image for model input
     resize_img = cv2.resize(img, (256,256)) # Resize the image
     norm_image = normalize_image(resize_img) # Normalize the image
     input_image = norm_image.reshape((1,256,256,3)) # Change the shape of the input image
     return input_image

def get_results(image): # Function to make predictions
    text = []

    bbox_coords = sorted(get_bbox(image), key = lambda x: x[0]) # Identify each word in the image and get their bounding boxes
    
    for coord in bbox_coords:

            no_rot = get_charac(image,coord) # Change background of the image
            rot_left= rotate_image(no_rot,20) # Rotate image and change the background
            rot_right = rotate_image(no_rot, -20) # Rotate image and change the background

            # Preprocess the images and prepare them for model
            proc_no_rot = preprocess_image(no_rot) 
            proc_rot_left = preprocess_image(rot_left)
            proc_rot_right = preprocess_image(rot_right)

            # Make the predictions
            pred_1 = np.argmax(ss.model.predict(proc_no_rot,verbose = False))
            pred_2 = np.argmax(ss.model.predict(proc_rot_left,verbose = False))
            pred_3 = np.argmax(ss.model.predict(proc_rot_right,verbose = False))
            predictions = [ss.class_labels[pred_1], ss.class_labels[pred_2], ss.class_labels[pred_3]]

            # Select most voted class as output
            text.append(mode(predictions))
    
    return text, bbox_coords

upload_image = st.sidebar.file_uploader('Upload your image',type = ['jpg','png','jpeg'])
    
# If image is not uploaded, show the canvas board
if upload_image is None:
    canvas_result = st_canvas(
    fill_color = "rgb(255,255,255)",
    background_color = '#FFFFFF',
    update_streamlit = True,
    drawing_mode = 'freedraw',
    stroke_width = 17,
    width = 1400,
    height = 400
    )

    if get_bbox(canvas_result.image_data[:,:,:3]):
        image_without_alpha = canvas_result.image_data[:,:,:3]
        ss.text,bbox_coords = get_results(image_without_alpha)


else:
    file_bytes = np.asarray(bytearray(upload_image.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)

    st.image(opencv_image, channels='BGR')

    ss.text,bbox_coords = get_results(opencv_image)

# Display the output
try:
    words_on_screen = len(bbox_coords)
    words = ''.join(ss.text[-words_on_screen:])
    st.subheader(words)

except:
        pass