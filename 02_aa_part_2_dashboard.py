import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from lime import lime_image
from skimage.segmentation import mark_boundaries
from skimage.transform import resize
from skimage.segmentation import quickshift
import gc
import os
import time

def clear_memory():
    gc.collect()  # Force garbage collection

# Load and preprocess the input image
@st.cache_data
def load_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

# print actual and prediction score
def print_actual_and_prediction(img_path, model):
    img_array = load_image(img_path)
    preds = model.predict(img_array)
    predicted_label = np.argmax(preds[0])

    class_labels = {0: 'NORMAL', 1: 'PNEUMONIA'}  # Example class labels
    pneumonia_keywords = ['bacteria', 'virus']

    # Check for 'normal' or pneumonia-related keywords in the img_path
    if 'normal' in img_path.lower():
        actual_label = 'NORMAL'
    elif any(keyword in img_path.lower() for keyword in pneumonia_keywords):
        actual_label = 'PNEUMONIA'
    else:
        # Default label if none of the keywords match
        actual_label = 'UNKNOWN'

    # Display the image in the sidebar
    img = Image.open(img_path)
    st.sidebar.image(img, caption="Input Original Image", use_container_width=True)

    # Display model name, actual and predicted labels in sidebar
  
    st.sidebar.write("**Actual Label:**", actual_label)
    st.sidebar.write("**Predicted Label:**", class_labels.get(predicted_label, "Unknown"))

    # Prepare prediction scores with labels
    prediction_scores = preds[0]
    prediction_with_labels = {class_labels[i]: prediction_scores[i] for i in range(len(prediction_scores))}

    # Convert prediction scores into a pandas DataFrame
    prediction_df = pd.DataFrame(list(prediction_with_labels.items()), columns=["Label", "Score"])

    # Display prediction scores in sidebar as a table
    st.sidebar.write("**Prediction Scores:**")
    st.sidebar.dataframe(prediction_df.style.set_table_styles(
        [{'selector': 'thead th', 'props': [('background-color', 'lightblue')]}],
        overwrite=False
    ))

    st.sidebar.write("**Model Used:**", 'mobilenetv2_model')
    
    model_comparison_data = {
        'Model': ['MobileNetV2', 'VGG16', 'CNN from Scratch', ],
        'Testing Accuracy': [0.977, 0.961, 0.956],
        'F1 Score': [0.98, 0.96, 0.96]
    }

    model_comparison_df = pd.DataFrame(model_comparison_data)
    st.sidebar.dataframe(model_comparison_df)

# GradCam

# last convolutional layer
@st.cache_data
def get_last_conv_layer_model(model, last_conv_layer_name):
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):  # Check if the layer is a sub-model
            sub_model = layer
            if last_conv_layer_name in [l.name for l in sub_model.layers]:
                return sub_model
    raise ValueError(f"Layer {last_conv_layer_name} not found in the model.")

@st.cache_data
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    sub_model = get_last_conv_layer_model(model, last_conv_layer_name)
    grad_model = tf.keras.models.Model(
        [sub_model.inputs], [sub_model.get_layer(last_conv_layer_name).output, sub_model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = tf.gather(preds[0], pred_index)

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    pooled_grads = tf.reshape(pooled_grads, [-1])
    heatmap = tf.einsum('ijk,k->ij', last_conv_layer_output, pooled_grads)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    return heatmap.numpy()

@st.cache_data
def display_gradcam(img_path, heatmap, alpha=0.4):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    heatmap_resized = np.uint8(255 * heatmap_resized)
    heatmap_resized = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

    # Superimpose the heatmap on the original image
    superimposed_img = cv2.addWeighted(img, alpha, heatmap_resized, 1 - alpha, 0)

    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(heatmap_resized, caption="Grad-CAM Heatmap", use_container_width=True)
    
    with col2:
        st.image(superimposed_img, caption="Superimposed Image", use_container_width=True)

    clear_memory()

# Saliency Map
@st.cache_data
def make_saliency_map(img_array, model, pred_index=None):
    img_array = tf.convert_to_tensor(img_array)
    with tf.GradientTape() as tape:
        tape.watch(img_array)  # Watch the input image
        preds = model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, img_array)
    saliency_map = tf.reduce_max(tf.abs(grads), axis=-1)  # Take the absolute value and find max over color channels
    saliency_map = saliency_map[0].numpy()

    # Normalize the saliency map
    saliency_map = np.maximum(saliency_map, 0)  # Remove negative values
    saliency_map = saliency_map / np.max(saliency_map)  # Normalize between 0 and 1

    return saliency_map

@st.cache_data
def display_saliency_map(img_path, saliency_map, alpha=0.4):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize the saliency map to match the image size
    saliency_resized = cv2.resize(saliency_map, (img.shape[1], img.shape[0]))

    # Normalize and apply color map to the saliency map
    saliency_resized = np.uint8(255 * saliency_resized)
    saliency_resized = cv2.applyColorMap(saliency_resized, cv2.COLORMAP_HOT)

    # Superimpose the saliency map on the original image
    superimposed_img = cv2.addWeighted(img, alpha, saliency_resized, 1 - alpha, 0)

    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(saliency_resized, caption="Saliency Map", use_container_width=True)
    
    with col2:
        st.image(superimposed_img, caption="Superimposed Image", use_container_width=True)
    
    clear_memory()

# Smooth Gradient
@st.cache_data
def make_smoothgrad_heatmap(img_array, model, last_conv_layer_name, pred_index=None, num_samples=50, noise_level=0.1):

    # Prepare the model for gradient calculation
    sub_model = get_last_conv_layer_model(model, last_conv_layer_name)
    grad_model = tf.keras.models.Model(
        [sub_model.inputs], [sub_model.get_layer(last_conv_layer_name).output, sub_model.output]
    )

    all_grads = []

    for _ in range(num_samples):
        # Add random noise to the image
        noise = np.random.normal(loc=0.0, scale=noise_level, size=img_array.shape)
        noisy_img = img_array + noise
        noisy_img = np.clip(noisy_img, 0.0, 255.0)

        with tf.GradientTape() as tape:
            noisy_img_tensor = tf.convert_to_tensor(noisy_img)
            last_conv_layer_output, preds = grad_model(noisy_img_tensor)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = tf.gather(preds[0], pred_index)

        # Calculate gradients
        grads = tape.gradient(class_channel, last_conv_layer_output)
        all_grads.append(grads.numpy())

    # Average the gradients over all samples
    avg_grads = np.mean(all_grads, axis=0)

    # Pool the gradients and apply them to the last convolutional layer output
    pooled_grads = np.mean(avg_grads, axis=(0, 1, 2))  # Take mean across the height, width, and channels
    last_conv_layer_output = last_conv_layer_output[0]
    pooled_grads = np.reshape(pooled_grads, [-1])

    # Create the heatmap by applying the pooled gradients to the output of the last convolutional layer
    heatmap = np.einsum('ijk,k->ij', last_conv_layer_output, pooled_grads)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)  # Normalize the heatmap

    return heatmap

@st.cache_data
def display_smoothgrad(img_path, heatmap, alpha=0.4):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    heatmap_resized = np.uint8(255 * heatmap_resized)
    heatmap_resized = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

    superimposed_img = cv2.addWeighted(img, alpha, heatmap_resized, 1 - alpha, 0)

    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(heatmap_resized, caption="SmoothGrad HeatMap", use_container_width=True)
    
    with col2:
        st.image(superimposed_img, caption="Superimposed Image", use_container_width=True)
    
    clear_memory()

# Occlusion Sensitivity
size = 16

def apply_occlusion(img_array, block_size=size, stride=size):
    img_height, img_width, _ = img_array.shape[1], img_array.shape[2], img_array.shape[3]
    occlusion_map = np.zeros((img_height, img_width))

    for y in range(0, img_height, stride):
        for x in range(0, img_width, stride):
            img_copy = np.copy(img_array)
            img_copy[0, y:y+block_size, x:x+block_size, :] = 0  # Occlude the block
            occlusion_map[y:y+block_size, x:x+block_size] = model.predict(img_copy)[0][np.argmax(model.predict(img_copy))]  # Record prediction

    return occlusion_map

@st.cache_data
def compute_occlusion_sensitivity(img_array, model, block_size=size, stride=size):
    occlusion_map = apply_occlusion(img_array, block_size, stride)
    sensitivity_map = np.abs(occlusion_map - np.max(occlusion_map))  # Measure how much occlusion affects the prediction
    sensitivity_map = sensitivity_map / np.max(sensitivity_map)  # Normalize the sensitivity map
    return sensitivity_map

@st.cache_data
def display_occlusion_sensitivity(img_path, sensitivity_map, alpha=0.5):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    sensitivity_map_resized = cv2.resize(sensitivity_map, (img.shape[1], img.shape[0]))

    sensitivity_map_resized = np.uint8(255 * sensitivity_map_resized)
    sensitivity_map_resized = cv2.applyColorMap(sensitivity_map_resized, cv2.COLORMAP_JET)

    superimposed_img = cv2.addWeighted(img, alpha, sensitivity_map_resized, 1 - alpha, 0)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(sensitivity_map_resized, caption="Occlusion Sensitivity Map", use_container_width=True)
    
    with col2:
        st.image(superimposed_img, caption="Superimposed Image", use_container_width=True)
    clear_memory()

# LIME

@st.cache_data
def load_image_LIME(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    return img_array  # Return the image as-is for visualization

@st.cache_data
def prediction_function_LIME(images):
    images = np.array(images)
    images = tf.keras.applications.mobilenet_v2.preprocess_input(images)  # Apply preprocessing for model
    predictions = model.predict(images)
    return predictions

def display_lime_explanation(img_path, explanation, segments):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize segments to match the original image size
    segments_resized = resize(segments, (img.shape[0], img.shape[1]), mode='reflect', preserve_range=True).astype(int)

    # Get the explanation and mask
    temp, mask = explanation.get_image_and_mask(
        label=np.argmax(explanation.top_labels),
        positive_only=True,
        num_features=7,
        hide_rest=False
    )

    if temp.max() <= 1.0:
        temp = (temp * 255).astype(np.uint8)  # Convert to [0, 255] for display

    mask_resized = resize(mask, (img.shape[0], img.shape[1]), mode='reflect', preserve_range=True)
    mask_resized = np.uint8(255 * mask_resized)

    heatmap = cv2.applyColorMap(mask_resized, cv2.COLORMAP_JET)

    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    segmented_image = mark_boundaries(img, segments_resized, color=(1, 0, 0), mode='outer')

    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(segmented_image, caption="Segmented Image", use_container_width=True)
    
    with col2:
        st.image(superimposed_img, caption="LIME Explanation Overlay", use_container_width=True)

    clear_memory()


def explain_with_lime(img_path, model):
    img_array = load_image_LIME(img_path)

    # Expand dimensions to match the batch shape (1, height, width, channels)
    img_array_batch = np.expand_dims(img_array, axis=0)

    # Initialize the LIME explainer
    explainer = lime_image.LimeImageExplainer()

    from skimage.segmentation import slic

    def custom_segmentation(image):
      # Apply SLIC segmentation
      segments = slic(image, n_segments=50, compactness=10, sigma=1)

      # Assuming you have a method to identify lung regions (e.g., using thresholding)
      # Create a mask for lung regions (this is an example; adjust it based on your data)
      gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
      _, lung_mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

      # Optionally, smooth or refine the mask to only focus on lung areas
      lung_mask = cv2.dilate(lung_mask, None, iterations=2)
      lung_mask = cv2.erode(lung_mask, None, iterations=2)

      # Filter out segments that are outside of the lung mask
      segments[lung_mask == 0] = -1  # Mark regions outside the lung mask as -1

      return segments
      # return slic(image, n_segments=200, compactness=10, sigma=1)

    # Explain the prediction
    explanation = explainer.explain_instance(
        image=img_array,
        classifier_fn=prediction_function_LIME,
        top_labels=2,
        hide_color=0,
        num_samples=100,
        segmentation_fn=custom_segmentation
    )

    # Get segmentation for the image
    segments = explanation.segments

    return explanation, segments
    


model = load_model('/content/drive/MyDrive/XAI/modelsaved/mobilenetv2_model.h5')

# Streamlit UI
st.title("XAI: Pneumonia Dashboard")
st.markdown('<p style="color:grey; font-style:italic;">Created by: Ham Jing Yi, 23100257</p>', unsafe_allow_html=True)

# folder where sample images are stored/uploaded
ROOT_FOLDER = '/content/drive/MyDrive/XAI/sample_data'
image_files = [f for f in os.listdir(ROOT_FOLDER) if os.path.isfile(os.path.join(ROOT_FOLDER, f))]

# drop down input
st.sidebar.header("Input")
selected_image = st.sidebar.selectbox("Select an image:", image_files)
img_path = os.path.join(ROOT_FOLDER, selected_image)

# testing input
# img_path = '/content/drive/MyDrive/XAI/chest_xray/test/PNEUMONIA/person111_bacteria_536.jpeg'

# upload image
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpeg", "png", "jpg"])
st.sidebar.markdown("<br>", unsafe_allow_html=True) 

# Handle paths for the selected image
if uploaded_file is not None:
    # Save the uploaded file to the ROOT_FOLDER
    uploaded_filename = uploaded_file.name
    tmp_path = os.path.join(ROOT_FOLDER, uploaded_filename)

    with open(tmp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())  # Save the file to the folder
    
    # Reload the image files list to include the newly uploaded file
    image_files = [f for f in os.listdir(ROOT_FOLDER) if os.path.isfile(os.path.join(ROOT_FOLDER, f))]

    upload_finish = st.empty()
    upload_finish.success(f"Uploaded image '{uploaded_filename}' successfully!")
    time.sleep(3)
    upload_finish.empty()

    selected_image = uploaded_filename
    img_path = tmp_path

last_conv_layer_name = 'Conv_1'

if selected_image != "Select an image":
  
  st.write(f"This is the output of XAI techniques of '**{selected_image}**'.")
  img_array = load_image(img_path)
  print_actual_and_prediction(img_path, model)

  col_width = 1
  col1, col2 = st.columns([col_width, col_width])

  with col1:
      # Grad-CAM
      st.subheader(":green[1: Grad-CAM]  ")
      gradcam_heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
      display_gradcam(img_path, gradcam_heatmap)
      with st.expander("Click to view explanation"):
        st.markdown("""
        1. Brightest colors (red/yellow): High importance (strong influence on model's prediction).
        2. Darkest colors (blue/green): Low importance (little impact on model's prediction).
        """)

  with col2:
    # Saliency Map
      st.subheader(":green[2: Saliency Map]  ")
      saliency_map = make_saliency_map(img_array, model)
      display_saliency_map(img_path, saliency_map)
      with st.expander("Click to view explanation"):
        st.markdown("""
        1. Brightest colors (blue): High importance (strong influence on model's prediction).
        2. Darkest colors (black): Low importance (little impact on model's prediction).
        """)


  col3, col4 = st.columns([col_width, col_width])
  with col3:
      # LIME
      st.subheader(":green[3: LIME]  ")
      explanation, segments = explain_with_lime(img_path, model)
      display_lime_explanation(img_path, explanation, segments)
      with st.expander("Click to view explanation"):
        st.markdown("""
        Blue Segments: High importance (strong influence on model's prediction).
        """)

  with col4:
      # Smooth Gradient
      st.subheader(":green[4: Smooth Gradient]  ")
      smoothgrad_heatmap = make_smoothgrad_heatmap(img_array, model, last_conv_layer_name)
      display_smoothgrad(img_path, smoothgrad_heatmap)
      with st.expander("Click to view explanation"):
        st.markdown("""
          1. Brightest colors (red/yellow): High importance (strong influence on model's prediction).
          2. Darkest colors (blue/green): Low importance (little impact on model's prediction).
          """)

  col5, col6 = st.columns([col_width, col_width])

  with col5:
      # Occlusion Sensitivity
      st.subheader(":green[5: Occlusion Sensitivity]  ")
      # size = st.number_input("Enter block size for occlusion", min_value=1, max_value=64, value=16, step=1)
      sensitivity_map = compute_occlusion_sensitivity(img_array, model, block_size=size)
      display_occlusion_sensitivity(img_path, sensitivity_map)
      with st.expander("Click to view explanation"):
        st.markdown("""
        1. Red/Yellow: The darker the red, higher importance (strong influence on model's prediction).
        2. Blue/green): Low importance (little impact on model's prediction).
        """)

  run_finish = st.empty()
  run_finish.success("XAI processing complete!.")
  time.sleep(3)
  run_finish.empty()



