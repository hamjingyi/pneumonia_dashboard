# XAI pneumonia Dashboard
2024/2025 Sem 1 WQF7009 Explainable AI (XAI)

1. Folder: **Part 1 Three Models Training Scripts**: This folder consisting the script for model training for AA Part 1
2. Folder: **sample_data**: This is the sample data needed for the XAI Dashboard AA Part 2. 
3. File: **02_aa_part_2_dashboard.py**: The python code for dashboard developing. Please replace with your path to sample_data and mobilenetv2_model.h5 to run the model.
4. File: **mobilenetv2_model.h5**: Backend model to run the pneumonia prediction for XAI dashboard.
5. File: **requirement.txt**: Necessary Python packages.
   
Dashboard is developed using Streamlit. The model used for backend prediction is the sequential model having MobineNetV2 as base model.

**Dashboard's features**
- Choose sample images from the drop-down list or drag and drop chest x-ray images to
feed into the model and make predictions.
- Show the name of the model used, accuracy information and prediction result.
- Visualize various popular Explainable AI (XAI) methods used for understanding
model prediction results.
- Providing color intensity explanations to convey feature importance across various
XAI methods. Users can click to view explanations.
