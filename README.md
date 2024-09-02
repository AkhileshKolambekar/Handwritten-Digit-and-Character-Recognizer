# Handwritten-Digit-and-Character-Recognizer
# Handwritten Digit and Character Recognizer

This project is a deep learning-based application that recognizes handwritten digits (0-9) and characters (A-Z, both uppercase and lowercase). It utilizes TensorFlow for model training and Streamlit for creating an interactive user interface.

## Table of Contents
- [Overview](#overview)
- [Working](#working)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Augmentation](#augmentation)
- [Streamlit App](#streamlit-app)
- [Files](#files)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview
This project involves the creation of a deep learning model to recognize handwritten digits and characters. The model is trained on a dataset of English handwritten characters and can identify both digits and alphabets (uppercase and lowercase). The model is deployed using a Streamlit app, where users can either draw characters using a drawable canvas or upload an image for recognition.

##Working
![proj](https://github.com/user-attachments/assets/ce9554ee-494a-4e99-bcc9-b918f8d15db6)

## Dataset
The dataset used for this project is the [English Handwritten Characters Dataset](https://www.kaggle.com/datasets/dhruvildave/english-handwritten-characters-dataset) from Kaggle.

## Model Architecture
The model is built using TensorFlow and Keras. It has been trained to achieve an accuracy of 80%. The model includes several convolutional layers for feature extraction, followed by dense layers for classification.

## Augmentation
The dataset was augmented using the following techniques:
- **Zooming out**: Enlarging the image while keeping the original content within the frame.
- **Rotation**: Rotating the images left and right by 20 degrees.

The code for augmentation is provided in the `augment.ipynb` file.

## Streamlit App
A Streamlit app has been created for user interaction. Users can either draw characters using the Streamlit drawable canvas or upload images for recognition. The app processes the input and displays the recognized characters.

## Files
- **augment.ipynb**: Contains the code for augmenting the dataset.
- **train_model.ipynb**: Contains the code for building and training the deep learning model.
- **app.py**: Contains the code for the Streamlit app.
- **ab.png and AD.png**: Sample images used to test the model.
- **model.keras**: The trained model with 80% accuracy.

## Usage
1. Clone the repository.
2. Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```
3. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```
4. Interact with the app by either drawing characters using the canvas or uploading an image.

## Results
The trained model achieves an accuracy of 80%. The app is capable of recognizing handwritten digits and characters effectively.

## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue if you have any suggestions or improvements.

## License
This project is licensed under the MIT License.
