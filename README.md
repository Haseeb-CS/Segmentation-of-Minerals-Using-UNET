
Segmentation of Minerals Using UNET


The provided code performs multiclass segmentation on a pre-prepared dataset of excavated mineral images using the UNET architecture. It loads and preprocesses grayscale images and their corresponding masks to train the UNET model for accurately segmenting and identifying different minerals within stones. The model is trained using the given dataset, and its performance is evaluated using accuracy and Intersection over Union (IoU) scores. The code also visualizes the segmentation results, demonstrating the model's capability to enhance the precision and efficiency of geological analysis by identifying and classifying minerals accurately.


## Authors

- [@Haseeb-CS](https://github.com/Haseeb-CS)


## Features

- Multiclass segmentation of minerals in geological images using the UNET architecture
- Loading and preprocessing of grayscale images and their corresponding masks
- Resizing and normalization of input images and masks for model training
- Splitting of dataset into training and testing sets
- Training of the UNET model with adjustable parameters like batch size and epochs
- Evaluation of model performance using accuracy and Intersection over Union (IoU) scores
- Visualization of training and validation loss and accuracy over epochs
- Prediction and visualization of segmented mineral areas on new test images
- Model saving and loading for future use and deployment
- Random selection and testing of images to validate the segmentation results

## 🚀 About Me
I'm a Machine Learning Engineer specializing in computer vision, natural language processing (NLP), and image generation. I develop AI solutions that leverage my expertise in these domains to solve complex problems and create innovative applications.


## 🔗 Links
[![portfolio](https://img.shields.io/badge/my_portfolio-000?style=for-the-badge&logo=ko-fi&logoColor=white)](https://github.com/Haseeb-CS?tab=repositories)
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](www.linkedin.com/in/shahhaseeb281)



## Installation

Follow these steps to set up and run the UNET model for multiclass segmentation on excavated mineral images:

Clone the Repository: Download or clone the repository to your local machine:

```
git clone https://github.com/YourUsername/Geological-Mineral-Segmentation.git
cd Geological-Mineral-Segmentation
```
Install Python: Ensure Python 3.10.1 is installed on your system. You can download it from the official Python website.

Set Up a Virtual Environment (Recommended): Create and activate a virtual environment to manage dependencies:

```
python -m venv venv
# Activate on Windows
venv\Scripts\activate
# Activate on macOS/Linux
source venv/bin/activate
```
Install Required Libraries: Install all necessary Python packages using pip:

```
pip install numpy
pip install opencv-python
pip install pillow
pip install matplotlib
pip install scikit-learn
pip install tensorflow
pip install keras
```
Prepare the Dataset:

Ensure you have the pre-prepared dataset of grayscale images and masks in the specified directory (Dataset/gray/ for images and Dataset/mask_new/ for masks).

Adjust the dataset paths in the code if necessary.

Run the Code: Execute the script to start the training and testing of the UNET model:
```
python script_name.py
```
Troubleshooting:

Verify that all necessary libraries are installed and that the dataset paths are correctly set in the code.
Check the Python environment for any missing dependencies and install them as needed.
By following these steps, you will have the UNET-based segmentation model set up and running on your machine.