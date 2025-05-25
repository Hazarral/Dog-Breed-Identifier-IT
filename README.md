# Machine-Learning

# Overview
This repository has all the tools needed to prepare the images (preprocess the images). You can download it with this link: https://drive.google.com/file/d/1cR9CvGNnjQykvkNn2ZiU3WuK6CA9fDxv/view?usp=sharing

You can find the link in the main() function in main.py, and the preprocessed images are meant to go into the empty "preprocessed images" folder.

GitHub has a limit so I cannot push the dataset into the repository.

# Notes on some parts
The current BATCH_SIZE is a tuple of (10, 10), used for debug printing via matplotlib.pyplot (plt). **matplotlib is not for real training, only for debugging.**

The current resize for every image is **224x224** (represented as a tuple of int, (224, 224)).

IMAGE_PATH and ANNOTATION_PATH are meant for **raw dataset**, these were used when I preprocessed the images before zipping the preprocessed output into the google drive link.

# How to deal with data
in preprocessor.py you will find several functions, but the only note is this:
- crop_and_resize() is only meant for images with annotation, meaning only for training
- preprocess_and_save() is only meant for hard-drive storage. You **EITHER** download the data from the google drive link, **OR** preprocess raw data from Stanford Dogs Dataset, **NOT BOTH**.
- resize() is meant for real usage when user has provided an image, and we only needs to resize it to 224x224. 
