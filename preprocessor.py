import xml.etree.ElementTree as ET
from matplotlib.text import Annotation
import numpy as np
import cv2
import os
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import csv
from tqdm import tqdm
import random
import pickle

def parse_bounding_box(xml_path : str) -> tuple[int, int, int, int]:
    #Prepare file
    tree : ET.ElementTree = ET.parse(xml_path)
    root : ET.Element = tree.getroot()

    #Parse the numbers needed for the FIRST bounding box
    bndbox : ET.Element = root.find(".//bndbox")
    xmin : int = int(bndbox.find('xmin').text)
    ymin : int = int(bndbox.find('ymin').text)
    xmax : int = int(bndbox.find('xmax').text)
    ymax : int = int(bndbox.find('ymax').text)

    return (xmin, ymin, xmax, ymax)

def crop_and_resize(image_path : str, annotation_path : str, output_size : tuple[int, int]) -> np.ndarray:
    """
    THIS FUNCTION IS ONLY MEANT FOR TRAINING WITH ANNOTATION
    IT WILL FAIL IF THERE IS NO ANNOTATION FILE, LIKE WHEN USER INPUTS THEIR IMAGE

    Crop image based on the corresponding annotation file, then resize it to a standard output value
    The result is a NumPy array
    """

    #Read image as an array
    img : np.ndarray = cv2.imread(image_path, cv2.IMREAD_COLOR_RGB)

    if img is None:
        raise ValueError(f"Failed to load image at {image_path}")


    h, w = img.shape[:2]
    if (h, w) == output_size:
        #Don't crop any further
        return img

    #Convert to rgb
    rgb_img : np.ndarray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


    #Get the bounding box to crop
    (xmin, ymin, xmax, ymax) = parse_bounding_box(annotation_path)

    cropped : np.ndarray = rgb_img[ymin:ymax, xmin:xmax] #Cropping is done like an array
    resized : np.ndarray = cv2.resize(cropped, output_size) #And resize with cv2's methods

    return resized

def preprocess_and_save(image_path : str, annotation_path : str, save_dir : str, size : tuple[int, int]) -> None:
    """
    THIS FUNCTION IS ONLY MEANT FOR MASS-SAVING CROPPED IMAGES IN HARD DRIVE
    IT IS ONLY MEANT TO BE RUN ON IMAGES THAT NEED TO BE STORED PERMANENTLY

    In the case of model evaluation, use crop_and_resize() to return a preprocessed images for the input
    """

    try:
        # Use crop_and_resize to get the preprocessed image as np.ndarray
        preprocessed_img: np.ndarray = crop_and_resize(image_path, annotation_path, size)
    except Exception as e:
        print(f"Error processing {image_path} with annotation {annotation_path}: {e}")
        return

    # Construct the full save path
    filename : str = os.path.basename(image_path)
    full_save_path : str = os.path.join(save_dir, filename)

    # Save the preprocessed image as a file
    success : bool = cv2.imwrite(full_save_path, preprocessed_img)
    if success:
        print(f"Saved preprocessed image to {full_save_path}")
    else:
        print(f"Failed to save image to {full_save_path}")

def read_id_and_label(xml_path : str) -> tuple[str, str]:
     #Prepare file
    tree : ET.ElementTree = ET.parse(xml_path)
    root : ET.Element = tree.getroot()


    img_id : str = os.path.basename(xml_path)
    label : str = root.find("object/name").text.lower()

    return (img_id, label)

def labels_to_csv(ANNOTATION_DIR : list[str]) -> None:
    if not os.path.exists(ANNOTATION_DIR):
        print("Annotation dir not found.")
        return

    data : list[tuple[str, str]] = []

    #Add field names
    data.append(("id", "breed"))
    
    #Then actual data
    for folder in tqdm(os.listdir(ANNOTATION_DIR), total=len(os.listdir(ANNOTATION_DIR)), desc="Appending data from folders"):
        full_annotation_folder_path : str = os.path.join(ANNOTATION_DIR, folder)
        for xml_path in os.listdir(full_annotation_folder_path):
            full_xml_path : str = os.path.join(full_annotation_folder_path, xml_path)

            if not os.path.exists(full_xml_path):
                print(f"XML path does not exists: {full_xml_path}")
                continue

            id_and_label : tuple[str, str] = read_id_and_label(full_xml_path)
            data.append(id_and_label)

    CSV_PATH : str = r"D:\stanford_dataset_train\labels.csv"

    with open(CSV_PATH, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
        print(f"Successfully wrote csv to {CSV_PATH}")

def train_and_val_split(label_dir : str, split_ratio : float = 0.8) -> None:
    TRAIN_IMAGE_PATH : str = r"D:\stanford_dataset_train\preprocessed images"

    SAVE_TRAIN_IMAGE_PATH_NPY : str = r"D:\stanford_dataset_train\X_train.npy"
    SAVE_VAL_IMAGE_PATH_NPY : str = r"D:\stanford_dataset_train\X_val.npy"
    SAVE_TRAIN_LABEL_PATH_NPY: str = r"D:\stanford_dataset_train\y_train.npy"
    SAVE_VAL_LABEL_PATH_NPY: str = r"D:\stanford_dataset_train\y_val.npy"

    SAVE_LABEL_PATH_ENCODE: str = r"D:\stanford_dataset_train\label_encoder.pkl"

    #Create directories if they do not exist
    os.makedirs(r"D:\stanford_dataset_train", exist_ok=True)

    #load csv file
    df = pd.read_csv(label_dir)
    allLabels : np.ndarray = df['breed'].to_numpy()  # Extract labels from the DataFrame

    #Encode label
    print("Encoding labels file...")
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(allLabels)  # strings ? integers
    with open(SAVE_LABEL_PATH_ENCODE, "wb") as f: #save labels as encoded labels file 
        pickle.dump(label_encoder, f)

    # Convert csv df to tuple list
    data = list(zip(df['id'], df['breed']))

    # Shuffle the data 
    random.shuffle(data)

    splitRatio : float = 0.8  # 80% for training, 20% for validation
    splitIndex : int = int(len(data) * splitRatio)

    trainData = data[:splitIndex]
    valData = data[splitIndex:]

    trainImages : list[np.ndarray] = []
    valImages : list[np.ndarray] = []
    trainLabels : list[str] = []
    valLabels : list[str] = []


    for img_id, label in tqdm(trainData, total=len(trainData), desc="Processing images"):
        #Access image paths
        img_dir : str = os.path.join(TRAIN_IMAGE_PATH, img_id + '.jpg')

        #Read image using cv2
        img : np.ndarray = cv2.imread(img_dir, cv2.IMREAD_COLOR_RGB)
        if img is None:
            print(f"Image {img_dir} not found or could not be read.")
            continue

        rgb_img : np.ndarray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        
        #Store in lists
        trainImages.append(rgb_img)
        trainLabels.append(label)

    for img_id, label in tqdm(valData, total=len(valData), desc="Processing validation images"):
        #Access image paths
        img_dir : str = os.path.join(TRAIN_IMAGE_PATH, img_id + '.jpg')

        #Read image using cv2
        img : np.ndarray = cv2.imread(img_dir, cv2.IMREAD_COLOR_RGB)
        if img is None:
            print(f"Image {img_dir} not found or could not be read.")
            continue

        rgb_img : np.ndarray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        
        #Store in lists
        valImages.append(rgb_img)
        valLabels.append(label)

    #Confirm list sizes
    print(f"Train image list size: {len(trainImages)}")
    print(f"Validation image list size: {len(valImages)}")
    print(f"Train label list size: {len(trainLabels)}")
    print(f"Validation label list size: {len(valLabels)}")

    #save into stanford_dataset_train folder
    print("Saving...")
    trainImages = np.array(trainImages)
    valImages = np.array(valImages)
    trainLabels = np.array(trainLabels)
    valLabels = np.array(valLabels)
    np.save(SAVE_TRAIN_IMAGE_PATH_NPY, trainImages)
    np.save(SAVE_VAL_IMAGE_PATH_NPY, valImages)
    np.save(SAVE_TRAIN_LABEL_PATH_NPY, trainLabels)
    np.save(SAVE_VAL_LABEL_PATH_NPY, valLabels)

    print("Success")