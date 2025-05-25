import xml.etree.ElementTree as ET
import numpy as np
import cv2
import os

def parse_bounding_box(xml_path : str) -> tuple[int, int, int, int]:
    #Prepare file
    tree : ET.ElementTree = ET.parse(xml_path)
    root : ET.Element = tree.getroot()

    #Parse the numbers needed
    bndbox : ET.Element = root.find(".//bndbox")
    xmin : int = int(bndbox.find('xmin').text)
    ymin : int = int(bndbox.find('ymin').text)
    xmax : int = int(bndbox.find('xmax').text)
    ymax : int = int(bndbox.find('ymax').text)

    return (xmin, ymin, xmax, ymax)

def resize(image_path : str, output_size : tuple[int, int]) -> np.ndarray:
    """
    THIS FUNCTION ASSUMES THAT THE IMAGE IS ALREADY CROPPED TO INCLUDE ONLY THE RELEVANT SUBJECT
    THUS, ITS ONLY RESPONSIBILITY IS RESIZING
    """

    #Read image as an array
    img : np.ndarray = cv2.imread(image_path, cv2.IMREAD_COLOR_RGB)

    if img is None:
        raise ValueError(f"Failed to load image at {image_path}")

    resized: np.ndarray = cv2.resize(img, output_size)
    return resized


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

def preprocess_and_save(image_path: str, annotation_path: str, save_dir: str, size: tuple[int, int]) -> None:
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
