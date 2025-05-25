from tqdm import tqdm
from preprocessor import *
import os

try:
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    from tqdm import tqdm
except ImportError as e:
    print("Missing dependencies. Please run: pip install -r requirements.txt")
    exit(1)

def get_image_paths(image_paths : list[str], IMAGE_DIR : str) -> None:
    print("Parsing images path...")

    for category in os.listdir(IMAGE_DIR): #This returns the name of each folder category
        category_path : str = os.path.join(IMAGE_DIR, category) #Construct full category path

        for image in os.listdir(category_path): #Then the name of each images in each categories
            image_full_path : str = os.path.join(category_path, image) 
            
            if not image_full_path.lower().endswith((".jpg",)):
                print(f"Error: expected jpg format, found {image_full_path} instead.")
                continue

            image_paths.append(image_full_path)

    print(f"Image path done!")
    print(f"Images: {len(image_paths)}")

def get_annotation_paths(annotation_paths : list[str], ANNOTATION_DIR : str) -> None:
    print("Parsing annotation paths...")

    for category in os.listdir(ANNOTATION_DIR): #This returns the name of each folder category
        category_path : str = os.path.join(ANNOTATION_DIR, category) #Construct full category path

        for annotation in os.listdir(category_path): #Then the name of each annotation in each categories
            annotation_full_path : str = os.path.join(category_path, annotation)
            annotation_paths.append(annotation_full_path)

    print(f"Annotation path done!")
    print(f"Annotations: {len(annotation_paths)}")
    print(f"Classes: {len(os.listdir(ANNOTATION_DIR))}")

def normalize_label(category: str) -> str:
    parts = category.split('-')
    label_parts = parts[1:] if len(parts) > 1 else []

    label = ' '.join(label_parts).replace('_', ' ').lower()
    return label.title()

def load_preprocessed_images(PREPROCESS_DIR: str, max_images: int = 100) -> list[np.ndarray]:
    preprocessed_images: list[np.ndarray] = []

    all_files = os.listdir(PREPROCESS_DIR)
    total_to_load = min(len(all_files), max_images)

    for image_file in tqdm(all_files[:total_to_load], desc="Loading preprocessed images"):
        image_file_path = os.path.join(PREPROCESS_DIR, image_file)
        img = cv2.imread(image_file_path, cv2.IMREAD_COLOR_RGB)
        if img is None:
            print(f"Warning: failed to read {image_file_path}")
            continue
        preprocessed_images.append(img)

    return preprocessed_images

def show_image_batch(images: list[np.ndarray], BATCH_SIZE: tuple[int, int]) -> None:
    fig : plt.Figure = plt.figure(figsize=(15, 15))
    fig.suptitle("Image Batch", fontsize=16)

    #Rows and cols to display
    rows, cols = BATCH_SIZE

    for i in range(min(len(images), rows * cols)):
        img : np.ndarray = images[i]

        subplot_index: int = i + 1  # matplotlib subplot indices start at 1
        ax: plt.Axes = fig.add_subplot(rows, cols, subplot_index)

        # No need to check channel count, all images are assumed RGB (H, W, 3
        ax.imshow(img)

        ax.axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)  # Adjust for title
    plt.show()

def main() -> None:
    """
    Please download the dataset at: https://drive.google.com/file/d/1cR9CvGNnjQykvkNn2ZiU3WuK6CA9fDxv/view?usp=sharing
    And place the preprocessed images into the empty "preprocessed images" folder
    The "Annotation" folder will also be extracted, but used later
    """

    #Directories and batch size
    #CHANGE THESE PATHS TO FIT YOUR COMPUTER
    IMAGE_DIR : str = r"D:\Stanford dog breeds\Images"
    ANNOTATION_DIR : str = r"D:\Stanford dog breeds\Annotation"
    BATCH_SIZE : tuple[int, int] = (10, 10)
    
    for path_name, path in [("IMAGE_DIR", IMAGE_DIR), ("ANNOTATION_DIR", ANNOTATION_DIR)]:
        if not os.path.isdir(path):
            raise FileNotFoundError(f"{path_name} not found: {path}")

    # Create folder "preprocessed images" in current working directory
    PREPROCESS_DIR = os.path.join(os.getcwd(), "preprocessed images")
    os.makedirs(PREPROCESS_DIR, exist_ok=True)

    #Store the image and annotation paths
    image_paths : list[str] = []
    annotation_paths : list[str] = []

    labels : list[str] = os.listdir(ANNOTATION_DIR)

    #Get image and annotation paths
    get_image_paths(image_paths, IMAGE_DIR)
    get_annotation_paths(annotation_paths, ANNOTATION_DIR)

    #Change label to normalized names
    for category in ANNOTATION_DIR:
        labels.append(normalize_label(category))

    #Load the prepocessed images
    preprocessed_images : list[np.ndarray] = load_preprocessed_images(PREPROCESS_DIR, 10)

    show_image_batch(preprocessed_images, BATCH_SIZE)

if __name__ == "__main__":
    main()