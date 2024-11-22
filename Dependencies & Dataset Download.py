#Install COCO library
pip install pycocotools

#Downloading annotations of the 2017 year.
import os
import requests
import zipfile

def download_annotations(fin_dir="coco_annotations"):
    url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    os.makedirs(fin_dir, exist_ok=True)
    file_path = os.path.join(fin_dir, "annotations_trainval2017.zip")

    print("Downloading COCO annotations...")
    response = requests.get(url, stream=True)
    with open(file_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
    print("Download complete. Extracting annotations...")

    with zipfile.ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall(fin_dir)
    print(f"Annotations extracted to {fin_dir}")

# Download and extract annotations
download_annotations()


#Downloading the Images from annotations path.
from pycocotools.coco import COCO

def download_coco_images(annotation_path, output_dir, num_images=10000):
    coco = COCO(annotation_path)
    os.makedirs(output_dir, exist_ok=True)
    
    image_ids = coco.getImgIds()[:num_images]  # Selects a subset
    for img_id in image_ids:
        img_info = coco.loadImgs(img_id)[0]
        img_url = img_info['coco_url']
        img_path = os.path.join(output_dir, img_info['file_name'])
        
        print(f"Downloading {img_info['file_name']}...")
        response = requests.get(img_url, stream=True)
        with open(img_path, "wb") as f:
            for chunk in response.iter_content(1024): #breaks the response content into chunks of 1024 bytes (1 KB).
                f.write(chunk) #Writes each chunk of data into the file opened at img_path
    print(f"Downloaded {num_images} images to {output_dir}.")

download_coco_images("/Users/harshavardhan/coco_annotations/annotations/instances_train2017.json", "data/coco_subset", num_images=10000) #(num_images= Can be your choice)

