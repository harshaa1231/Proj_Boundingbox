#Install COCO library
pip install pycocotools

import os
import requests
import zipfile
from pycocotools.coco import COCO

# Step 1: Download COCO annotations
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

# Step 2: Download COCO images (10,000 images)
def download_coco_images(annotation_path, output_dir, image_ids, num_images=10000):
    coco = COCO(annotation_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Download only the first 'num_images' images from the list of image_ids
    image_ids = image_ids[:num_images]
    for img_id in image_ids:
        img_info = coco.loadImgs(img_id)[0]
        img_url = img_info['coco_url']
        img_path = os.path.join(output_dir, img_info['file_name'])
        
        try:
            print(f"Downloading {img_info['file_name']}...")
            response = requests.get(img_url, stream=True)
            with open(img_path, "wb") as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
        except Exception as e:
            print(f"Failed to download {img_info['file_name']} due to {e}")
    
    print(f"Downloaded {num_images} images to {output_dir}.")

# Step 3: Filter annotations based on the downloaded image IDs
def filter_annotations(annotation_file, image_ids):
    coco = COCO(annotation_file)
    # Get annotations that correspond to downloaded images
    annotations = coco.loadAnns(coco.getAnnIds(imgIds=image_ids))
    return annotations

# Step 4: Process images and annotations
def process_data(image_dir, annotation_file, num_images=10000):
    # Step 4.1: Download COCO annotations
    download_annotations()

    # Step 4.2: Load COCO dataset
    coco = COCO(annotation_file)

    # Step 4.3: Get image IDs for the first 'num_images'
    image_ids = coco.getImgIds()[:num_images]
    print(f"Found {len(image_ids)} images.")

    # Step 4.4: Download images
    download_coco_images(annotation_file, image_dir, image_ids, num_images)

    # Step 4.5: Filter annotations for the images downloaded
    annotations = filter_annotations(annotation_file, image_ids)
    print(f"Loaded {len(annotations)} annotations for {len(image_ids)} images.")

    return image_ids, annotations

# Step 5: Verify image and annotation alignment
def verify_data(image_ids, annotations, image_dir):
    print(f"Images: {len(image_ids)}")
    print(f"Annotations: {len(annotations)}")
    
    # Verify that images exist for the annotations
    coco = COCO("coco_annotations/annotations/instances_train2017.json")
    for ann in annotations[:5]:  # Checking first 5 annotations
        img_id = ann['image_id']
        img_info = coco.loadImgs(img_id)[0]
        img_filename = img_info['file_name']
        img_path = os.path.join(image_dir, img_filename)
        if os.path.exists(img_path):
            print(f"Image {img_filename} found!")
        else:
            print(f"Image {img_filename} not found!")

# Step 6: Main execution
def main():
    image_dir = "data/coco_subset"
    annotation_file = "coco_annotations/annotations/instances_train2017.json"
    
    image_ids, annotations = process_data(image_dir, annotation_file, num_images=10000)
    
    # Verify the alignment of images and annotations
    verify_data(image_ids, annotations, image_dir)

if __name__ == "__main__":
    main()
