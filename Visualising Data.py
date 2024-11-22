import os
import cv2
import matplotlib.pyplot as plt
from pycocotools.coco import COCO

# Function to load COCO annotations
def load_coco_annotations(annotation_path, image_dir):
    coco = COCO(annotation_path)
    
    # Get image IDs
    image_ids = coco.getImgIds()
    images = []
    annotations = []
    
    for img_id in image_ids:
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(image_dir, img_info['file_name'])
        
        # Get annotations (bounding boxes, categories)
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        
        # Extract bounding box and class information
        boxes = []
        classes = []
        for ann in anns:
            bbox = ann['bbox']  # [x, y, width, height]
            category = ann['category_id']
            boxes.append(bbox)
            classes.append(category)
        
        images.append(img_path)
        annotations.append({
            'boxes': boxes,
            'classes': classes
        })
    
    return coco, images, annotations

# Function to display image with annotations
def display_image_with_annotations(img_path, annotations, coco):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Image not found: {img_path}")
        return
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    
    # Get category names
    categories = coco.loadCats(coco.getCatIds())
    category_names = {cat['id']: cat['name'] for cat in categories}
    
    for bbox, category in zip(annotations['boxes'], annotations['classes']):
        x, y, w, h = bbox
        plt.gca().add_patch(plt.Rectangle((x, y), w, h, fill=False, color='green', linewidth=2))
        plt.text(x, y - 10, category_names[category], color='green', fontsize=12)
    
    plt.axis('off')  # Hiding axes
    plt.show()

# Display multiple images with annotations
def display_multiple_images(coco, images, annotations, num_images=20):
    for i in range(min(num_images, len(images))):
        print(f"Displaying image {i+1}/{num_images}: {os.path.basename(images[i])}")
        display_image_with_annotations(images[i], annotations[i], coco)

#Adding paths to visualise the image data
image_dir = "data/coco_subset"  # Path to your image directory
annotation_path = "../annotations/instances_train2017.json"  # Path to annotations (@ '..' insert your file path) 

# Load annotations and images
coco, images, annotations = load_coco_annotations(annotation_path, image_dir)

# Display the first 20 images and their annotations
display_multiple_images(coco, images, annotations, num_images=20) #(num_images= is your choice)

