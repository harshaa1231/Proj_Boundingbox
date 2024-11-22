# Proj_Boundingbox
This repository contains the code and dataset for a group project being developed as part of the final assessment for Neural Networks and Deep Learning.

# Team Members
Isha Paliwal (G49146952)
Harsha Talapaka (G46330983)
Needhi Kore (G20475943)

# AIM
The aim of the project is to Implement a CNN model using a deep learning library that can determine the bounding box for set objects. 

# Procedure
Firstly, we were determined on using the https://cocodataset.org/#download
As can be seen in the Dependencies & Dataset download file, we then started the process of importing the required libraries.
As can be seen in the Visualization code file, we then visualized the dataset without using any models for training.
Next, we intend to use a deep learning package to create a CNN model, and we look for objects' bounding boxes surrounding it.
Lastly, by including our own image, we intend to test this model on photos other than the train file.
we will attempt to run it on streamlit and provide a demo if we can.

# Method Explanation
As, you may think after going through the Dependencies & Dataset download file on why we did the following and why we used only 10,000 images to train but downloaded all the annotations?
This is because as you read through the COCO dataset file you understand that the annotations file consist of the object bounding boxes, segmentation masks, and captions for each image downloading all of the annotations and training on the fewer pictures also helps build a good model though it maybe not be able to predict everything after training due to the lack of images but it atleast tries to guess what the image might be.

Since, the aim of my project is to implement a CNN model using a deep learning library that can determine the bounding box for set objects, I have limited my image count to 10K, to be able to train on whatever is available in the 10K images present but the model will be trained on all the annotations.

You can refer to the following links for more clarification:
https://viso.ai/computer-vision/image-annotation/?ref=labellerr.com#:~:text=Objects%20can%20be%20annotated%20within,recognize%20present%20objects%20as%20persons. 
https://www.labellerr.com/blog/exploring-the-coco-dataset/#:~:text=In%20addition%20to%20object%20annotations,activities%20depicted%20in%20the%20images.
https://docs.ultralytics.com/datasets/detect/coco/

The idea of being able to do it with limited amount of data stemmed from the following discussion: https://stackoverflow.com/questions/60227833/how-to-filter-coco-dataset-classes-annotations-for-custom-dataset, when we found it difficult to download such a large dataset.
