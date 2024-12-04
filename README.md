# Proj_Boundingbox
This repository contains the code and dataset for a group project being developed as part of the final assessment for Neural Networks and Deep Learning.

# Team Members
Isha Paliwal (G49146952)
Harsha Talapaka (G46330983)
Needhi Kore (G20475943)

# AIM
The aim of the project is to Implement a CNN model using a deep learning library that can determine the bounding box for set objects. 

# Procedure
After selecting the dataset: https://cocodataset.org/#download

From the dependencies and dataset file, you can infer the steps one step at time on how we imported the dependencies and how to download the correct data to be able to complete the project.

This is optional but if you wish to see your dataset before training then you can use the Visualization code file to visualise the dataset without using any models for training.

Then went into preprocesing the Data, you can learn more going through the document on how it was done. 

We then build our basic and custom cnn model from Scratch, without any libraries, and then complied the model and ran the fitted data in it.

After, fitting it and visualising the results, the model was saved into our local so that we could make predictions on any given image.

As mentioned, I have been able to run the project on streamlit successfully, since the model file was large and wasnt been able to be uploaded on github, I would be demonstrating on how it can be used below.

# Method Explanation
As, you may think after going through the Dependencies & Dataset download file on why we did the following and why we used only 10,000 images to train but downloaded all the annotations?
This is because as you read through the COCO dataset file you understand that the annotations file consist of the object bounding boxes, segmentation masks, and captions for each image downloading all of the annotations and training on the fewer pictures also helps build a good model though it maybe not be able to predict everything after training due to the lack of images but it atleast tries to guess what the image might be.

Since, the aim of my project is to implement a CNN model using a deep learning library that can determine the bounding box for set objects, I have limited my image count to 10K, to be able to train on whatever is available in the 10K images present but the model will be trained on all the annotations.

You can refer to the following links for more clarification:
https://viso.ai/computer-vision/image-annotation/?ref=labellerr.com#:~:text=Objects%20can%20be%20annotated%20within,recognize%20present%20objects%20as%20persons. 
https://www.labellerr.com/blog/exploring-the-coco-dataset/#:~:text=In%20addition%20to%20object%20annotations,activities%20depicted%20in%20the%20images.
https://docs.ultralytics.com/datasets/detect/coco/

The idea of being able to do it with limited amount of data stemmed from the following discussion: https://stackoverflow.com/questions/60227833/how-to-filter-coco-dataset-classes-annotations-for-custom-dataset, when we found it difficult to download such a large dataset.

# Dataset Download
Inorder to download the annotations file or any dataset file follow this link: https://gist.github.com/mkocabas/a6177fc00315403d31572e17700d7fd9 , this is the link from where we downloaded the annotations 

In our case, we used the pycocotools library to interact with the COCO annotations, specifically the instances_train2017.json file. This file contains metadata for the images, including their URLs. We then used this metadata to download the image dataset.

# Predicting Results & Implementing model on streamlit.
Once you have been able to run the model sucessfully and have been able to run it inorder to make predictions for the same, create a file on VScode ( I suggest VScode because there are other dependencies that you need to download for anaconda, or jupyter notebook.

Once you have done creating a file, load the model and then as seen in the Prediction.py file load your model.
After loading your model, write a function to preprocess any new data that you want to predict. We preprocess the data inorder to satisfy the models ability to take image sizes of 224, 224 pixels
After resizing your images, write a function to visualise your prediction. For which in your function, load the image, use the cnn model to predict it and denormalise the image after predicting it and extract the co-ordinates.
You can use any shape to draw a bounding box on the image based on your comfort, I used a rectangle. You can use this link to do the same: https://www.scaler.com/topics/cv2-rectangle/

To run it on streamlit, After having written the prediction.py file with the needed as mentioned, write your own UI for the streamlit app: https://docs.vultr.com/how-to-deploy-a-deep-learning-model-with-streamlit

Finally you can run it on your local IP and be able to use to add any images you want to see the bounding boxes.
