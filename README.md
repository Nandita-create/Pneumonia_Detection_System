# Pneumonia_Detection_System
A trained model to detect whether a person has pneumonia or not, by analysing a chest radiograph

First, the user uploads a chest radiograph.
Once it has been uploaded the system checks if the radiograph clear shows the pair of lungs.
If it is clear enough, it moves on to processing the image, extracting regions and comparing it with the trained model to understand if the person is affected by pneumonia or not.
The model was initially trained using a radiograph dataset from Kaggle which included x-rays of affected and normal lungs.

Support Vector Machine (SVM) has the best accuracy in this case.
Confusion matrix of the SVM:
<img width="690" height="567" alt="image" src="https://github.com/user-attachments/assets/ba90131a-3ee9-4ec2-8321-65af7e764d2d" />


The following are 2 possible outputs
1. Person has Pneumonia
<img width="1197" height="792" alt="image" src="https://github.com/user-attachments/assets/4b152b99-ce3b-440b-a24f-291dd130f863" />
<img width="1080" height="876" alt="image" src="https://github.com/user-attachments/assets/5251d5c5-c1ad-440e-9c89-9fb4ec77b802" />


2. Person does not have Pneumonia
<img width="1060" height="717" alt="image" src="https://github.com/user-attachments/assets/3ef3bdc4-2c81-40d4-9c10-e81cc56250fe" />
<img width="1013" height="877" alt="image" src="https://github.com/user-attachments/assets/0af58bfb-c511-4ab7-be22-ca3f10333253" />

This system is an academic project and is not a medical diagnostic tool.
