# Justraigs
![ResNet34 Architecture](https://github.com/user-attachments/assets/bbfc4d60-31a7-4a8b-b50a-1d3e8586645b)

Our algorithm is a deep learning-based approach developed for the Justified Referral in AI Glaucoma Screening (JustRAIGS) challenge.
It consists of two tasks:

- Task 1 involves binary classification of fundus images into "referable glaucoma" (RG) and "no referable glaucoma" (NRG) classes, while
- Task 2 focuses on multi-label classification of ten additional features related to glaucoma.

The algorithm was developed using PyTorch and leverages the ResNet34 architecture pretrained on ImageNet for feature extraction.

For Task 1, the algorithm utilizes a ResNet34 network with a linear classifier to predict the presence or absence of glaucomatous signs in fundus images. Class imbalance is addressed by applying a weighted binary cross-entropy loss function during training.

For Task 2, the same ResNet34 architecture is employed, but an additional classifier is added to predict the presence or absence of specific glaucomatous features in fundus images. Each feature is treated as a separate binary classification task, and binary cross-entropy loss is applied independently for each feature during training.

The algorithm was trained on a large dataset provided for the challenge, consisting of over 110,000 annotated fundus photographs collected from about 60,000 screenees. The training process involved multiple epochs of training using Adam optimizer with a learning rate of 0.0005.

Overall, our algorithm aims to accurately identify fundus images associated with referable glaucoma and provide detailed information about specific glaucomatous features, thereby assisting in early glaucoma detection and referral decision-making.

Mechanism
Mechanism
- Target Population:

Our algorithm targets individuals undergoing glaucoma screening, particularly those at risk of or suspected to have glaucoma. This includes individuals with family history, older age, or other risk factors predisposing them to glaucoma.

    Algorithm Description:

The algorithm utilizes deep learning techniques to analyze fundus images for glaucoma screening. It employs a ResNet34 architecture pretrained on ImageNet for feature extraction.
