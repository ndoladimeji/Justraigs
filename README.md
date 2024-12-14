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

The algorithm utilizes deep learning techniques to analyze fundus images for glaucoma screening. It employs a ResNet34 architecture pretrained on ImageNet for feature extraction.- For Task 1 (binary classification), the algorithm predicts whether a fundus image exhibits signs indicative of glaucoma, helping to identify cases requiring referral for further examination.

- For Task 2 (multi-label classification), the algorithm predicts the presence or absence of ten specific features related to glaucoma, providing detailed information about the characteristics observed in fundus images of referable glaucoma cases.

- Both tasks involve training the model on a large dataset annotated with labels for referable glaucoma and additional glaucomatous features.
- Inputs and Outputs:
Inputs:
Fundus images: The algorithm accepts fundus images as input, which are standardized and preprocessed using transformations such as resizing, cropping, and normalization before being fed into the model.

Outputs:
Task 1:

- Binary classification output: The algorithm produces a binary output indicating whether the fundus image is classified as "referable glaucoma" or "no referable glaucoma".
Task 2:

- Multi-label classification output: The algorithm outputs predictions for the presence or absence of ten specific glaucomatous features in the fundus image, providing detailed information about the characteristics observed in cases classified as "referable glaucoma".
- Validation and Performance
Validation and performance

| Metric                              | Value  |
|-------------------------------------|--------|
| Average Modified Hamming Loss       | 0.1327 |
| Sensitivity (at Specificity 95%)    | 0.3242 |
| Specificity                         | 0.95   |
| Area Under the ROC Curve (AUC)      | 0.72   |
| True Positive Rate (TPR)            | 0.4414 |
| False Positive Rate (FPR)           | 0.3561 |

Uses and Directions
This algorithm was developed for research purposes only.

Warnings
Potential Risks and Inappropriate Settings
Overreliance on Algorithm Outputs: Users should avoid overreliance on the algorithm outputs without considering clinical context and additional patient information. Relying solely on algorithm predictions may lead to diagnostic errors or inappropriate treatment decisions.

Inadequate Data Quality: The algorithm's performance may be compromised if applied to datasets with poor quality, incomplete, or inaccurate data. Users should ensure that input data meet quality standards and are representative of the target population.

Unintended Biases: The algorithm may inadvertently perpetuate biases present in the training data, leading to disparities in diagnosis or treatment recommendations across demographic or clinical subgroups. Careful consideration should be given to potential biases, and efforts to mitigate them should be undertaken.

Limited Generalizability: The algorithm's performance may vary across different patient populations or clinical settings. Users should be cautious when applying the algorithm to populations or conditions not adequately represented in the training data.

Algorithmic Errors: Despite rigorous validation, the algorithm may still exhibit errors or limitations in certain scenarios. Users should remain vigilant for unexpected outcomes and exercise clinical judgment in such cases.

Security and Privacy Concerns: Users should adhere to data privacy and security protocols when utilizing the algorithm, especially when handling sensitive patient information. Measures should be taken to safeguard patient privacy and prevent unauthorized access to data.

Regulatory Compliance: Users should ensure compliance with relevant regulatory requirements and standards when deploying the algorithm in clinical practice. Failure to adhere to regulatory guidelines may result in legal or ethical consequences.

Algorithmic Updates: Updates or modifications to the algorithm may affect its performance or recommendations. Users should stay informed about algorithm updates and follow best practices for implementation and validation of new versions.

Patient Safety: The primary concern should always be the safety and well-being of patients. Users should prioritize patient safety and consider the potential impact of algorithm recommendations on patient care outcomes.

Clinical Consultation: The algorithm is not a substitute for professional medical advice or consultation. Users should seek guidance from qualified healthcare professionals when interpreting algorithm outputs or making clinical decisions.

Common Error Messages
Common error messages
| Error Message                           | Solution                                                                 |
|-----------------------------------------|--------------------------------------------------------------------------|
| File Not Found Error                    | Ensure that the file path provided to the algorithm is correct and the file exists at that location.                                   |
| Input Data Format Error                 | Verify that the input data format adheres to the expected structure and data types specified by the algorithm.                           |
| Out of Memory Error                     | Reduce the batch size or use a machine with higher memory capacity to accommodate the algorithm's memory requirements.                 |
| Invalid Input Range Error               | Check if the input data falls within the valid range specified by the algorithm and adjust it accordingly.                              |
| Model Loading Error                     | Verify that the model file path is correct and the model file is accessible. Check for any issues with model file integrity.          |
| GPU Memory Overflow Error               | Reduce the batch size or utilize a machine with a GPU having larger memory capacity. Ensure that other processes do not occupy GPU memory.|
| Unsupported Data Type Error             | Convert the input data to a supported data type compatible with the algorithm's requirements.                                          |
| Dependency Version Conflict             | Check for conflicting dependencies or version mismatches and resolve them by updating or downgrading the dependencies accordingly.     |
| Network Connection Error                | Ensure that the device running the algorithm has an active and stable network connection to access any required external resources.    |
| Permission Denied Error                 | Grant necessary permissions to access files, directories, or resources required by the algorithm.                                    |



