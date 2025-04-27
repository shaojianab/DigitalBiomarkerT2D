# DigitalBiomarkerT2D
This project contains the code and example dataset that used to derive a digital biomarker of Type 2 diabetes from continuous glucose data.
# Requirements
All analyses were done using Python (version 3.7) with the following libraries: keras (version 2.2.4); tensorflow (version 2.4.0); pandas (version 0.24.2); seaborn (version 0.9.0); sklearn (version 0.21.2), and R (version 3.6.2) with the following libraries: pROC (version 1.16.2); glmnet (version 4.1).
# Model training
The code/trainModel.py trains a deep learning model based on a 1D ResNet architecture to classify diabetes status (normal or diabetes) using CGM (Continuous Glucose Monitoring) data.
It reads training, validation, and test datasets, preprocesses them, builds and trains a model, evaluates it, and saves it for future use.

