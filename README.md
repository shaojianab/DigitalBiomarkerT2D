# DigitalBiomarkerT2D
This project contains the code and example dataset that used to derive a digital biomarker of Type 2 diabetes from continuous glucose data.
# Requirements
All analyses were done using Python (version 3.7) with the following libraries: keras (version 2.2.4); tensorflow (version 2.4.0); pandas (version 0.24.2); seaborn (version 0.9.0); sklearn (version 0.21.2), and R (version 3.6.2) with the following libraries: pROC (version 1.16.2); glmnet (version 4.1).
# Model training
The code/trainModel.py trains a deep learning model based on a 1D ResNet architecture to classify diabetes status (normal or diabetes) using CGM (Continuous Glucose Monitoring) data.
It builds a one-dimensional ResNet architecture to classify whether a person is normal or diabetic. When the script starts, it loads three datasets named train.csv, dev.csv, and test.csv, and normalizes specific columns by dividing them by 18. It then separates the feature data and the label, where the label column is named diabtype. The labels are transformed into a one-hot encoding format to be suitable for classification tasks.
The model takes an input shape of 96 time steps with 1 feature per step. It is built using a ResNet structure consisting of convolutional layers, batch normalization, ReLU activation functions, residual connections, and dropout for regularization. The final output layer applies a softmax activation to predict two classes: normal or diabetes. After training, the model is saved to the directory model/ under the name model.hdf5. 
# Model testing


