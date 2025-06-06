# DigitalBiomarkerT2D
This project contains the code and example dataset that used to derive a digital biomarker of Type 2 diabetes from continuous glucose data.
# Requirements
All analyses were done using Python (version 3.7) with the following libraries: keras (version 2.2.4); tensorflow (version 2.4.0); pandas (version 0.24.2); seaborn (version 0.9.0); sklearn (version 0.21.2), and R (version 3.6.2) with the following libraries: pROC (version 1.16.2); glmnet (version 4.1).
# Model training
The code/trainModel.py trains a deep learning model based on a 1D ResNet architecture to classify diabetes status (normal or diabetes) using CGM (Continuous Glucose Monitoring) data.

It builds a one-dimensional ResNet architecture to classify whether a person is normal or diabetic. When the script starts, it loads three datasets named train.csv, dev.csv, and test.csv, and normalizes specific columns by dividing them by 18. It then separates the feature data and the label, where the label column is named diabtype. The labels are transformed into a one-hot encoding format to be suitable for classification tasks.

The model takes an input shape of 96 time steps with 1 feature per step. It is built using a ResNet structure consisting of convolutional layers, batch normalization, ReLU activation functions, residual connections, and dropout for regularization. The final output layer applies a softmax activation to predict two classes: normal or diabetes. After training, the model is saved to the directory model/ under the name model.hdf5. 
# Model testing
The code/testModel.py script is used to load the pre-trained model and evaluate its performance on a test dataset. First, it loads the trained model from the file model/model.hdf5. Then, it reads the test dataset from a CSV file (data/test.csv). 

The script then transforms the labels into a one-hot encoded format using the transy function, which converts the binary labels into two classes: [1, 0] for normal and [0, 1] for diabetes. Next, the features are reshaped to match the model's expected input shape, as the model was likely trained with 3D data (samples, features, 1).After preprocessing the data, the script uses the loaded model to make predictions on the test data. The model outputs probabilities for each class, and the predictions are rounded to get binary class labels (either 0 or 1). The predictions are reshaped to match the expected format for evaluation.Finally, the script calculates the classification report using classification_report from scikit-learn, which includes metrics like precision, recall, F1-score, and support for both classes. This report is printed out so that you can assess how well the model performs on the test dataset.

To use the script, make sure the test data is properly formatted, and the pre-trained model is saved as model.hdf5 in the model/ directory. After that, simply run the script, and it will output the classification report with performance metrics on the test dataset.
# Model validation
The code/validationModel.py is used to evaluate the pre-trained model on a validation dataset. It begins by loading the pre-trained model saved as model/model.hdf5. Then, it reads the validation dataset from data/validationDataset.csv. 

Next, the script reshapes the feature data into a 3D array, which is necessary for the model's input format. It then uses the model to predict probabilities for each sample in the test set. The probabilities are rounded to get binary predictions (either 0 or 1). These predictions are reshaped into the required format.The script then calculates the classification report using classification_report from scikit-learn, which provides various performance metrics such as precision, recall, F1-score, and support for both classes (normal and diabetes). The classification report is printed, showing how well the model performed on the validation dataset.

To use this script, you need to have the validation dataset in the correct format (data/validationDataset.csv) and the pre-trained model saved in the model/model.hdf5 file. Once these files are ready, running the script will generate a classification report based on the performance of the model on the validation dataset.
