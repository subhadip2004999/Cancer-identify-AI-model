Cancer Prediction Model

Overview

This project uses a deep learning approach to predict cancer diagnosis based on a dataset of patient characteristics. The model is built using TensorFlow and Keras, and is trained on a labeled dataset to learn patterns and relationships between the input features and the target output.

Dataset

The dataset used for this project is the (link unavailable), which contains 569 instances of breast cancer data, each described by 30 features.

Model Architecture

The model consists of three fully connected (dense) layers:

- The first layer has 256 neurons and uses the ReLU activation function.
- The second layer also has 256 neurons and uses the ReLU activation function.
- The third layer has 1 neuron and uses the sigmoid activation function.

The model is compiled with the Adam optimizer and binary cross-entropy loss function.

Training and Evaluation

The model is trained on the labeled dataset using a batch size of 32 and 50 epochs. The model's performance is evaluated on a separate test dataset using accuracy as the metric.

Usage

To use this model, simply clone the repository and install the required dependencies. You can then use the model to make predictions on new, unseen data.

Dependencies

- TensorFlow
- Keras
- Pandas
- NumPy
- Scikit-learn

License

This project is licensed under the MIT License.
