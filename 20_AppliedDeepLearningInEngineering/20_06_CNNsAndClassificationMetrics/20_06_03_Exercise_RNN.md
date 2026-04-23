


# 1 Problem 1: Data Loading and Preprocessing (at home)
Objective: Load and prepare image data (MNIST or Fruit dataset) for training a convolutional neural net (CNN).
(a) Load either the MNIST (see Exercise 00) or Fruit dataset.
(b) For the Fruit data set, follow these instructions:
    (a) Download data set from here and unpack to your local working directory.
    (b) Get load_fruits.py from ISIS to use the load_fruit_dataset function for reading (and down- sampling) the fruits data
    (c) Use the following code to load the (banana) data from the unpacked directory:
(c) Preprocess the input data for CNNs:
    • Reshape the data sets of images to include a channel dimension, resulting in an array of shape
    (n_samples, height, width, n_channels) . 
    • Normalize the pixel values to the range `[0, 1]`.
(d) Perform a stratified split into training, validation, and test sets to maintain class balance across subsets using sklearn . Use a ratio of 70% training, 15% validation, and 15% testing.
(e) Convert all class labels to one-hot encoded vectors using sklearn.preprocessing.OneHotEncoder . The resulting target vectors are binary vectors with a single true entry identifying the positive class, and false entries for all other negative classes. Verify that the number of output classes matches the dimensionality of the one-hot encoded targets of shape n_samples, n_classes .


![](image/Pasted%20image%2020260208162202.png)


# 2 Problem 2: Implementing and Training a CNN Image Classifier


Objective: Design, train, and evaluate a CNN model using the Keras Sequential API.
(a) Define a CNN architecture for image classification following the lecture slides. The model should include:
    • An Input layer
    • One or more convolutional layers ( Conv2D ) with ReLU activations,
    • Pooling layers ( MaxPooling2D ) for spatial downsampling,
    • A Flatten layer followed by one or more Dense layers for classification.
(b) Choose an appropriate loss function and output activation:
• Use categorical_crossentropy for multi-class classification. 
• Use softmax activation in the output layer.
(c) Print the model summary using model.summary() and determine: • The total number of trainable parameters,
• The total number of trainable parameters,
•The fraction of parameters in the feature extractor(convolutional blocks) versus the classification head.
(d) Train the CNN for a reasonable number of epochs. Plot the training and validation losses using the history object from model.fit() . Optionally, implement an EarlyStopping callback to prevent over-fitting.


# 3 Problem 3: Comparison with a Fully Connected Image Classifier
Objective: Compare the CNN to a fully connected (dense) feed-forward network in terms of performance and parameter efficiency. Consider the fully connected feed-forward neural network from Exercise 04 that operates directly on flattened image inputs.
(a) Train the dense model using the same training, validation, and test data splits as in Problem 2. Plot and compare the training and validation loss curves for both models.
(b) Evaluate both models on the test set using model.evaluate() and compare their classification accura- cies. Discuss the differences in performance, generalization behavior, and the number of trainable param- eters.
(c) Reflect on the benefits of convolutional architectures in terms of feature extraction, spatial structure utiliza- tion, and parameter efficiency relative to dense networks.
