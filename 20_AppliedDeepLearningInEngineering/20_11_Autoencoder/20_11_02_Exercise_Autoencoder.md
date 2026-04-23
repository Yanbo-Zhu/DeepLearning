


# 1 Problem 1: Data preparation (at home)
Objective: Students prepare the Fruits Data set for anomaly detection.
(a) Go back to exercise No. 6 and copy the banana data set along with the data set loader functions.
(b) Load the banana data set and validate shapes of the arrays
(c) Obtain the indices of all class A banana samples in the training and testing set. Type A corresponds to the highest-quality banana samples, while types B and C form lower quality fruits. The type A examples will form the negative class for this exercise.
(d) Obtain all positive samples (class B and class C samples) from the training and test set and join them into a separate data set. The result is a training set containing only type A samples, a test set containing only type A samples, and a data set containing all type B and type C images.
(e) Visualize samples from the negative and positive class.

# 2 Problem 2: Implementation of a Convolutional Autoencoder

Objective: Students implement both the encoder and decoder of the autoencoder using the Keras Sequential API. The encoder will reduce the spatial dimensions of the input image, while the decoder will reconstruct the image back to its original size using transpose convolutions. For a blueprint, students can revisit the solution to the U-Net exercise solution in exercise 08.

(a) Encoder:
• Use Conv2D layers with increasing filter sizes.
• Apply MaxPooling2D to reduce spatial dimensions.
• The depth of the encoder should be defined as a parameter, where the first layer starts with a base number of filters (e.g., 32) that may increase along the depth.

(b) Decoder:
• Use Conv2DTranspose layers (transpose convolutions) for upsampling.
• Apply Conv2D layers to reconstruct the image.
• The number of filters in the decoder should decrease symmetrically to the encoder.

(c) Compile the model with the Adam optimizer. Use Mean Squared Error (MSE) as the loss function since you are performing a regression task (reconstruction of pixel values).

(d) Print the model summary after defining both encoder and decoder.

# 3 Problem 3: Model Training and Anomaly Detection
Objective: Students train the autoencoder and tune the decision threshold to detect anomalies.

(a) After training, visualize the original and reconstructed images.
(b) Show at least 5 original images along with their reconstructed counterparts: Use matplotlib to plot the original and reconstructed images in a grid.
(c) Compute the mean reconstruction error per image. Display histograms of the reconstruction errors for the training set (negative samples), test set (negative samples), and validation set (positive samples).
(d) Set a decision threshold above which one would detect an anomaly.
(e)Binarize the predictions on the testset(negatives) and the validation set (positives) according to the chosen threshold, and compute the resulting classification F1 score. Optionally: plot the PRC through varying the decision threshold.


![](image/Pasted%20image%2020260208135455.png)

![](image/Pasted%20image%2020260208135502.png)




