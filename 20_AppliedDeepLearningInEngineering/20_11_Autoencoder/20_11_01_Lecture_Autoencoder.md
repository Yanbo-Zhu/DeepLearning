


Learn to ...
 Formulate an unsupervised anomaly detection model architecture 
 Find a suitable bottleneck dimension

Know about...
 The concept of a reconstruction loss
 Use cases for encoder-decoder models

Motivation: Anomaly Detection
 Spot a-priori unknown abnormal / edge cases in large data sets:



# 1 Basic 

 Unsupervised (targets = inputs ) deep learning technique for 
     Data compression (audio, image, video)
     Sparse data representations
     Generative tasks (variational autoencoders)

 Dimensionality reduction/compression by architectural design: bottleneck approach
![](image/Pasted%20image%2020260208131133.png)

---

Dimensionality Reduction

 Input 𝐱 ∈ R𝒏 from high-dimensional feature space
     May contain redundant information (dependent feature columns)
     May contain noise and artifacts that shadow underlying relations
     May carry more information than needed for a given modeling task
     May be too large to handle in specific cases (constrained hardware, real-time inference) 

 Aim: reduce dimensionality of the data while preserving the most dominant content: latent h

High precision data
 Complex model
 Slow model inference

Compressed data
 Simple model
 Fast model inference

h: Low-dimensional & compact representation of the original rich & high-dim. data

---

Intuition

 Think of it like ... stenography
     Faster to write
     Shorter (compression of characters) 
     Lossy, i.e not distinct
     Keeps meaning and content


![](image/Pasted%20image%2020260208131352.png)


---

Unsupervised Learning

![](image/Pasted%20image%2020260208132309.png)


# 2 Definition

Input: high-dimensional feature vector 𝒙 ∈ R𝒏.

Encoder: compresses input into latent space representation. Number of neurons per layer decreases with each subsequent layer.

Latent space representation: compressed 𝒙 representation 𝒉 ∈ R𝒑 of the input that defines behaviorally relevant variables. 𝑝 ≪ 𝑛

Decoder: reconstructs input from latent space representation. Number of neurons per layer increases with each subsequent layer.

Output: Lossy high-dimensional reconstruction 𝒚􏰅 ∈ R𝒏 of the input. Degradation due to compression.

![](image/Pasted%20image%2020260208132357.png)


# 3 Models


 Linear Models 
 autoencoder for 𝑋 ∈ R􏰆×􏰇, i.e. 𝑁 samples from a 𝑛-dimensional space
 
 Encoder 𝒉 = 𝑒(𝒙􏰈) with weight matrix 𝑊 ∈ R􏰊×􏰇 􏰀􏰉
 Decoder 𝒚􏰅 = 𝑑(𝒉) with weight matrix 𝑊 ∈ R􏰇×􏰊 􏰀􏰋
 𝒉 = 𝑊 𝒙􏰈, 𝒉 ∈ R􏰊 􏰉
 𝒚􏰅 = 𝑊 𝒉 , 𝒚􏰅 ∈ R 􏰇 􏰋
𝒙 𝒆 ( 𝒙 )
h 𝒅 ( 𝒉 )
𝒚􏰅 ! = 𝒙

 Training: find optimal weight matrices min 􏰌
If𝒚􏰅≈𝒙􏰈:𝑊 and𝑊 arepseudo-inverse𝑊 𝑊 ≈I 􏰉􏰋􏰋􏰉
using gradient descent

 Solution for 𝐿􏰐 norm: solution is equivalent to PCA

![](image/Pasted%20image%2020260208132559.png)

---

Nonlinear Models

 Nonlinear autoencoder for 𝑋 ∈ R􏰆×􏰇, i.e. 𝑁 samples from a 𝑛-dimensional space
 Encoder: nonlinear function 𝒉 = 𝑒(𝒙􏰈, 𝑊 ) 􏰀􏰉
Decoder:nonlinearfunction𝒚􏰅 =𝑑(𝒉,𝑊)

![](image/Pasted%20image%2020260208132644.png)


----

 Deep Models

 Deep autoencoders
 Encoder 𝑒􏰔() and decoder 𝑑􏰔() are neural networks with multiple hidden layers
 In theory, a single layer (of unknown width) would be sufficient (universal approx. Theorem), in practice we use several layers
 Decoder structure mirrors the encoder structure

![](image/Pasted%20image%2020260208132741.png)


# 4 optimal dimensionality of latent representation 𝒉


## 4.1 Sparsity 

 How to find the optimal dimensionality of 𝒉 ∈ R􏰊?
 𝑝 too small: inacceptable reconstruction error 
 𝑝 too large: suboptimal compression

 Basic concepts:
1. Grid-search (elbow method) for increasing 𝑝
2. Uselarge𝑝andasparsity-promotinglossfunction


input 𝒙
 Sparsity: fraction of zero entries in 𝒉. Number of non-zero entries: nnz(𝒉)
latent representation 𝒉
output 𝒚􏰅
loss值越小 h中0值越多
 Sparsity-prom. loss: L = min 􏰍 ∑ 𝑊 𝑊 𝒙􏰏 − 𝒙􏰈 􏰐 + 𝜆 nnz(𝒉) (very complex optimization!)
􏰀􏰋􏰉􏰀􏰀
 𝐿 regularization: L = min 􏰍 ∑ 𝑊 𝑊 𝒙􏰏 − 𝒙􏰈 􏰐 + 𝜆 𝒉 (approximation to nnz(𝒉))
(𝐿􏰑 reg. conserves convex optimization problem)


![](image/Pasted%20image%2020260208133132.png)


![](image/Pasted%20image%2020260208133159.png)

![](image/Pasted%20image%2020260208133215.png)


# 5 Autoencoders: Use Cases

 Data Compression in Communication Systems: compress data for efficient transmission in communication systems. By learning the essential information in the data, autoencoders contribute to reducing the required bandwidth while maintaining data integrity

 Anomaly Detection: learn the normal behavior of mechanical systems based on sensor data. Any deviation from this learned pattern can be flagged as an anomaly, aiding in predictive maintenance by detecting potential faults in machinery.
用 reconstructed image - input image, 如果 error 很大， 则证明图片中有异物

 Robotics and Sensor Fusion: processing data from multiple sensors on a robot, helping to learn a compact representation of the environment. This can improve decision-making and navigation capabilities in robotic systems.


## 5.1 Alnomay Detection 

 Anomaly detection: observe how well an AE can reconstruct its inputs, after it has been trained on only normal (healthy, negative class) samples. If it cannot reconstruct a given input, this input must be very dissimilar from the training samples, i.e. an anomaly
 Process
1. Measure the reconstruction error 𝐸􏰀 = 𝒚􏰅􏰀 − 𝒙􏰀 for new data point
2. Binary classification based on error threshold: anomaly if 𝐸􏰀 > 𝐸􏰕􏰖􏰗􏰉􏰘􏰖􏰙􏰚􏰋

 Example: Anomaly detection for Electrocardiograms (ECG) (Link to data set)
![](image/Pasted%20image%2020260208133618.png)

Anomaly Detection in ECG Signals 
 Inspired by TensorFlow example (Link)

![](image/Pasted%20image%2020260208133641.png)

nagative samples: 图中没有异物
positive sample: 图中有异物，  因为目的就是 探测异物， 所以 有异物的图片 就是 positive sample 

 Compute statistics of reconstruction errors 
 On training set (only negative samples) 
 On test set (contains positive samples)

 Set decision boundary to mean+1*std.dev. of training set reconstruction error
![](image/Pasted%20image%2020260208133742.png)

---


 Spot the abnormal cases in high-dimensional time series data streams: Example from the ISS
and Airbus (Link, P. Grashorn, J. Hansen, M. Rummens from Airbus) 
 Approx. 17,000 sensors in the Columbus
module, sending telemetry data @1Hz
 Approach: slicing of time series data into smaller chunks (sub-sequences)
 Anomaly detection through LSTM autoencoder (trained on historic, manually labeled data)

![](image/Pasted%20image%2020260208133945.png)

Types of Autoencoders
 Fully connected AEs: using fully connected (dense) layers (universal approach for tabular data, often also applied to unstructured data)
 Convolutional AEs: using convolutional and up-convolutional layers (mostly used for image and time series data).
 Recurrent AEs: using recurrent layers (mostly used for time series data)

![](image/Pasted%20image%2020260208134015.png)






