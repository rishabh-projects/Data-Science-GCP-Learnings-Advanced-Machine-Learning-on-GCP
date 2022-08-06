## Advanced Machine Learning on GCP
### About
##### Analytics Cloud Native ML DL AWS Azure GCP Advanced Machine Learning on Google Cloud
##### Here we focus on advanced machine learning using Google Cloud Platform where we will get hands-on experience optimizing, deploying, and scaling production ML models of various types in hands-on labs. 
##### This teaches us how to build scalable, accurate, and production-ready models for structured data, image data, time-series, and natural language text. It ends with building recommendation systems.

### Here we will do End-to-End Machine Learning with TensorFlow on Google Cloud Platform. Then we learn about the components and best practices of a high-performing ML system in production environments.
#### End-to-End Machine Learning with TensorFlow on Google Cloud
1. Machine Learning (ML) on Google Cloud Platform (GCP)
  a. Effective ML
  b. Fully managed ML
2. Explore the Data
  a. Exploring the dataset
  b. BigQuery
  c. BigQuery and AI Platform Notebooks
  d. Getting started with GCP and Qwiklabs
3. Create the dataset
  a. Creating a dataset
4. Build the Model
  a. Create TensorFlow model
5. Operationalize the model
  a. Operationalizing the model
  b. Preprocessing with Cloud Dataflow
6. Cloud AI Platform
  a. Train and deploy with Cloud AI Platform
  b. BigQuery ML
  c. Deploying and Predicting with Cloud AI Platform
  d. Building an App Engine app to serve ML predictions

#### Production Machine Learning Systems
###### 1. Architecting Production ML Systems
  ###### a. Architecting ML systems
  ###### b. Data extraction, analysis, and preparation
  ###### c. Model training, evaluation, and validation
  ###### d. Trained model, prediction service, and performance monitoring
  ###### e. Training design decisions
  ###### f. Serving design decisions
  ###### g. Designing from scratch
  ###### h. Using Vertex AI
  ###### i. Structured data prediction
  ###### j. Getting Started with GCP and Qwiklabs
  ###### k. Structured data prediction using Vertex AI Platform
  ###### 2. Designing Adaptable ML Systems
###### Adapting to data
###### Changing distributions
###### Right and wrong decisions
###### System failure
###### Concept drift
###### Actions to mitigate concept drift
###### TensorFlow data validation
###### Components of TensorFlow data validation
###### Advanced Visualizations with TensorFlow Data Validation
###### Mitigating training-serving skew through design
###### Serving ML Predictions in Batch and Real Time
###### Diagnosing a production model
###### 3. Designing High-Performance ML Systems
###### Training
###### Predictions
###### Why distributed training is needed
###### Distributed training architectures
###### TensorFlow distributed training strategies
###### Mirrored strategy
###### Multi-worker mirrored strategy
###### TPU strategy
###### Parameter server strategy
###### Distributed Training with Keras
###### Distributed Training using GPUs on Cloud AI Platform
###### Training on large datasets with tf.data API
###### TPU-speed Data Pipelines
###### Inference
###### 4. Building Hybrid ML Systems
###### Machine Learning on Hybrid Cloud
###### Kubeflow
###### Kubeflow Pipelines with AI Platform
###### TensorFlow Lite
###### Optimizing TensorFlow for mobile


This section offers a look at different strategies for building an image classifier using convolutional neural networks. You will improve a machine learning model's accuracy with augmentation, feature extraction, and fine-tuning hyperparameters while trying to avoid overfitting the data. You will also look at practical issues that arise, for example, when you don’t have enough data; as well as how to incorporate the latest research findings into our models. You will get hands-on practice building and optimizing your own image classification models on a variety of public datasets. The section next introduces sequence models and their applications, including an overview of sequence model architectures and how to handle inputs of variable length. You will get hands-on practice building and optimizing your own text classification and sequence models on a variety of public datasets.

Image Understanding with TensorFlow on GCP
1.	Welcome to Image Understanding with TensorFlow on GCP
a.	Images as Visual Data
b.	Structured vs Unstructured Data
c.	Getting started with GCP and Qwiklabs
2.	Linear and DNN Models
a.	Linear Models
b.	Linear Models for Image Classification
c.	Image Classification with a Linear Model
d.	DNN Models Review
e.	DNN Models for Image Classification
f.	Image Classification with a Deep Neural Network Model
g.	DNNs with Dropout Layer for Image Classification
3.	Convolutional Neural Networks (CNNs)
a.	Understanding Convolutions
b.	CNN Model Parameters
c.	Working with Pooling Layers
d.	Implementing CNNs with TensorFlow
e.	Creating an Image Classifier with a Convolutional Neural Network
f.	Image Classification with a CNN Model
g.	Creating an Image Classifier with a Convolutional Neural Network
4.	Dealing with Data Scarcity
a.	The Data Scarcity Problem
b.	Data Augmentation
c.	Implementing image augmentation
d.	Image Augmentation in TensorFlow
e.	Transfer Learning
f.	Implementing Transfer Learning
g.	Image Classification Transfer Learning with Inception v3
h.	No Data, No Problem
5.	Going Deeper Faster
a.	Batch Normalization
b.	Residual Networks
c.	Accelerators (CPU vs GPU, TPU)
d.	TPU Estimator
e.	Neural Architecture Search
6.	Pre-built ML Models for Image Classification
a.	Pre-built ML Models
b.	Cloud Vision API
c.	Vision API
d.	AutoML Vision
e.	AutoML
f.	AutoML Architecture
g.	Training with Pre-built ML Models using Cloud Vision API and AutoML

7.	Sequence Models for Time Series and Natural Language Processing on Google …
1.	Working with Sequences
8.	Sequence data and models
9.	From sequences to inputs
10.	Modeling sequences with linear models
11.	Getting started with GCP and Qwiklabs
12.	using linear models for sequences
13.	Time Series Prediction with a Linear Model
14.	Modeling sequences with DNNs
15.	using DNNs for sequences
16.	Time Series Prediction with a DNN Model
17.	Modeling sequences with CNNs
18.	using CNNs for sequences
19.	Time Series Prediction with a CNN Model
20.	The variable-length problem
2.	Recurrent Neural Networks
21.	Introducing Recurrent Neural Networks
22.	How RNNs represent the past
23.	The limits of what RNNs can represent
24.	The vanishing gradient problem
3.	Dealing with Longer Sequences
25.	LSTMs and GRUs
26.	RNNs in TensorFlow
27.	Time series prediction:end-to-end (rnn)
28.	Time Series Prediction with a RNN Model
29.	Time series prediction:end-to-end (rnn)
30.	Deep RNNs
31.	Time series prediction:end-to-end (rnn2)
32.	Time Series Prediction with a Two-Layer RNN Model
33.	Improving our Loss Function
34.	Working with Real Data
35.	Time Series Prediction - Temperature from Weather Data
36.	An RNN Model for Temperature Data
4.	Text Classification
37.	Working with Text
38.	Text Classification
39.	Selecting a Model
40.	Text Classification
41.	Text Classification using TensorFlow/Keras on AI Platform
42.	Text Classification
43.	Python vs Native TensorFlow
44.	Text Classification with Native TensorFlow
5.	Reusable Embeddings
45.	Historical methods of making word embeddings
46.	Modern methods of making word embeddings
47.	Introducing TensorFlow Hub
48.	Evaluating a pre-trained embedding from TensorFlow Hub
49.	Using pre-trained embeddings with TensorFlow Hub
50.	TensorFlow Hub
51.	Using TensorFlow Hub within an estimator
6.	Encoder-Decoder Models
52.	Introducing Encoder-Decoder Networks
53.	Attention Networks
54.	Training Encoder-Decoder Models with TensorFlow
55.	Introducing Tensor2Tensor
56.	Cloud poetry:Training custom text models on Cloud ML Engine
57.	Text generation using tensor2tensor on Cloud AI Platform
58.	Cloud poetry:Training custom text models on Cloud ML Engine
59.	AutoML Translation
60.	Dialogflow

61.	In this section, you'll apply your knowledge of classification models and embeddings to build a ML pipeline that functions as a recommendation engine.

62.	Recommendation Systems with TensorFlow on GCP
1.	Recommendation Systems Overview
63.	Getting started with GCP and Qwiklabs
64.	Types of Recommendation Systems
65.	Content-Based or Collaborative
66.	Recommendation System Pitfalls
2.	Content-Based Recommendation Systems
67.	Content-Based Recommendation Systems
68.	Similarity Measures
69.	Building a User Vector
70.	Making Recommendations Using a User Vector
71.	Making Recommendations for Many Users
72.	Create a Content-Based Recommendation System
73.	Content-Based Filtering by Hand
74.	Using Neural Networks for Content-Based Recommendation Systems
75.	Content-Based Filtering using Neural Networks
76.	Create a Content-Based Recommendation System Using a Neural Network
3.	COLLABORATIVE FILTERING RECOMMENDATION SYSTEMS
77.	Types of User Feedback Data
78.	Embedding Users and Items
79.	Factorization Approaches
80.	The ALS Algorithm
81.	Preparing Input Data for ALS
82.	Creating Sparse Tensors For Efficient WALS Input
83.	Instantiating a WALS Estimator:From Input to Estimator
84.	Instantiating a WAL Estimator:Decoding TFRecords
85.	Instantiating a WALS Estimator:Recovering Keys
86.	Instantiating a WALS Estimator:Training and Prediction
87.	Collaborative Filtering with Google Analytics Data
88.	Collaborative Filtering on Google Analytics data
89.	Issues with Collaborative Filtering
90.	Productionized WALS Demo
91.	Cold Starts
4.	Neural Networks for Recommendation Systems
92.	Hybrid Recommendation Systems
93.	Designing a Hybrid Recommendation System
94.	Designing a Hybrid Collaborative Filtering Recommendation System
95.	Designing a Hybrid Knowledge-based Recommendation System
96.	Building a Neural Network Hybrid Recommendation System
97.	Neural network hybrid recommendation system on Google Analytics
98.	Building a Neural Network Hybrid Recommendation System
99.	Context-Aware Recommendation Systems
100.	Context-Aware Algorithms
101.	Contextual Postfiltering
102.	Modeling Using Context-Aware Algorithms
103.	YouTube Recommendation System Case Study:Overview
104.	YouTube Recommendation System Case Study:Candidate Generation
105.	YouTube Recommendation System Case Study:Ranking
5.	Building an End-to-End Recommendation System
106.	Architecture Overview
107.	Cloud Composer Overview
108.	Cloud Composer:DAGs
109.	Cloud Composer:Operators for ML
110.	Cloud Composer:Scheduling
111.	Cloud Composer:Triggering Workflows with Cloud Functions
112.	Cloud Composer:Monitoring and Logging
113.	End-to-End Recommendation System
114.	End to End Recommendation System

