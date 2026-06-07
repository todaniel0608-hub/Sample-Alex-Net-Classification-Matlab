# Sample-Alex-Net-Classification-Matlab
An AlexNet Convolutional Neural Network built in MATLAB for custom image dataset classification

# AlexNet-Inspired Image Classification System (MATLAB)

Project Overview:
This project implements a deep Convolutional Neural Network (CNN) in MATLAB based on the classic AlexNet architecture. The script provides training for a custom image classifier, featuring automated data augmentation, register-level layer design, validation tracking, and performance evaluation via confusion matrices.

Network Architecture:
The network is built from scratch using 25 layers configured for optimal spatial feature extraction:
* Image Input: Standard 227x227x3 RGB configuration.
* Convolutional Layers: 5 distinct 2D convolution layers utilizing varied filter sizes (11x11, 5x5, and 3x3) and custom feature maps (up to 384 channels).
* Regularization & Activation: Employs Rectified Linear Units (ReLU) for non-linearity, Cross-Channel Normalization for local response scaling, and Dropout layers (50%) to aggressively mitigate overfitting.
* Classification Output: Dynamically scales the final fully connected layer to match the unique class count of the input dataset, utilizing a Softmax layer for probability mapping.

Technical Features & Deep Learning Concepts:
* Dynamic Data Datastores: Uses MATLAB `imageDatastore` to handle large collections of images efficiently without overloading system memory.
* On-the-Fly Preprocessing & Augmentation: Implements an automated data augmenter that applies random horizontal reflections and spatial translations (pixel shifts) to expand the training footprint and improve model generalization.
* Hyperparameter Tuning: Optimized using Stochastic Gradient Descent with Momentum (SGDM) with a structured initial learning rate (0.001), specific mini-batch sizing (128), and automated validation checkpointing every 30 iterations.
* **Granular Metrics:** Evaluates performance by compiling a matrix chart normalized by both row (precision) and column (recall) to thoroughly evaluate classification accuracy.

Technical Stack:
* Language: MATLAB
* Toolboxes Required: Deep Learning Toolbox, Image Processing Toolbox
* Dataset Source Compatibility: Designed for custom local directories or standard structured datasets (e.g., Kaggle folder splits).
