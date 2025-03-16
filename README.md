# Deep-Learning-Powered-Image-Classifier
This project implements an AI-powered image classification system using MobileNetV2 and EfficientNetV2, two powerful deep learning architectures. The model is trained using TensorFlow and Keras, with advanced data augmentation, transfer learning, and fine-tuning techniques to improve accuracy.

Key Features

Pre-trained CNN Models: Uses MobileNetV2 (fast & lightweight) and EfficientNetV2 (high accuracy).
Custom Model Training: Option to train a new model or use an existing one.
Automated Image Classification: Reads images from a dataset or fetches them from URLs.
Data Augmentation: Enhances training with rotation, zoom, brightness, and flipping.
Adaptive Training Strategy: Adjusts batch size and validation split dynamically.
GPU/CPU Compatibility: Automatically detects and utilizes GPU if available.
Parallel Processing: Uses multi-threading for faster image classification.

Technologies Used

Python
TensorFlow & Keras – For deep learning model training
Pandas & NumPy – For data handling
OpenCV & PIL – For image processing
Requests – Fetching images from URLs
TQDM – For real-time progress tracking
Multi-threading – Optimized parallel execution

How It Works

Model Selection

The script prompts the user to choose between using an existing trained model or training a new one.
Data Preprocessing & Augmentation

If training a new model, it applies image augmentation techniques like rotation, zoom, and brightness adjustments to improve learning.
Model Training

Uses MobileNetV2 or EfficientNetV2 as a base model, applies fine-tuning, and trains the model with the given dataset.
Implements early stopping and learning rate reduction for optimal training.
Image Classification

The script processes images from local storage or image URLs (listed in an Excel file).
Classifies images using the trained model and saves predictions in a CSV file.
Parallel Processing for Speed

Uses multi-threading to process multiple images simultaneously, improving efficiency.

Use Cases

This project can be applied in various real-world scenarios, such as:

Object Recognition – Identifying objects in images.
E-commerce – Automatically categorizing product images.
Medical Image Analysis – Detecting patterns in X-rays and MRIs.
Surveillance – Recognizing objects in security footage.

Conclusion

This project demonstrates the power of deep learning and transfer learning in image classification. It is highly scalable and can be adapted for real-world AI applications. 
