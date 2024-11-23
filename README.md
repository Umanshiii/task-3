# Cat vs Dog Classifier Using SVM

This project implements an image classification system to distinguish between **cats** and **dogs** using a **Support Vector Machine (SVM)** classifier. The model leverages images from the **Dogs vs Cats dataset** on Kaggle for training and testing the classifier.

## Dataset

- **Classes**: 2 (Cat, Dog).
- **Image Size**: 64x64 pixels (grayscale or RGB, depending on preprocessing).
- **Total Images**: ~25,000 images.
- You can download the dataset from [Kaggle: Dogs vs Cats](https://www.kaggle.com/c/dogs-vs-cats/data).

## Requirements

To get started, install the following libraries:

```bash
pip install numpy pandas scikit-learn opencv-python matplotlib
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/cat-dog-svm-classifier.git
   ```
2. Change directory:
   ```bash
   cd cat-dog-svm-classifier
   ```
3. Download and extract the dataset into the project folder.

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training the Model
To train the model and classify the images:

```bash
python train_model.py
```

### Evaluating the Model
To evaluate the trained model on a test set:

```bash
python evaluate_model.py
```

### Model Details

The classifier is built using **Support Vector Machine (SVM)** with the following steps:

- **Image Preprocessing**: Images are resized to 64x64 pixels and flattened to feature vectors.
- **Feature Scaling**: Features are standardized using `StandardScaler`.
- **Model Training**: An SVM model with a linear kernel is trained on the dataset.

### Results

The model achieves a classification accuracy of ~62% on the test set. The performance can be improved with more advanced image processing techniques or using different machine learning models.

