import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from skimage.io import imread
from skimage.transform import resize

def load_images(folder, label, image_size=(64, 64), max_images=1000):
    images = []
    labels = []
    for i, filename in enumerate(os.listdir(folder)):
        if i >= max_images:
            break  # Load only up to max_images
        if filename.endswith(".jpg"):
            filepath = os.path.join(folder, filename)
            img = imread(filepath)
            img_resized = resize(img, image_size)
            images.append(img_resized.flatten())  # Flatten the image
            labels.append(label)
    return np.array(images), np.array(labels)

# Load the data
cat_folder = r"C:\Users\umanshi\Downloads\train\cats"
dog_folder = r"C:\Users\umanshi\Downloads\train\dogs"

cat_images, cat_labels = load_images(cat_folder, label=0)  # Label 0 for cats
dog_images, dog_labels = load_images(dog_folder, label=1)  # Label 1 for dogs

# Combine and shuffle
X = np.vstack((cat_images, dog_images))
y = np.hstack((cat_labels, dog_labels))

# Shuffle and split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

svm_model = SVC(kernel='linear', C=1)  # Use a linear kernel
svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)

# Print accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Classification report
print(classification_report(y_test, y_pred, target_names=['Cat', 'Dog']))

