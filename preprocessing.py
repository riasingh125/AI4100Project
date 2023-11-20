import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import Bunch

def resize_images(images, target_size=(64, 64)):
    resized_images = []
    for image in images:
        resized_image = cv2.resize(image, target_size)
        resized_images.append(resized_image)
    return resized_images

def load_data(data_path):
    images = []
    labels = []

    for filename in os.listdir(data_path):
        if filename.endswith(".mp4"):
            video_path = os.path.join(data_path, filename)
            cap = cv2.VideoCapture(video_path)

            # Capture the first frame
            ret, frame = cap.read()
            if not ret:
                continue

            # Resize the frame
            resized_frame = cv2.resize(frame, (64, 64))

            # Append the resized frame to the images list
            images.append(resized_frame)

            # Extract the label from the filename (assuming filename format: label_word.mp4)
            label = filename.split("_")[0]
            labels.append(label)

            cap.release()

    return images, labels

def preprocess_data(data_path):
    # Load data
    images, labels = load_data(data_path)

    # Resize images
    resized_images = resize_images(images)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(resized_images, labels, test_size=0.2, random_state=42)

    # Encode labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    # Create a Bunch object to store the preprocessed data
    data = Bunch(
        data=X_train,
        target=y_train_encoded,
        target_names=list(label_encoder.classes_),
        images=X_test,
        labels=y_test_encoded
    )

    return data


data_path = "DataCSV.csv"
preprocessed_data = preprocess_data(data_path)
