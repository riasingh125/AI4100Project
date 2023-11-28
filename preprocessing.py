import os
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import Bunch
import imageio
import matplotlib.pyplot as plt

def resize_frame(frame, target_size=(100, 100)):
    return cv2.resize(frame, target_size)

def load_data(data_path):
    df = pd.read_excel(data_path, header=None, names=['video', 'word'])
    videos = []
    labels = []

    for index, row in df.iterrows():
        video_path = row['video']
        label = row['word']

        # Read the video using imageio
        video_reader = imageio.get_reader(video_path, 'ffmpeg')
        frames = [resize_frame(frame, target_size=(64, 64)) for frame in video_reader]

        videos.append(frames)
        labels.append(label)

    return videos, labels

def preprocess_data(data_path):
    # Load data
    videos, labels = load_data(data_path)

   # print("Number of video: ", len(videos))
    #print("Number of labels: ", len(labels))

   # print("first video franes: ", videos[0])
    #print("first label: ", labels[0])
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(videos, labels, test_size=5, random_state=42, stratify=labels)

    print("Number of training samples: ", len(X_train))
    print("Number of testing samples: ", len(X_test))
    #print("first training sample: ", X_train[0])
    print("first training label: ", y_train[0])
    # Encode labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    print("label mapping", dict(zip(label_encoder.classes_, range(len(label_encoder.classes_)))))
    # Create a Bunch object to store the preprocessed data
    data = Bunch(
        data=X_train,
        target=y_train_encoded,
        target_names=list(label_encoder.classes_),
        images=X_test,
        labels=y_test_encoded
    )

    return data

data_path = "DataCSV.xlsx" 
preprocessed_data = preprocess_data(data_path)
print(preprocessed_data.data[0][0])

plt.imshow(preprocessed_data.data[0][0], cmap='gray')
plt.show()