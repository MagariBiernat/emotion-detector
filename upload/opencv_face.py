from django.conf import settings
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
import os

from .models import EmotionModel

model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

pathToModel = os.path.join(
                settings.BASE_DIR,
                "opencv_haarcascade_data/model.h5",
            )

pathToHaarscascade = os.path.join(
                settings.BASE_DIR,
                "opencv_haarcascade_data/haarcascade_frontalface_default.xml",
            )

def opencv_face(path):
    model.load_weights(pathToModel)
    img = cv2.imread(path)

    # dictionary which assigns each label an emotion (alphabetical order)
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

    if type(img) is np.ndarray:
        print(img.shape)

        face_cascade = cv2.CascadeClassifier( pathToHaarscascade )

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y : y + h, x : x + w]
            roi_color = img[y : y + h, x : x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))
            # AddEmotionToDB(emotion_dict[maxindex])
            cv2.putText(img, emotion_dict[maxindex], (x+50, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            

        cv2.imwrite(path, img)

    else:
        print("something error")
        print(path)


def AddEmotionToDB(EmotionFromPicture):
    Emotion = EmotionModel(emotion=EmotionFromPicture)
    Emotion.save()