import cv2
from keras.models import load_model
import numpy as np
np.set_printoptions(suppress=True)
# Loading the cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the Keras model
model = load_model("keras_model.h5", compile=False)


def get_className(classNo):
    if classNo == 0:
        return "No face"
    elif classNo == 1:
        return "Yash"

# Initialize video capture
cap = cv2.VideoCapture(1)
cv2.namedWindow("Webcam Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Webcam Image", 800, 600)
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    success, imgOriginal = cap.read()

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2GRAY)

    # Perform face detection
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(500, 500))

    for x, y, w, h in faces:
        # Crop the detected face
        crop_img = imgOriginal[y:y+h, x:x+w]
        image = cv2.resize(crop_img, (224, 224), interpolation=cv2.INTER_AREA)

        # Make the image a numpy array and reshape it to the model's input shape.
        image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

        # Normalize the image array
        image = (image / 127.5) - 1

        # Make predictions using the loaded model
        prediction = model.predict(image)
        classIndex = np.argmax(prediction)
        probabilityValue = np.amax(prediction)

        # Draw bounding box on the original image
        cv2.rectangle(imgOriginal, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Display class name and confidence score only when class is "Yash" and 100% confident
        if classIndex == 1 and prediction[0][1]==1 and prediction[0][0] ==0:
            class_name = get_className(classIndex)
            cv2.putText(imgOriginal, class_name, (x, y-10), font, 1.5, (0, 255, 0), 4, cv2.LINE_AA)
        else:
            class_name = get_className(classIndex)
            cv2.putText(imgOriginal, class_name, (x, y - 10), font, 1.5, (0, 255, 0), 4, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow("Result", imgOriginal)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
