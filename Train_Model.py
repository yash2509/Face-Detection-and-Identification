from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dropout
from keras.optimizers import Adam
from keras.applications.resnet50 import ResNet50
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from keras.layers import Input, Flatten, Dense
from keras.models import Model

IMAGE_SIZE = [218, 178]
train_path = 'Datasets/Train'
valid_path = 'Datasets/Test'

resnet = ResNet50(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)
for layer in resnet.layers[:10]:  # Specify the index range of layers you want to train
    layer.trainable = False
for layer in resnet.layers[10:]:  # Set the remaining layers as frozen
    layer.trainable = False

folders = glob('Datasets/Train')
num_classes =len(folders)+1
print(num_classes)

x = Flatten()(resnet.output)
x = Dropout(0.99)(x)
x = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=resnet.input, outputs=x)
model.compile(optimizer=Adam(learning_rate=0.00095), loss='categorical_crossentropy', metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1. / 255)

training_set = train_datagen.flow_from_directory(
    train_path,
    target_size=IMAGE_SIZE,
    batch_size=10,
    class_mode='categorical'  # Update: Set class_mode to 'sparse'
)

test_set = test_datagen.flow_from_directory(
    valid_path,
    target_size=IMAGE_SIZE,
    batch_size=10,
    class_mode='categorical'  # Update: Set class_mode to 'sparse'
)
history = model.fit(
    training_set,
    validation_data=test_set,
    epochs=70,
    steps_per_epoch=len(training_set),
    validation_steps=len(test_set)
)
model.save('keras_model.h5')
test_predictions = model.predict(test_set)
test_labels = np.argmax(test_predictions, axis=1)

class_labels = list(test_set.class_indices.keys())
classification_report = classification_report(test_set.classes, test_labels, target_names=class_labels)
print("Classification Report:\n", classification_report)

confusion_mtx = confusion_matrix(test_set.classes, test_labels)
plt.figure(figsize=(8, 6))
plt.imshow(confusion_mtx, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(class_labels))
plt.xticks(tick_marks, class_labels, rotation=90)
plt.yticks(tick_marks, class_labels)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(test_set.classes, test_predictions[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(8, 6))
for i in range(num_classes):
    plt.plot(fpr[i], tpr[i], label='Class {} (AUC = {:.2f})'.format(class_labels[i], roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.show()

