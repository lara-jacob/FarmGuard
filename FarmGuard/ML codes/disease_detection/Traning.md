# importing necessaary libraries
```python
import tensorflow as tf
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns
```
# loading training set
```python
training_set = tf.keras.utils.image_dataset_from_directory(
    'train',
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(128, 128),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False
)
```
# loading validation set
```python
validation_set = tf.keras.utils.image_dataset_from_directory(
    'valid',
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(128, 128),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False
)
```
# loading the model
```python
cnn = tf.keras.models.Sequential()
```
# training the model
```python
cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,padding='same',activation='relu',input_shape=[128,128,3]))
cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))
```
```python
cnn.add(tf.keras.layers.Conv2D(filters=64,kernel_size=3,padding='same',activation='relu'))
cnn.add(tf.keras.layers.Conv2D(filters=64,kernel_size=3,activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))
```
```python
cnn.add(tf.keras.layers.Conv2D(filters=128,kernel_size=3,padding='same',activation='relu'))
cnn.add(tf.keras.layers.Conv2D(filters=128,kernel_size=3,activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))
```
```python
cnn.add(tf.keras.layers.Conv2D(filters=256,kernel_size=3,padding='same',activation='relu'))
cnn.add(tf.keras.layers.Conv2D(filters=256,kernel_size=3,activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))
```
```python
cnn.add(tf.keras.layers.Conv2D(filters=512,kernel_size=3,padding='same',activation='relu'))
cnn.add(tf.keras.layers.Conv2D(filters=512,kernel_size=3,activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))
```
```python
cnn.add(tf.keras.layers.Dropout(0.25))
```
```python
cnn.add(tf.keras.layers.Flatten())
```
```python
cnn.add(tf.keras.layers.Dense(units=1500,activation='relu'))
```
```python
cnn.add(tf.keras.layers.Dropout(0.4)) #To avoid overfitting
```
```python
cnn.add(tf.keras.layers.Dense(units=38,activation='softmax'))
```
```python
cnn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy', metrics=['accuracy'])
```
```python
cnn.summary()
```
```python
training_history = cnn.fit(x=training_set,validation_data=validation_set,epochs=10)
```
# evaluvating training and validation accuracy
```python
train_loss, train_acc = cnn.evaluate(training_set)
print('Training accuracy:', train_acc)
```
```python
val_loss, val_acc = cnn.evaluate(validation_set)
print('Validation accuracy:', val_acc)
```
```python
cnn.save('trained_plant_disease_model.keras')
```

```python
training_history.history #Return Dictionary of history
```
```python
import json
with open('training_hist.json','w') as f:
  json.dump(training_history.history,f)
```
```python
print(training_history.history.keys())
```
```python
epochs = [i for i in range(1,11)]
plt.plot(epochs,training_history.history['accuracy'],color='red',label='Training Accuracy')
plt.plot(epochs,training_history.history['val_accuracy'],color='blue',label='Validation Accuracy')
plt.xlabel('No. of Epochs')
plt.title('Visualization of Accuracy Result')
plt.legend()
plt.show()
```
```python
class_name = validation_set.class_names
```
```python
y_pred = cnn.predict(test_set)
predicted_categories = tf.argmax(y_pred, axis=1)
```
```python
true_categories = tf.concat([y for x, y in test_set], axis=0)
Y_true = tf.argmax(true_categories, axis=1)
```
```python
from sklearn.metrics import confusion_matrix,classification_report
cm = confusion_matrix(Y_true,predicted_categories)
```
```python
print(classification_report(Y_true,predicted_categories,target_names=class_name))
```
```python
plt.figure(figsize=(40, 40))
sns.heatmap(cm,annot=True,annot_kws={"size": 10})

plt.xlabel('Predicted Class',fontsize = 20)
plt.ylabel('Actual Class',fontsize = 20)
plt.title('pest prediction Confusion Matrix',fontsize = 25)
plt.show()
```
# EDA 
```python
class_names = training_set.class_names
num_classes = len(class_names)

print(f"Number of classes: {num_classes}")
print(f"Class names: {class_names}")

num_images = sum(1 for _ in training_set.unbatch())
print(f"Number of images: {num_images}")
```
```python
import numpy as np
import matplotlib.pyplot as plt

class_counts = {class_name: 0 for class_name in class_names}

for _, labels in training_set.unbatch():
    class_index = np.argmax(labels)
    class_name = class_names[class_index]
    class_counts[class_name] += 1

plt.figure(figsize=(10, 5))
plt.bar(class_counts.keys(), class_counts.values())
plt.title("Class Distribution")
plt.xlabel("Class")
plt.ylabel("Number of Images")
plt.xticks(rotation=90)
plt.show()
```
```python
plt.figure(figsize=(10, 10))
for images, labels in training_set.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[np.argmax(labels[i])])
        plt.axis("off")
plt.show()
```
```python
image_shapes = [image.shape for image, _ in training_set.unbatch()]
image_shapes = np.array(image_shapes)
unique_shapes, shape_counts = np.unique(image_shapes, axis=0, return_counts=True)

print("Unique image shapes and their counts:")
for shape, count in zip(unique_shapes, shape_counts):
    print(f"{shape}: {count}")

aspect_ratios = image_shapes[:, 1] / image_shapes[:, 0]
plt.figure(figsize=(10, 5))
plt.hist(aspect_ratios, bins=20, color='blue', edgecolor='black')
plt.title("Aspect Ratio Distribution")
plt.xlabel("Aspect Ratio (Width / Height)")
plt.ylabel("Frequency")
plt.show()
```
```python
def plot_color_distribution(images, title):
    color_channels = ('Red', 'Green', 'Blue')
    colors = np.stack([images[:, :, :, i] for i in range(3)], axis=-1).mean(axis=(0, 1, 2))
    
    plt.figure(figsize=(10, 5))
    plt.bar(color_channels, colors, color=['r', 'g', 'b'])
    plt.title(title)
    plt.ylabel("Mean Pixel Value")
    plt.show()

for images, _ in training_set.take(1):
    images = images.numpy().astype("float32") / 255.0
    plot_color_distribution(images, "Color Distribution in Sample Images")
```










