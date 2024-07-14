# importing necessary libraries
```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
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
class_name = validation_set.class_names
print(class_name)
```
# loading saved model
```python
cnn = tf.keras.models.load_model('trained_plant_disease_model.keras')
```
# testing the model
```python
import cv2
import matplotlib.pyplot as plt

# Example image path
image_path = 'test/PotatoEarlyBlight2.JPG'

# Attempt to read the image
img = cv2.imread(image_path)

# Check if the image was successfully loaded
if img is not None:
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    plt.imshow(img_rgb)
    plt.show()
else:
    print(f"Error: Unable to read image at {image_path}")

```
```python
image = tf.keras.preprocessing.image.load_img(image_path,target_size=(128,128))
input_arr = tf.keras.preprocessing.image.img_to_array(image)
input_arr = np.array([input_arr])  # Convert single image to a batch.
predictions = cnn.predict(input_arr)
```
```python
print(predictions)
```
```python
result_index = np.argmax(predictions) #Return index of max element
print(result_index)
```
```python
model_prediction = class_name[result_index]
plt.imshow(img)
plt.title(f"Disease Name: {model_prediction}")
plt.xticks([])
plt.yticks([])
plt.show()
```
