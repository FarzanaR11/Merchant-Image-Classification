import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load pre-trained model
model = tf.keras.models.load_model("merchant_classifier.h5")

# Predict
def classify_image(image_path):
    img = tf.keras.utils.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img) / 255.0
    img_array = tf.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    return prediction

if __name__ == "__main__":
    prediction = classify_image("shop_image.jpg")
    print("Prediction:", prediction)
