import tensorflow as tf
import os
import logging
from tensorflow.python.keras.backend import flatten
from src.utils.all_utils import get_timestamp

def get_VGG_16_model(input_shape, model_path):
    model = tf.keras.applications.vgg16.VGG16(
        input_shape=input_shape,
        weights="imagenet",
        include_top=False # this is made false to not include fully connected layer ANN at end of VGG16 model
    )

    model.save(model_path)
    logging.info(f"VGG16 base model saved at: {model_path}")
    return model

def prepare_model(model,CLASSES,freeze_all,freeze_till,learning_rate):
    if freeze_all:
        for layers in model.layers:
            layers.trainable=False
    if (freeze_till is not None) and (freeze_till>0):
        for layers in model.layers[:freeze_till]:
            layers.trainable=False
    
    # Add our fully connected layers
    flatten_in = tf.keras.layers.Flatten()(model.output)
    prediction = tf.keras.layers.Dense(
        units=CLASSES,
        activation = "softmax"
        )(flatten_in)
    
    full_model = tf.keras.models.Model(
        inputs = model.input,
        outputs = prediction
    )

    full_model.compile(
        optimizer = tf.keras.optimizers.SGD(learning_rate = learning_rate),
        loss = tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"]
        )
    logging.info("custom model is compiled and ready to be trained")

    return full_model
