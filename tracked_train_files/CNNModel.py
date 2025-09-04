"""
Author: Jamie McGrath

Class representing a convolutional neural network (CNN) model. 
This class provides methods to create, compile, and train the CNN model 
for image classification tasks.
The class also provides methods to generate a visualization of the model 
architecture, saving the weights of the best performing model during training.
    
Params:
~ 'model' (Sequential):
    CNN model object.
~ 'valid_gen' (ImageDataGenerator):
    data generator for validation data.
~ 'train_data' (ImageDataGenerator):
    data generator for training data.
~ 'input_shape' (tuple):
    input shape of the images in format (height, width, channels).
~ 'num_classes' (int):
    number of classes in the classification task.
"""

from IPython.display import Image
from tensorflow.keras.utils import plot_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, BatchNormalization, Activation, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
import numpy as np


class CNNModel:

    def __init__(self, valid_gen, train_gen):
        """
        Constructor for the CNNModel class.
        
        Args:
            valid_gen (ImageDataGenerator): data generator for validation data.
            train_gen (ImageDataGenerator): data generator for training data.
        """
        self.model = Sequential()
        self.valid_gen = valid_gen
        self.input_shape = (48, 48, 1)
        self.train_data = train_gen
        self.num_classes = len(self.train_data.class_indices)

    def create_model(self):
        """
        Method to create the CNN model architecture.
        
        Returns:
            Image: The visualisation of the model architecture as an Image object.
        """
        
        # Layer 1
        self.model.add(Conv2D(32, (3, 3), padding='same',
                       input_shape=self.input_shape))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.2))

        # Layer 2
        self.model.add(Conv2D(64, (3, 3), padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.2))

        # Layer 3
        self.model.add(Conv2D(128, (3, 3), padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.2))

        # Layer 4
        self.model.add(Conv2D(256, (3, 3), padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.2))

        # Layer 5
        self.model.add(Conv2D(512, (3, 3), padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.2))

        # Layer 6
        self.model.add(Conv2D(1028, (3, 3), padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(1, 1)))
        self.model.add(Dropout(0.2))

        # Flattening layers
        self.model.add(Flatten())

        # Fully connected 1st layer
        self.model.add(Dense(512))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.2))

        # Fully connected 2nd layer
        self.model.add(Dense(256))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.2))

        # Output Layer
        self.model.add(Dense(self.num_classes, activation='softmax'))

        plot_model(
            self.model,
            to_file='model.png',
            show_shapes=True,
            show_layer_names=True)
    
    def calculate_class_weights(self):
        """
        Menghitung class weights berdasarkan distribusi data training
        untuk mengatasi masalah class imbalance
        """
        # Ekstrak label dari generator
        y_train = []
        
        # Reset generator untuk memastikan kita mendapat semua data
        self.train_data.reset()
        
        # Ekstrak label dari filepaths
        for filepath in self.train_data.filepaths:
            for class_name, class_index in self.train_data.class_indices.items():
                if class_name in filepath:
                    y_train.append(class_index)
                    break
        
        y_train = np.array(y_train)
        classes = np.array(list(range(len(self.train_data.class_indices))))
        
        # Hitung class weights menggunakan sklearn
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=classes,
            y=y_train
        )
        
        # Convert ke dictionary
        class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
        
        print("Class weights calculated:")
        for class_name, class_index in self.train_data.class_indices.items():
            print(f"{class_name}: {class_weight_dict[class_index]:.4f}")
        
        return class_weight_dict
        self.model.summary()

        return Image('model.png', width=400, height=200)

    def compile_model(self):
        """
        Method to compile the CNN model with specified optimiser, loss function, 
        and metrics. Menggunakan class weights untuk mengatasi class imbalance.

        Returns:
            History: 
                The training history containing loss and accuracy values for 
                each epoch.
        """
        epochs = 8  # Jumlah epochs yang optimal untuk training
        optimizer = Adam(learning_rate=0.001)  # utilise Adam as the optimiser
        # create a decay for the learning rate to reduce when accuracy does not improve
        decay_rate = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                       patience=6, min_lr=0.00001, mode='auto')
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy'])

        # checkpoint to save best model weights
        checkpoint = ModelCheckpoint(
            "model_weights.h5",
            monitor='val_accuracy',
            verbose=1,
            save_best_only=True,
            mode='max')
        callbacks_list = [checkpoint, decay_rate]
        
        # Hitung class weights untuk mengatasi imbalance
        print("\nCalculating class weights to handle imbalanced dataset...")
        class_weights = self.calculate_class_weights()
        
        print(f"\nStarting training with {epochs} epochs and balanced class weights...")
        print(f"Training samples: {self.train_data.n}")
        print(f"Validation samples: {self.valid_gen.n}")
        print(f"Classes: {list(self.train_data.class_indices.keys())}")

        # create a history for the model using all metrics dengan class weights
        history = self.model.fit(
            x=self.train_data,
            steps_per_epoch=self.train_data.n // self.train_data.batch_size,
            epochs=epochs,
            validation_data=self.valid_gen,
            validation_steps=self.valid_gen.n // self.valid_gen.batch_size,
            callbacks=callbacks_list,
            class_weight=class_weights,  # Tambahkan class weights!
            verbose=1
        )
        
        print("\nTraining completed! Model weights saved to model_weights.h5")
        return history
