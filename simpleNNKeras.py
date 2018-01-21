"""
Adapted from post from Keras GitHub, Kaggle blog.
"""
import keras
from keras.models import Sequential
from keras import regularizers
import keras.layers as ll
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout
from keras.layers.advanced_activations import LeakyReLU

def MNIST_MLP(input_shape = (28, 28), classes = 10):
    """
    A simple template for fully connected MLP. After 12 Epochs with Adam it is about 98% on MNIST dataset, 
    """
    # Define the input as a tensor with shape input_shape
    model = Sequential(name="mlp")
    model.add(ll.InputLayer(input_shape))
    model.add(ll.Flatten())
    model.add(ll.Dense(120))
    model.add(ll.Activation('relu'))
    model.add(ll.Dense(40, kernel_regularizer=regularizers.l2(0.0001)))
    model.add(ll.Activation('relu'))

    # output layer: 10 neurons for each class with softmax
    model.add(ll.Dense(10, kernel_regularizer=regularizers.l2(0.001), activation='softmax'))
    return model    
 
 def MNIST_CNN(input_shape = (28, 28, 1), classes = 10):
    """
    A simple template for CNN. After 12 Epochs with Adam it is about 99.3% on MNIST dataset, 
    """
    model = Sequential(name="cnn")
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(LeakyReLU(0.01))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(classes, activation='softmax'))
    
    return model
    
 def AUGMENTED_MNIST_CNN(X_train, Y_train, X_val, Y_val, classes = 10, batch_size=32):
    """
    A simple template for CNN with augumented data. After 30 Epochs with Adam it is about 99.6% on MNIST dataset, 
    """  
    model = MNIST_CNN(input_shape, classes)
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    datagen.fit(X_train)
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)
    # Fit the model with generator
    model.fit_generator(datagen.flow(X_train,y_train, batch_size),
                              epochs = 30, validation_data = (X_val,y_val),
                              verbose = 2, steps_per_epoch=X_train.shape[0] // batch_size
                              , callbacks=[learning_rate_reduction])
                              
