import tensorflow as tf
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def train(normalized_train_data : tuple,
           normalized_test_data : tuple,
                   dense_layers = 2,
              neurons_per_layer = [128, 128],
           activation_per_layer = [tf.nn.relu, tf.nn.relu],
                      optimizer = 'adam',
                         epochs = 5,
              output_activation = tf.nn.softmax):

    # check all of the list parameter len
    if dense_layers != len(neurons_per_layer) or dense_layers != len(activation_per_layer):
        print("length of neurons_per_layer/activation_per_layer is not uniform")
        raise IndexError

    # initialize the model
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())

    # add dense layers
    for i in range(dense_layers):
        model.add(tf.keras.layers.Dense(neurons_per_layer[i], activation=activation_per_layer[i]))

    # add output layer
    model.add(tf.keras.layers.Dense(10, activation=output_activation))  

    model.compile(optimizer='adam',  
                loss='sparse_categorical_crossentropy',  
                metrics=['acc'])  

    model.fit(normalized_train_data[0], normalized_train_data[1], epochs=epochs)  

    val_loss, val_acc = model.evaluate(normalized_test_data[0], normalized_test_data[1])

    return (val_loss, val_acc)