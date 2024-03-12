from __future__ import absolute_import, division, print_function

def mlp_network(combination, learning_rate, epochs, batches,seeds):





    from numpy.random import seed
    seed(seeds)
    from tensorflow import set_random_seed
    set_random_seed(seeds)


    import tensorflow as tf
    from tensorflow import keras
    from tensorflow import  layers
    from tensorflow.layers import Dense, Conv2D, Flatten
    import tensorflow as tf
    from tensorflow import keras
    from keras.callbacks import TensorBoard

    imdb = keras.datasets.imdb
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

    import numpy as np


    word_index = imdb.get_word_index()


    word_index = {k:(v+3) for k,v in word_index.items()}
    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2
    word_index["<UNUSED>"] = 3

    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

    def decode_review(text):
        return ' '.join([reverse_word_index.get(i, '?') for i in text])



    train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                            value=word_index["<PAD>"],
                                                            padding='post',
                                                            maxlen=256)

    test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                           value=word_index["<PAD>"],
                                                           padding='post',
                                                           maxlen=256)

    vocab_size = 10000

    ######################################Combination 1#################################################


    model1 = keras.Sequential()
    model1.add(keras.layers.Embedding(vocab_size, 16))
    model1.add(keras.layers.Dropout(0.4))
    model1.add(keras.layers.GlobalAveragePooling1D())
    model1.add(keras.layers.Dense(16, activation=tf.nn.relu))
    model1.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))







    ######################################combination2#################################################

    model2 = keras.Sequential()
    model2.add(keras.layers.Embedding(vocab_size, 16))
    model2.add(keras.layers.Dropout(0.2))
    model2.add(keras.layers.LSTM(1, activation=tf.nn.tanh))
    model2.add(keras.layers.Dense(32, activation=tf.nn.sigmoid))
    model2.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))


    ###########################################Combination3 ################################################


    model3 = keras.Sequential()
    model3.add(keras.layers.Embedding(vocab_size, 6))
    model3.add(keras.layers.Dropout(0.4))

    model3.add(keras.layers.LSTM(1,input_shape=(vocab_size, 1),return_sequences=True,activation=tf.nn.sigmoid))
    model3.add(keras.layers.TimeDistributed(Dense(16,activation=tf.nn.softmax)))
    model3.add(keras.layers.GlobalAveragePooling1D())
    model3.add(keras.layers.Dense(32, activation=tf.nn.relu))
    model3.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))





    ###############################Combinataion 4#####################################################



    model4 = keras.Sequential()
    model4.add(keras.layers.Embedding(vocab_size, 6))
    model4.add(keras.layers.Dropout(0.2))

    model4.add(keras.layers.Bidirectional(tf.keras.layers.LSTM(16,input_shape=(vocab_size, 1),return_sequences=True,activation='relu')))
    model4.add(keras.layers.TimeDistributed(Dense(16, activation='softmax')))
    model4.add(keras.layers.GlobalMaxPooling1D())
    model4.add(keras.layers.Dense(32, activation=tf.nn.relu))
    model4.add(keras.layers.Dense(16, activation=tf.nn.relu))
    model4.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

    ################################################################################################





    if combination==1:
        model=model1
        optim = tf.keras.optimizers.Adam(learning_rate)
        model.compile(optimizer=optim,
                      loss='binary_crossentropy',
                      metrics=['acc'],
                      )

    if combination==2:
        model = model2
        optim = tf.keras.optimizers.Adamax(learning_rate)
        model.compile(optimizer=optim,
                      loss='binary_crossentropy',
                      metrics=['acc'])

    if combination == 3:
        model = model3
        optim = tf.keras.optimizers.Nadam(learning_rate)
        model.compile(optimizer=optim,
                      loss='binary_crossentropy',
                      metrics=['acc'])


    if combination == 4:
        model = model4
        optim = tf.keras.optimizers.SGD(learning_rate)
        model.compile(optimizer=optim,
                      loss='binary_crossentropy',
                      metrics=['acc'])

    x_val = train_data[:10000]
    partial_x_train = train_data[10000:]

    y_val = train_labels[:10000]
    partial_y_train = train_labels[10000:]

    model.save('./imdb-1-0.01-20-100-12345.cpkt')

    tbCallBack = keras.callbacks.TensorBoard(log_dir='logs', histogram_freq=0, write_graph=True, write_images=True)

    model.fit(partial_x_train,
                       partial_y_train,
                       epochs=epochs,#2,
                       batch_size=batches,#20,
                       validation_data=(x_val, y_val),
                      verbose=1,callbacks=[tbCallBack])




    #tf.keras.utils.plot_model(
     #   model,
      #  to_file='imdbmodel.png',
       # show_shapes=True,
        #show_layer_names=True,
        #rankdir='TB'
    #)


    results_test = model.evaluate(test_data, test_labels)
    results_train = model.evaluate(partial_x_train, partial_y_train)

    print("Training Accuracy: %.5f" %results_train[1])

    print("Testing Accuracy: %5f" %results_test[1])




