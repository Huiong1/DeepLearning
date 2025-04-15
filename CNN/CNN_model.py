import tensorflow as tf

class CNN:
    def __init__(self, num_labels, img_shape_x, img_shape_y):
        self.num_labels = num_labels
        self.img_shape_x = img_shape_x
        self.img_shape_y = img_shape_y
        self.classifier_model = None
    
    def build_CNN_model(self):
        input_layer = tf.keras.Input(shape=[self.img_shape_x, self.img_shape_y,1,])
        
        hidden_layer = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1,1),
                                            padding='valid', activation='relu')(input_layer)
        hidden_layer = tf.keras.layers.MaxPooling2D((2,2))(hidden_layer)
        hidden_layer = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1,1),
                                            padding='valid', activation='relu')(hidden_layer)
        hidden_layer = tf.keras.layers.MaxPooling2D((2,2))(hidden_layer)
        hidden_layer = tf.keras.layers.Flatten()(hidden_layer)
        
        hidden_layer = tf.keras.layers.Dense(units=64, activation='relu')(hidden_layer)
        output = tf.keras.layers.Dense(units = self.num_labels, activation = 'softmax')(hidden_layer)
        
        classifier_model = tf.keras.Model(inputs=input_layer, outputs = output)
        classifier_model.summary()
        
        opt_alg = tf.keras.optimizers.Adam(learning_rate = 0.001)
        loss_cross_e = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        classifier_model.compile(optimizer = opt_alg, loss = loss_cross_e, metrics=['accuracy'])
        self.classifier = classifier_model
    def fit (self, x, y, batch_size, epochs):
        self.classifier_model.fit(x=x, y=y, batch_size = batch_size, epochs=epochs)
        
    def predict (self, x, batch_size):
        prediction = self.classifier_model.predict(x=x, batch_size=batch_size)
        return prediction