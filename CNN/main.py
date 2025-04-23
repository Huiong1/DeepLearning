import tensorflow as tf
import matplotlib.pyplot as plt
from CNN_model import CNN ##일단 주석처리 나중에 클래스 만들고 주석해제

def run_classifier():
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt','Sneaker', 'Bag', 'Ankle boot']
    print("Train data shape")
    print(train_images.shape)
    print("Train data labels")
    print(train_labels)
    print("Test data shape")
    print(test_images.shape)
    print("Test data labels")
    print(test_labels)
    
    plt.figure()
    plt.imshow(train_images[0])
    plt.colorbar()
    plt.grid(False)
    plt.show()
    
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap = plt.cm.binary)
        plt.xlabel(class_names[train_labels[i]])
    plt.show()    
    
    my_classifier = CNN(img_shape_x=28, img_shape_y = 28, num_labels=10)
    my_classifier.build_CNN_model()
    
    train_labels_onehot = my_classifier.to_onehotvec_label(train_labels, 10)
    
    my_classifier.fit(train_images, train_labels_onehot, epochs=10)
    predicted_labels = my_classifier.predict(test_imgs = test_images)
    predicted_labels = tf.math.argmax(input = predicted_labels, axis=1)
    
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(test_images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[predicted_labels[i]])
    plt.show()
    
    
if __name__ == "__main__":
    run_classifier()    
