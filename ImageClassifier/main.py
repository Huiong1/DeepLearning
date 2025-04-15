import tensorflow as tf
import matplotlib.pyplot as plt
from ImageClassifier import imageClassifier_MLP ##일단 주석처리 나중에 클래스 만들고 주석해제

def run_classifier():
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Bag', 'Ankle boot']
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
    
    img_x, img_y = train_images.shape[1], train_images.shape[2]
    num_classes = 10
    model = imageClassifier_MLP(img_x, img_y, num_classes)
    model.build_MLP_model()

    # 라벨을 one-hot 벡터로 변환 (TensorFlow 사용)
    train_labels_onehot = model.to_onehotvec_label(train_labels, num_classes)
    test_labels_onehot = model.to_onehotvec_label(test_labels, num_classes)

    # 모델 학습
    model.fit(train_images, train_labels_onehot, num_epochs=10)

    # 예측
    predictions = model.predict(test_images)

    # 정확도 평가
    predicted_classes = tf.argmax(predictions, axis=1)
    correct = tf.equal(predicted_classes, test_labels)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    print(f"Test accuracy: {accuracy.numpy() * 100:.2f}%")

    # 예측 결과 시각화
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(test_images[i], cmap=plt.cm.binary)
        pred_label = class_names[predicted_classes[i].numpy()]
        true_label = class_names[test_labels[i]]
        color = 'blue' if pred_label == true_label else 'red'
        plt.xlabel(f"{pred_label} ({true_label})", color=color)
    plt.show()

    
if __name__ == "__main__":
    run_classifier()    
