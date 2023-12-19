import os
import input_data
import tensorflow.compat.v1 as tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.disable_v2_behavior()
from sklearn.model_selection import train_test_split
import numpy as np
import random
import matplotlib.pyplot as plt
from time import time

# 将 numpy 数组中的图片和标签顺序打乱
def shuffer_images_and_labels(images, labels):
    shuffle_indices = np.random.permutation(np.arange(len(images)))
    shuffled_images = images[shuffle_indices]
    shuffled_labels = labels[shuffle_indices]
    return shuffled_images, shuffled_labels

# 将label从长度10的one hot向量转换为0~9的数字
# 例：get_label(total_labels[0]) 获取到total_labels中第一个标签对应的数字
def get_label(label):
    return np.argmax(label)

def show_image(image):
    tmp = np.zeros((28,28))
    for i in range(len(image)):
        tmp[i//28][i%28] = image[i]
    plt.imshow(tmp)
    plt.show()

# images：训练集的feature部分
# labels：训练集的label部分
# batch_size： 每次训练的batch大小
# epoch_num： 训练的epochs数
# shuffle： 是否打乱数据
# 使用示例：
#   for (batchImages, batchLabels) in batch_iter(images_train, labels_train, batch_size, epoch_num, shuffle=True):
#       sess.run(feed_dict={inputLayer: batchImages, outputLabel: batchLabels})
def batch_iter(images,labels, batch_size, epoch_num, shuffle=True):
    
    data_size = len(images)
    
    num_batches_per_epoch = int(data_size / batch_size)  # 样本数/batch块大小,多出来的“尾数”，不要了
    
    for epoch in range(epoch_num):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            
            shuffled_data_feature = images[shuffle_indices]
            shuffled_data_label   = labels[shuffle_indices]
        else:
            shuffled_data_feature = images
            shuffled_data_label = labels

        for batch_num in range(num_batches_per_epoch):   # batch_num取值0到num_batches_per_epoch-1
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)

            yield (shuffled_data_feature[start_index:end_index] , shuffled_data_label[start_index:end_index])

#FCN全卷积神经网络
def fcn_layer(inputs,  # input data
              input_dim,  # Input numbers of Neurons
              output_dim,  # Output numbers of Neurons
              activation=None):  # activation function
    # Random numbers that generate data that is more than twice the standard deviation will be replaced here
    W = tf.Variable(tf.truncated_normal([input_dim, output_dim], stddev=0.1))
    b = tf.Variable(tf.zeros([output_dim]))  # init as 0

    XWb = tf.matmul(inputs, W) + b

    if activation is None:
        outputs = XWb
    else:
        outputs = activation(XWb)

    return outputs

# 构建和训练模型
def train_and_test(images_train, labels_train, images_test, labels_test,
                   images_validation, labels_validation):
    # Input layers (28*28*1)
    x = tf.placeholder(tf.float32, [None, 784], name="X")
    # 0-9 => 10 numbers
    y = tf.placeholder(tf.float32, [None, 10], name="Y")
    # 2 Hidden layers
    h1 = fcn_layer(inputs=x,
                   input_dim=784,
                   output_dim=256,
                   activation=tf.nn.relu)
    h2 = fcn_layer(inputs=h1,
                   input_dim=256,
                   output_dim=64,
                   activation=tf.nn.relu)
    # Output layers
    forward = fcn_layer(inputs=h2,
                        input_dim=64,
                        output_dim=10,
                        activation=None)
    pred = tf.nn.softmax(forward)

    loss_function = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=forward, labels=y))

    train_epochs = 20  # Train times
    batch_size = 100  # single batch train size
    learning_rate = 0.001  # learning rate

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss_function)

    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    startTime = time()

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    for epoch in range(train_epochs):
        for xs, ys in batch_iter(images_train, labels_train, batch_size, 1, shuffle=True):
            sess.run(optimizer, feed_dict={x: xs, y: ys})

        loss, acc = sess.run([loss_function, accuracy],
                             feed_dict={x: images_test, y: labels_test})

        print(f"[+] {'%02d' % (epoch + 1)}th train:\tloss:", "{:.9f}".format(loss), "\taccuracy:", "{:.4f}".format(acc))

    duration = time() - startTime
    print("[+] Train finished successfully. It takes:", "{:.2f}s".format(duration))

    accu_test = sess.run(accuracy, feed_dict={x: images_test, y: labels_test})
    accu_validation = sess.run(accuracy, feed_dict={x: images_validation, y: labels_validation})
    return accu_test,accu_validation

# 划分数据集并调用train_and_test测试和验证
def hold_out(images, labels, train_percentage):
    X_train, X_test, Y_train, Y_test = train_test_split(images,
                                                        labels,
                                                        test_size=1 - train_percentage,
                                                        random_state=1,
                                                        stratify=labels)
    return X_train, Y_train, X_test, Y_test

def cross_validation(images, labels, k, vali_images, vali_labels):
    total_images = [[] for _ in range(10)]
    total_labels = [[] for _ in range(10)]

    for i in range(len(images)):
        index = get_label(labels[i])
        total_images[index].append(images[i])
        total_labels[index].append(labels[i])

    k_total_images = []
    k_total_labels = []  # 大小为k
    for i in range(10):
        for j in range(k):
            k_total_images.append(total_images[i][int(j * len(total_images[i]) / k):int((j + 1) * len(total_images[i]) / k)])  # 长度为k*10，里面的列表长度为len(total_images[i])/k
            k_total_labels.append(total_labels[i][int(j * len(total_images[i]) / k):int((j + 1) * len(total_images[i]) / k)])

    tmp_accu_test = 0
    tmp_accu_vali = 0
    for idex in range(k):
        X_test_images = k_total_images[idex]  # 大小为1
        Y_test_labels = k_total_labels[idex]
        X_train_images = k_total_images  # 大小为k-1
        Y_train_labels = k_total_labels
        del X_train_images[idex]  # 大小为k-1 del删除变量
        del Y_train_labels[idex]

        f_X_train_images = []
        f_Y_train_labels = []

        for i in range(len(X_train_images)):
            for j in range(len(X_train_images[i])):
                f_X_train_images.append(X_train_images[i][j])
                f_Y_train_labels.append(Y_train_labels[i][j])

        f_X_train_images, f_Y_train_labels = np.array(f_X_train_images), np.array(f_Y_train_labels)
        X_test_images, Y_test_labels = np.array(X_test_images), np.array(Y_test_labels)

        print("[-] k = {}，当前第{}组为测试集".format(k, idex+1))
        accu_test,accu_validation = train_and_test(f_X_train_images, f_Y_train_labels, X_test_images, Y_test_labels,
                                               vali_images, vali_labels)
        print("[*] Temp accuracy of test :", accu_test)
        print("[*] Temp accuracy of validation :", accu_validation)
        tmp_accu_test += accu_test
        tmp_accu_vali += accu_validation

    print("[*] Average accuracy of test :", tmp_accu_test / k)
    print("[*] Average accuracy of validation :", tmp_accu_vali / k)
       
def main():
    # 读取数据集
    mnist = input_data.read_data_sets('./mnist_dataset', one_hot=True)
    # 训练集
    total_images = mnist.train.images
    total_labels = mnist.train.labels
    total_images, total_labels = shuffer_images_and_labels(total_images, total_labels)
    # 验证集
    validation_images = mnist.validation.images
    validation_labels = mnist.validation.labels
    validation_images, validation_labels = shuffer_images_and_labels(validation_images, validation_labels)

    # print(total_images.shape, total_labels.shape)
    # print(validation_images.shape, validation_labels.shape)
    # print(get_label(total_labels[0]))
    # show_image(total_images[0])

    # 简单划分前50000个为训练集，后5000个为测试集，对其进行训练，并使用验证集评估模型
    print("简单划分前50000个为训练集，后5000个为测试集，对其进行训练，并使用验证集评估模型")
    origin_images_train = total_images[:50000]
    origin_labels_train = total_labels[:50000]
    origin_images_test = total_images[50000:]
    origin_labels_test = total_labels[50000:]
    accu_test,accu_validation = train_and_test(origin_images_train, origin_labels_train, origin_images_test, origin_labels_test,
                                               validation_images, validation_labels)
    print("[*] Accuracy of test :", accu_test)
    print("[*] Accuracy of validation :", accu_validation)


    # 使用分层采样的留出法训练、测试模型，并使用验证集评估模型
    # h = 0.8
    # print("使用分层采样的留出法训练、测试模型，并使用验证集评估模型")
    # print("划分比例为 {}%".format(h * 100))
    # origin_images_train, origin_labels_train, origin_images_test, origin_labels_test = hold_out(total_images, total_labels, h)
    # accu_test,accu_validation = train_and_test(origin_images_train, origin_labels_train, origin_images_test, origin_labels_test,
    #                                            validation_images, validation_labels)
    # print("[*] Accuracy of test :", accu_test)
    # print("[*] Accuracy of validation :", accu_validation)

    # 使用分层采样的k折交叉验证法训练、测试模型，并使用验证集评估模型
    # k = 20
    # print("使用分层采样的k折交叉验证法训练、测试模型，并使用验证集评估模型")
    # print("k =", k)
    # cross_validation(total_images, total_labels, k, validation_images, validation_labels)

if __name__ == '__main__':
    main()