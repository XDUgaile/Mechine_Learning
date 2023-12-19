from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import models, layers, regularizers   # 模型 层 正则
from tensorflow.keras.optimizers import RMSprop             # 均方根传播
import matplotlib.pyplot as plt

# 导入mnist数据集
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# 数据类型处理
train_images = train_images.reshape(60000, 28*28).astype('float')
train_labels = to_categorical(train_labels)
test_images = test_images.reshape(10000, 28*28).astype('float')
test_labels = to_categorical(test_labels)

# 神经网络模型 序贯模型
network = models.Sequential()
# 隐藏层
network.add(layers.Dense(units=256, activation='relu', input_shape=(28*28, ),
                         kernel_regularizer=regularizers.l1(0.0001)))
network.add(layers.Dropout(0.01))
network.add(layers.Dense(units=64, activation='relu',
                         kernel_regularizer=regularizers.l1(0.0001)))
network.add(layers.Dropout(0.01))
# 输出层
network.add(layers.Dense(units=10, activation='softmax'))

# 编译 确定优化器和损失函数
network.compile(optimizer=RMSprop(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
# 训练
network.fit(train_images, train_labels, epochs=20, batch_size=128, verbose=2)

# 测试
test_loss, test_accuracy = network.evaluate(test_images, test_labels)
print("test_loss:", test_loss, "test_accuracy:", test_accuracy)