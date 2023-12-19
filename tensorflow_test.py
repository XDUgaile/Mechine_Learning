import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

#初始化一个Tensorflow的常量：Hello Tensorflow! 字符串，并命名为greeting作为一个计算模块 
greeting = tf.constant('Hello Tensorflow!') 
#启动一个会话 
sess = tf.Session() 
#使用会话执行greeting计算模块 
result = sess.run(greeting) 
print(result)
sess.close()