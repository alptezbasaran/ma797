#import the mnist data set
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#import tensorflow
import tensorflow as tf

#start an interactive session
sess = tf.InteractiveSession()

#placeholders for input images and 1 hot class vector
#inputs are 28x28 images flattened to 1x784
#note that the shape is optional, but will allow us to catch bugs later
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])


#define functions that initialize weights and biases, with slightly positive initial bias to break symmetry
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

#define 2d conv with stride 1 and pad so that the input image size is preserved
# padding = 'SAME' indicates to use enough padding to ensure image size is same
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

#define max pooling over 2 by 2 blocks, non-overlapping stride
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

#define weights and biases for the FIRST LAYER
#which consists of 32 5x5 filters
#note that the 3rd entry is the number of input channels (3 for RGB, 1 for BW)
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

#reshape the input image to a 4d tensor
#the 2nd and 3rd entries are the shape of the input
#the last entry is the number of input channels
x_image = tf.reshape(x, [-1,28,28,1])

#perform convolution followed by relu of the activations
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

#max pool the activations
h_pool1 = max_pool_2x2(h_conv1)

#define the SECOND LAYER and apply it
#its 64 5x5 filters, with 32 input channels because
#we used 32 filters (features) in the last layer
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

#compute the second convolution layer with max pooling
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#define the FC LAYER with 1024 neurons. Recall that after the second max-pooling we have
# 7x7 (with depth 64)
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

#flatten the 7x7x64 input from the previous layer
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

#apply relu activations
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#apply dropout to the FC layer
keep_prob = tf.placeholder(tf.float32) #create a placehold for dropout, allowing you to turn it on during
#training and off during testing
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#make an output layer for 10 classes
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

#form the soft max cross entropy loss
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

#define a training step, which you will run later with the train_step.run command
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

#define a true/false vector to evaluate predictions
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
#recast the true/false vector as a binary vetcor and take the average
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#initialize all variables
sess.run(tf.global_variables_initializer())

#run 5000 iterations of SGD with batch size equal to 50
for i in range(5000):
    batch = mnist.train.next_batch(50)
    if i%100 == 0: #evaluate and print the training accuracy every 100 iterations
        train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
    if i%500 == 0: #evaluate and print the testing accuracy every 500 iterations

        print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))


    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0}) 
    # you can change keep_prob to use dropout. Keep_prob of 1 indicates no dropout, $
    # keep probe of 0.5 indicates 50% dropout





