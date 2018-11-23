import tensorflow as tf
import numpy as np
import time
import os
import math

dir_path = os.path.dirname(os.path.realpath(__file__))
bp_times_count = 10000

# create folder to save training process
new_path = r"{0}\exp_2_2".format(dir_path)
if not os.path.exists(new_path):
    os.makedirs(new_path)
new_path = r"{0}\none".format(new_path)
if not os.path.exists(new_path):
    os.makedirs(new_path)

all_major_samples = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=object)

# 取得所有9 rule挑出的majority
for i in range(1, 10):
        # read all owl data from file
        data_path = r"{0}\exp_2_1\data\rule_{1}\mal_training_data.txt".format(dir_path, i)
        data = np.loadtxt(data_path, dtype=float, delimiter=" ")
        major_data_x_mal_part = data
        data_path = r"{0}\exp_2_1\data\rule_{1}\benign_training_data.txt".format(dir_path, i)
        data = np.loadtxt(data_path, dtype=float, delimiter=" ")
        major_data_x_benign_part = data
        print(major_data_x_mal_part.shape)
        print(major_data_x_benign_part.shape)
        all_major_samples[i] = major_data_x_mal_part
        if i == 1:
            all_major_samples[0] = major_data_x_benign_part
        else:
            all_major_samples[0] = np.append(all_major_samples[0], major_data_x_benign_part, axis=0)

#
for i in range(all_major_samples.shape[0]):
    # np.savetxt('___rule'+str(i), all_major_samples[i])
    y_element = np.zeros(10)
    y_element[i] = 1
    # print(y_element)
    y_element_arr = np.tile(y_element, all_major_samples[i].shape[0]).reshape(-1, 10)
    # print(y_element_arr.shape)
    if i == 0:
        training_x = all_major_samples[i]
        training_y = y_element_arr
    else:
        training_x = np.append(training_x, all_major_samples[i], axis=0)
        training_y = np.append(training_y, y_element_arr, axis=0)
print("all training sample amount:")
print(training_x.shape)

# 三種用相同的初始神經網路調權重
with tf.Graph().as_default():
    with tf.name_scope('placeholder_x'):
        x = tf.placeholder(tf.float64, name='x_placeholder')
    with tf.name_scope('placeholder_y'):
        y_ = tf.placeholder(tf.float64, name='y_placeholder')
    W = tf.Variable(tf.zeros([training_x.shape[1],
                              training_y.shape[1]], dtype=tf.float64), dtype=tf.float64, name='weight')
    b = tf.Variable(tf.zeros([training_y.shape[1]], dtype=tf.float64), dtype=tf.float64, name='bias')
    y = tf.nn.softmax(tf.matmul(x, W) + b, name='predict_y')

    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(0.0001).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    correct_amount = tf.reduce_sum(tf.cast(correct_prediction, tf.int32))

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    last_loss = 10000000.0
    last_times_count = 0
    last_execute_time = 0
    last_correct_rate = 0
    execute_start_time = time.time()

    # writer = tf.summary.FileWriter("C:/logfile", sess.graph)
    # writer.close()

    for i in range(bp_times_count):
        sess.run(train_step, feed_dict={x: training_x, y_: training_y})
        correct_rate = sess.run(accuracy, feed_dict={x: training_x, y_: training_y})
        if i == (bp_times_count-1):
            current_W, current_b = sess.run([W, b], feed_dict={x: training_x, y_: training_y})
            np.savetxt(new_path + r"\weight.txt", current_W)
            np.savetxt(new_path + r"\bias.txt", current_b)

            training_detail = open(new_path + r"\_training_detail.txt", 'w')
            training_detail.writelines('---training data---\n')
            for j in range(all_major_samples.shape[0]):
                if j == 0:
                    training_detail.writelines('Benign data: {0}\n'.format(all_major_samples[j].shape[0]))
                else:
                    training_detail.writelines('Rule {0}: {1}\n'.format(j, all_major_samples[j].shape[0]))
            training_detail.writelines('-------------------\n')
            training_detail.writelines('train times count: ' + str(i+1) + "\n")
            training_detail.writelines("execution time: " + str(time.time() - execute_start_time) + " seconds" + "\n")
            training_detail.close()

            training_accuracy = open(new_path + r"\_training_false_rate.txt", 'w')
            testing_accuracy = open(new_path + r"\_testing_false_rate.txt", 'w')
            for j in range(all_major_samples.shape[0]):
                # training accuracy
                training_xxx = all_major_samples[j]
                y_element = np.zeros(10)
                y_element[j] = 1
                training_yyy = np.tile(y_element, training_xxx.shape[0]).reshape(-1, 10)
                training_correct_amount = sess.run(correct_amount,feed_dict={x: training_xxx, y_: training_yyy})
                if j == 0:
                    training_accuracy.writelines('Benign: {0}/{1}\n'.format(training_xxx.shape[0] - training_correct_amount, training_xxx.shape[0]))
                else:
                    training_accuracy.writelines('Rule {0}: {1}/{2}\n'.format(j, training_xxx.shape[0] - training_correct_amount, training_xxx.shape[0]))
                # testing accuracy
                if j == 0:
                    testing_xxx = np.loadtxt("owl_benign_samples.txt", dtype=float, delimiter=" ")
                else:
                    testing_xxx = np.loadtxt("rule_{0}.txt".format(j), dtype=float, delimiter=" ")
                testing_yyy = np.tile(y_element, testing_xxx.shape[0]).reshape(-1, 10)
                testing_correct_amount = sess.run(correct_amount, feed_dict={x: testing_xxx, y_: testing_yyy})
                testing_correct_amount -= training_correct_amount
                if j == 0:
                    testing_accuracy.writelines(
                        'Benign: {0}/{1}\n'.format(testing_xxx.shape[0]-training_xxx.shape[0] - testing_correct_amount, testing_xxx.shape[0]-training_xxx.shape[0]))
                else:
                    testing_accuracy.writelines(
                        'Rule {0}: {1}/{2}\n'.format(j, testing_xxx.shape[0]-training_xxx.shape[0] - testing_correct_amount, testing_xxx.shape[0]-training_xxx.shape[0]))

            training_accuracy.close()
            testing_accuracy.close()

        if i % 1000 == 0:
            loss = sess.run(cross_entropy, feed_dict={x: training_x, y_: training_y})
            print('training data predict accuracy: '+str(correct_rate * 100)+'%   cross entropy: '+str(loss))
