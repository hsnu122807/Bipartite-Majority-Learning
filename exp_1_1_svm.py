# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import time
import os


dir_path = os.path.dirname(os.path.realpath(__file__))
new_path = r"{0}\exp_1_1".format(dir_path)
if not os.path.exists(new_path):
    os.makedirs(new_path)
new_path = r"{0}\svm".format(new_path)
if not os.path.exists(new_path):
    os.makedirs(new_path)

np.random.seed(1)
tf.set_random_seed(1)
for owl in range(1, 10):
    with tf.Graph().as_default():
        sess = tf.Session()
        owl_rule = str(owl)
        new_path = r"{0}\exp_1_1".format(dir_path)
        if not os.path.exists(new_path):
            os.makedirs(new_path)
        new_path = r"{0}\svm".format(new_path)
        if not os.path.exists(new_path):
            os.makedirs(new_path)
        new_path = r"{0}\rule_{1}".format(new_path, owl_rule)
        if not os.path.exists(new_path):
            os.makedirs(new_path)
        # 從事先抽好的train中拿資料
        mal_training_data = np.loadtxt(r"exp_1_1\data\rule_{0}\mal_training_data.txt".format(owl_rule), dtype=float, delimiter=" ")
        benign_training_data = np.loadtxt(r"exp_1_1\data\rule_{0}\benign_training_data.txt".format(owl_rule), dtype=float, delimiter=" ")
        for i in range(mal_training_data.shape[0]):
            mal_y = np.array([-1], dtype=float)
            benign_y = np.array([1], dtype=float)
            mal_row = np.concatenate([mal_y, mal_training_data[i]]).reshape(1, -1)
            benign_row = np.concatenate([benign_y, benign_training_data[i]]).reshape(1, -1)
            if i == 0:
                training_data = mal_row
            else:
                training_data = np.concatenate([training_data, mal_row], axis=0)
            training_data = np.concatenate([training_data, benign_row], axis=0)
        x_training_data = training_data[:, 1:]
        y_training_data = training_data[:, 0].reshape((-1, 1))

        x_training_data_mal_part = x_training_data[np.where(y_training_data == -1)[0]]
        x_training_data_benign_part = x_training_data[np.where(y_training_data == 1)[0]]
        y_training_data_mal_part = y_training_data[np.where(y_training_data == -1)[0]]
        y_training_data_benign_part = y_training_data[np.where(y_training_data == 1)[0]]
        # print(x_training_data_mal_part.shape)
        # print(y_training_data_mal_part.shape)

        mal_testing_data = np.loadtxt(r"exp_1_1\data\rule_{0}\mal_testing_data.txt".format(owl_rule), dtype=float, delimiter=" ")
        benign_testing_data = np.loadtxt(r"exp_1_1\data\rule_{0}\benign_testing_data.txt".format(owl_rule), dtype=float, delimiter=" ")
        mal_y = np.array([-1], dtype=float)
        benign_y = np.array([1], dtype=float)

        x_testing_data_mal_part = mal_testing_data
        x_testing_data_benign_part = benign_testing_data
        y_testing_data_mal_part = np.tile(mal_y, x_testing_data_mal_part.shape[0]).reshape(-1, 1)
        y_testing_data_benign_part = np.tile(benign_y, x_testing_data_benign_part.shape[0]).reshape(-1, 1)
        # print(x_testing_data_mal_part.shape)
        # print(x_testing_data_benign_part.shape)

        # parameters
        every_stage_max_thinking_times = 10000
        m = x_training_data.shape[1]
        data_size = training_data.shape[0]
        outlier_rate = 0.05
        learning_rate = 0.01

        #批训练中批的大小
        # batch_size = x_training_data.shape[0]
        x_data = tf.placeholder(dtype=tf.float32)
        y_target = tf.placeholder(dtype=tf.float32)
        W = tf.Variable(tf.random_normal(shape=[52,1]))
        b = tf.Variable(tf.random_normal(shape=[1,1]))
        #定义损失函数
        model_output=tf.matmul(x_data,W)+b
        l2_norm = tf.reduce_sum(tf.square(W))
        #软正则化参数
        alpha = tf.constant([0.1])
        #定义损失函数
        classification_term = tf.reduce_mean(tf.maximum(0.,1.-model_output*y_target))
        loss = classification_term+alpha*l2_norm
        # 定義排序準則
        order_term = -model_output*y_target
        #输出
        prediction = tf.sign(model_output)
        correct_count = tf.reduce_sum(tf.cast(tf.equal(prediction, y_target),tf.int32))
        accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, y_target),tf.float32))
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
        # weight init
        sess.run(tf.global_variables_initializer())

        # create file to save training process
        training_process_log = open(new_path + r"\_two_class_training_process.txt", 'w')

        # counter
        bp_times_count = 0

        start_time = time.time()
        for stage in range(m+2, int(data_size * (1 - outlier_rate) + 1)):
            print('-----stage: ' + str(stage) + '-----')
            training_process_log.writelines('-----stage: ' + str(stage) + '-----' + "\n")

            if stage == (m+2):
                current_stage_training_x = x_training_data[:m+2]
                current_stage_training_y = y_training_data[:m+2]
                # print(current_stage_training_x.shape)
                # print(current_stage_training_y.shape)
            else:  # 用order term sorting
                order_term_of_all_data = sess.run(order_term, feed_dict={x_data: x_training_data, y_target: y_training_data}).reshape(-1, 1)

                concat_x_and_y = np.concatenate((x_training_data, y_training_data), axis=1)
                concat_order_and_x_y = np.concatenate((order_term_of_all_data, concat_x_and_y), axis=1)
                # print(concat_entropy_and_x_y.shape)
                sort_result = concat_order_and_x_y[np.argsort(concat_order_and_x_y[:, 0])]
                x_training_data_sort_by_entropy = np.delete(sort_result, (0, m + 1), axis=1)  # 去除0和m+1欄
                y_training_data_sort_by_entropy = np.delete(sort_result, slice(0, m + 1), axis=1)  # 去除從0到m欄
                current_stage_training_x = x_training_data_sort_by_entropy[:stage]
                current_stage_training_y = y_training_data_sort_by_entropy[:stage]

                last_loss = 1e6
                for i in range(every_stage_max_thinking_times):
                    # print(current_stage_training_x.shape)
                    # print(current_stage_training_y.shape)
                    correct_rate = sess.run(accuracy,
                                            feed_dict={x_data: current_stage_training_x, y_target: current_stage_training_y})
                    if correct_rate == 1:
                        if i == 0:
                            print('all data in this stage correctly classified.')
                            training_process_log.writelines('all data in this stage correctly classified.' + "\n")
                        else:
                            print('all data in this stage correctly classified after {0} times bp.'.format(i))
                            training_process_log.writelines(
                                'all data in this stage correctly classified after {0} times bp.'.format(i) + "\n")
                        break

                    sess.run(train_step, feed_dict={x_data: current_stage_training_x, y_target: current_stage_training_y})
                    bp_times_count += 1

                    if i % 1000 == 0:
                        curr_loss = sess.run(loss,
                                        feed_dict={x_data: current_stage_training_x, y_target: current_stage_training_y})
                        print(curr_loss)
                        if (last_loss - curr_loss) < 0.001:
                            print('learning too slow, break.')
                            training_process_log.writelines(
                                'BP failed: after {0} times training, learning too slow.'.format((stage + 1)) + "\n")
                            break
                        else:
                            last_loss = curr_loss
        training_process_log.close()
        end_time = time.time()
        print('train end, save networks')
        current_W, current_b = sess.run([W, b])
        np.savetxt(new_path + r"\two_class_weight.txt", current_W)
        np.savetxt(new_path + r"\two_class_bias.txt", current_b)

        order_term_of_all_data = sess.run(order_term, feed_dict={x_data: x_training_data, y_target: y_training_data}).reshape(-1, 1)
        concat_x_and_y = np.concatenate((x_training_data, y_training_data), axis=1)
        concat_order_and_x_y = np.concatenate((order_term_of_all_data, concat_x_and_y), axis=1)
        sort_result = concat_order_and_x_y[np.argsort(concat_order_and_x_y[:, 0])]
        np.savetxt(new_path + r"\training_data_order_x_y.txt", sort_result)

        file = open(new_path + r"\_two_class_training_detail.txt", 'w')
        file.writelines('learning rate: {0}\n'.format(learning_rate))
        file.writelines('input node amount: {0}\n'.format(m))
        file.writelines('training data amount: {0}\n'.format(data_size))
        file.writelines('outlier rate: {0}\n'.format(outlier_rate))
        file.writelines('thinking times count: {0}\n'.format(bp_times_count))
        file.writelines('execute time: {0} sec\n'.format(end_time - start_time))
        file.close()

        file = open(new_path + r"\_training_analyze.txt", 'w')
        benign_train_correct_count = sess.run(correct_count, {x_data: x_training_data_benign_part, y_target: y_training_data_benign_part})
        benign_train_accuracy = sess.run(accuracy, {x_data: x_training_data_benign_part, y_target: y_training_data_benign_part})
        file.writelines('benign accuracy: {0}/{1} , {2}\n'.format(benign_train_correct_count, x_training_data_benign_part.shape[0], benign_train_accuracy))
        mal_train_correct_count = sess.run(correct_count, {x_data: x_training_data_mal_part, y_target: y_training_data_mal_part})
        mal_train_accuracy = sess.run(accuracy, {x_data: x_training_data_mal_part, y_target: y_training_data_mal_part})
        file.writelines('mal accuracy: {0}/{1} , {2}\n'.format(mal_train_correct_count, x_training_data_mal_part.shape[0], mal_train_accuracy))
        file.close()

        file = open(new_path + r"\_testing_analyze.txt", 'w')
        benign_test_correct_count = sess.run(correct_count, {x_data: x_testing_data_benign_part,
                                                              y_target: y_testing_data_benign_part})
        benign_test_accuracy = sess.run(accuracy, {x_data: x_testing_data_benign_part,
                                                    y_target: y_testing_data_benign_part})
        file.writelines('benign accuracy: {0}/{1} , {2}\n'.format(benign_test_correct_count,
                                                                  x_testing_data_benign_part.shape[0],
                                                                  benign_test_accuracy))
        mal_test_correct_count = sess.run(correct_count,
                                           {x_data: x_testing_data_mal_part, y_target: y_testing_data_mal_part})
        mal_test_accuracy = sess.run(accuracy,
                                      {x_data: x_testing_data_mal_part, y_target: y_testing_data_mal_part})
        file.writelines(
            'mal accuracy: {0}/{1} , {2}\n'.format(mal_test_correct_count, x_testing_data_mal_part.shape[0],
                                                   mal_test_accuracy))
        file.close()
