import tensorflow as tf
import numpy as np
import time
import random
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
# create folder to save training process
new_path = r"{0}\exp_2_1".format(dir_path)
if not os.path.exists(new_path):
    os.makedirs(new_path)
new_path = r"{0}\analyze".format(new_path)
if not os.path.exists(new_path):
    os.makedirs(new_path)

path = r"{0}\exp_2_1".format(dir_path)

# exp training result
file = open(new_path + r"\training_analyze.txt", 'w')
file.writelines('Rule, Input Amount, Method, #HN, Outliers(#FB/#B,#FM/#M), Execute Time(s), BP Times\n')
for i in range(1,10):
    # softmax
    target_path = r"{0}\softmax\rule_{1}".format(path, i)
    rule = i
    method = 'softmax'
    h_n = 0
    with open(target_path+r'\_two_class_training_detail.txt') as f1:
        while 1:
            line = f1.readline()
            if not line:
                break
            if "training data amount: " in line:
                input_amount = int(line.replace("training data amount: ", ""))
            if "execute time: " in line:
                execute_time = line.replace("execute time: ", "")
                execute_time = execute_time.replace(" sec\n", "")
            if "thinking times count: " in line:
                bp_times = int(line.replace("thinking times count: ", ""))
    training_data_entropy_x_y = np.loadtxt(target_path + r'\training_data_entropy_x_y.txt', dtype=float, delimiter=' ')
    outlier_index = int(training_data_entropy_x_y.shape[0] * 0.95)
    outlier_data = training_data_entropy_x_y[outlier_index:]
    outlier_x = outlier_data[:, 1:53]
    outlier_y = outlier_data[:, 53:]
    with tf.Graph().as_default():
        with tf.name_scope('placeholder_x'):
            x = tf.placeholder(tf.float64, name='x_placeholder')
        with tf.name_scope('placeholder_y'):
            y_ = tf.placeholder(tf.float64, name='y_placeholder')

        weight = np.loadtxt(target_path+r'\two_class_weight.txt', dtype=float)
        bias = np.loadtxt(target_path+r'\two_class_bias.txt', dtype=float)

        W = tf.Variable(weight, dtype=tf.float64, name='weight')
        b = tf.Variable(bias, dtype=tf.float64, name='bias')
        y = tf.nn.softmax(tf.matmul(x, W) + b, name='predict_y')

        with tf.name_scope('cross_entropy'):
            cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
            cross_entropy_for_sorting = -tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1])

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()

        outlier_correct_prediction = sess.run(correct_prediction, feed_dict={x: outlier_x, y_: outlier_y}).reshape(-1, 1)

        false_classified_benign_outlier = 0
        false_classified_mal_outlier = 0
        total_benign_outlier = 0
        total_mal_outlier = 0
        for j in range(outlier_y.shape[0]):
            if outlier_y[j][0] == 1:
                is_benign = True
            else:
                is_benign = False
            if is_benign:
                total_benign_outlier += 1
                if not outlier_correct_prediction[j]:
                    false_classified_benign_outlier += 1
            else:
                total_mal_outlier += 1
                if not outlier_correct_prediction[j]:
                    false_classified_mal_outlier += 1
    file.writelines('Rule {0}, {1}, {2}, {3}, ({4}/{5},{6}/{7}), {8}, {9}\n'.format(rule, input_amount, method, h_n, false_classified_benign_outlier, total_benign_outlier, false_classified_mal_outlier, total_mal_outlier, execute_time, bp_times))

    # svm
    target_path = r"{0}\svm\rule_{1}".format(path, i)
    rule = i
    method = 'svm'
    h_n = 0
    with open(target_path + r'\_two_class_training_detail.txt') as f1:
        while 1:
            line = f1.readline()
            if not line:
                break
            if "training data amount: " in line:
                input_amount = int(line.replace("training data amount: ", ""))
            if "execute time: " in line:
                execute_time = line.replace("execute time: ", "")
                execute_time = execute_time.replace(" sec\n", "")
            if "thinking times count: " in line:
                bp_times = int(line.replace("thinking times count: ", ""))
    training_data_order_x_y = np.loadtxt(target_path + r'\training_data_order_x_y.txt', dtype=float, delimiter=' ')
    outlier_index = int(training_data_order_x_y.shape[0] * 0.95)
    outlier_data = training_data_order_x_y[outlier_index:]
    outlier_x = outlier_data[:, 1:-1]
    outlier_y = outlier_data[:, -1]
    outlier_order = outlier_data[:, 0]

    false_classified_benign_outlier = 0
    false_classified_mal_outlier = 0
    total_benign_outlier = 0
    total_mal_outlier = 0
    for j in range(outlier_y.shape[0]):
        if outlier_y[j] == 1:
            is_benign = True
        else:
            is_benign = False
        if is_benign:
            total_benign_outlier += 1
            if outlier_order[j] >= 0:
                false_classified_benign_outlier += 1
        else:
            total_mal_outlier += 1
            if outlier_order[j] >= 0:
                false_classified_mal_outlier += 1
    file.writelines('Rule {0}, {1}, {2}, {3}, ({4}/{5},{6}/{7}), {8}, {9}\n'.format(rule, input_amount, method, h_n,
                                                                                    false_classified_benign_outlier,
                                                                                    total_benign_outlier,
                                                                                    false_classified_mal_outlier,
                                                                                    total_mal_outlier, execute_time,
                                                                                    bp_times))

    # env
    target_path = r"{0}\env\rule_{1}".format(path, i)
    rule = i
    method = 'env'
    with open(target_path + r'\_training_detail.txt') as f1:
        while 1:
            line = f1.readline()
            if not line:
                break
            if "training data amount: " in line:
                input_amount = int(line.replace("training_data_amount: ", ""))
            if "hidden_node_amount: " in line:
                h_n = int(line.replace("hidden_node_amount: ", ""))
            if "total execution time: " in line:
                execute_time = line.replace("total execution time: ", "")
                execute_time = execute_time.replace(" seconds\n", "")
            if "thinking_times_count: " in line:
                if "softening" not in line:
                    bp_times = int(line.replace("thinking_times_count: ", ""))
    with tf.Graph().as_default():
        x_placeholder = tf.placeholder(tf.float64)
        training_data_result = np.loadtxt(
            target_path + r"\training_data_residual_predict_output_desire_output_desire_input.txt", dtype=float,
            delimiter=" ")
        outlier_index = int(int(input_amount) * 0.95)
        major_data_x = training_data_result[:outlier_index, 3:]
        major_data_y = training_data_result[:outlier_index, 2].reshape((-1, 1))
        major_data_predict_y = training_data_result[:outlier_index, 1].reshape(
            (-1, 1))
        outlier_y = training_data_result[outlier_index:, 2].reshape((-1, 1))
        outlier_predict_y = training_data_result[outlier_index:, 1].reshape(
            (-1, 1))
        alpha = min(major_data_predict_y[np.where(major_data_y == 1)])
        beta = max(major_data_predict_y[np.where(major_data_y == -1)])
        middle_point = (alpha + beta) / 2
        false_classified_benign_outlier = 0
        false_classified_mal_outlier = 0
        total_benign_outlier = 0
        total_mal_outlier = 0
        for j in range(outlier_y.shape[0]):
            if outlier_y[j] == 1:
                is_benign = True
            else:
                is_benign = False
            if is_benign:
                total_benign_outlier += 1
                if outlier_predict_y[j] < middle_point:
                    false_classified_benign_outlier += 1
            else:
                total_mal_outlier += 1
                if outlier_predict_y[j] >= middle_point:
                    false_classified_mal_outlier += 1
    file.writelines('Rule {0}, {1}, {2}, {3}, ({4}/{5},{6}/{7}), {8}, {9}\n'.format(rule, input_amount, method, h_n,
                                                                                    false_classified_benign_outlier,
                                                                                    total_benign_outlier,
                                                                                    false_classified_mal_outlier,
                                                                                    total_mal_outlier, execute_time,
                                                                                    bp_times))

    # bml
    target_path = r"{0}\bml\rule_{1}".format(path, i)
    rule = i
    method = 'bml'
    with open(target_path + r'\_two_class_training_detail.txt') as f1:
        while 1:
            line = f1.readline()
            if not line:
                break
            if "training_data_amount: " in line:
                input_amount = int(line.replace("training_data_amount: ", ""))
            if "hidden_node_amount: " in line:
                h_n = int(line.replace("hidden_node_amount: ", ""))
            if "total execution time: " in line:
                execute_time = line.replace("total execution time: ", "")
                execute_time = execute_time.replace(" seconds\n", "")
            if "thinking_times_count: " in line:
                if "softening" not in line:
                    bp_times = int(line.replace("thinking_times_count: ", ""))
            if 'classify middle point' in line:
                middle_point = float(line.split(': ')[1])
    training_data_result = np.loadtxt(target_path + r"\two_class_training_data_distance_x_y_yp.txt", dtype=float, delimiter=" ")
    outlier_index = int(int(input_amount) * 0.95)
    outlier_y = training_data_result[outlier_index:, -2].reshape((-1, 1))
    outlier_predict_y = training_data_result[outlier_index:, -1].reshape((-1, 1))
    false_classified_benign_outlier = 0
    false_classified_mal_outlier = 0
    total_benign_outlier = 0
    total_mal_outlier = 0
    for j in range(outlier_y.shape[0]):
        if outlier_y[j] == 1:
            is_benign = True
        else:
            is_benign = False
        if is_benign:
            total_benign_outlier += 1
            if outlier_predict_y[j] < middle_point:
                false_classified_benign_outlier += 1
        else:
            total_mal_outlier += 1
            if outlier_predict_y[j] >= middle_point:
                false_classified_mal_outlier += 1
    file.writelines('Rule {0}, {1}, {2}, {3}, ({4}/{5},{6}/{7}), {8}, {9}\n'.format(rule, input_amount, method, h_n,
                                                                                    false_classified_benign_outlier,
                                                                                    total_benign_outlier,
                                                                                    false_classified_mal_outlier,
                                                                                    total_mal_outlier, execute_time,
                                                                                    bp_times))
file.close()

# training/testing FP/FN
# bml
file = open(new_path + r"\bml_train_test_analyze.txt", 'w')
file.writelines('Rule, Train FP, Train FN, Test FP, Test FN\n')
for i in range(1, 10):
    target_path = r"{0}\bml\rule_{1}".format(path, i)
    o_t = np.loadtxt(target_path + r'\two_class_output_threshold.txt', dtype=float, delimiter=' ').reshape(1)
    o_w = np.loadtxt(target_path+r'\two_class_output_neuron_weight.txt', dtype=float, delimiter=' ').reshape((-1,1))
    h_t = np.loadtxt(target_path + r'\two_class_hidden_threshold.txt', dtype=float, delimiter=' ').reshape((1, -1))
    h_w = np.loadtxt(target_path + r'\two_class_hidden_neuron_weight.txt', dtype=float, delimiter=' ').reshape((-1, h_t.shape[1]))
    with tf.Graph().as_default():
        x_placeholder = tf.placeholder(tf.float64)
        hidden_thresholds = tf.Variable(h_t, dtype=tf.float64)
        hidden_weights = tf.Variable(h_w, dtype=tf.float64)
        hidden_layer = tf.tanh(tf.add(tf.matmul(x_placeholder, hidden_weights), hidden_thresholds))
        output_threshold = tf.Variable(o_t, dtype=tf.float64)
        output_weights = tf.Variable(o_w, dtype=tf.float64)
        output_layer = tf.add(tf.matmul(hidden_layer, output_weights), output_threshold)
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        with open(target_path + r'\_two_class_training_detail.txt') as f1:
            while 1:
                line = f1.readline()
                if not line:
                    break
                if 'classify middle point' in line:
                    middle_point = float(line.split(': ')[1])
                    break
        target_path = r"{0}\data\rule_{1}".format(path, i)
        mal_training_data = np.loadtxt(target_path + r'\mal_training_data.txt', dtype=float, delimiter=' ')
        mal_testing_data = np.loadtxt(target_path + r'\mal_testing_data.txt', dtype=float, delimiter=' ')
        benign_training_data = np.loadtxt(target_path + r'\benign_training_data.txt', dtype=float, delimiter=' ')
        benign_testing_data = np.loadtxt(target_path + r'\benign_testing_data.txt', dtype=float, delimiter=' ')
        mal_train_result = np.array(sess.run([output_layer], {x_placeholder: mal_training_data}))
        mal_test_result = np.array(sess.run([output_layer], feed_dict={x_placeholder: mal_testing_data}))
        benign_train_result = np.array(sess.run([output_layer], feed_dict={x_placeholder: benign_training_data}))
        benign_test_result = np.array(sess.run([output_layer], feed_dict={x_placeholder: benign_testing_data}))

        false_classified_mal_train_count = mal_train_result[np.where(mal_train_result >= middle_point)].shape[0]
        false_classified_mal_test_count = mal_test_result[np.where(mal_test_result >= middle_point)].shape[0]
        false_classified_benign_train_count = benign_train_result[np.where(benign_train_result < middle_point)].shape[0]
        false_classified_benign_test_count = benign_test_result[np.where(benign_test_result < middle_point)].shape[0]
    file.writelines('Rule {0}, {1}/{2}, {3}/{4}, {5}/{6}, {7}/{8}\n'.format(i, false_classified_benign_train_count, benign_training_data.shape[0], false_classified_mal_train_count, mal_training_data.shape[0], false_classified_benign_test_count, benign_testing_data.shape[0], false_classified_mal_test_count, mal_testing_data.shape[0]))
file.close()

# svm
file = open(new_path + r"\svm_train_test_analyze.txt", 'w')
file.writelines('Rule, Train FP, Train FN, Test FP, Test FN\n')
for i in range(1, 10):
    target_path = r"{0}\svm\rule_{1}".format(path, i)
    with open(target_path + r'\_training_analyze.txt') as f1:
        while 1:
            line = f1.readline()
            if not line:
                break
            if "benign accuracy: " in line:
                benign_train_acc = line.replace("benign accuracy: ", "")
                benign_train_acc = benign_train_acc.split(' , ')[0]
            if "mal accuracy: " in line:
                mal_train_acc = line.replace("mal accuracy: ", "")
                mal_train_acc = mal_train_acc.split(' , ')[0]
    with open(target_path + r'\_testing_analyze.txt') as f1:
        while 1:
            line = f1.readline()
            if not line:
                break
            if "benign accuracy: " in line:
                benign_test_acc = line.replace("benign accuracy: ", "")
                benign_test_acc = benign_test_acc.split(' , ')[0]
            if "mal accuracy: " in line:
                mal_test_acc = line.replace("mal accuracy: ", "")
                mal_test_acc = mal_test_acc.split(' , ')[0]
    one = int(benign_train_acc.split('/')[1]) - int(benign_train_acc.split('/')[0])
    two = int(benign_train_acc.split('/')[1])
    three = int(mal_train_acc.split('/')[1]) - int(mal_train_acc.split('/')[0])
    four = int(mal_train_acc.split('/')[1])
    five = int(benign_test_acc.split('/')[1]) - int(benign_test_acc.split('/')[0])
    six = int(benign_test_acc.split('/')[1])
    seven = int(mal_test_acc.split('/')[1]) - int(mal_test_acc.split('/')[0])
    eight = int(mal_test_acc.split('/')[1])
    file.writelines('Rule {0}, {1}/{2}, {3}/{4}, {5}/{6}, {7}/{8}\n'.format(i, one, two, three, four, five, six, seven, eight))
file.close()

# env
file = open(new_path + r"\env_train_test_analyze.txt", 'w')
file.writelines('Rule, Train FP, Train FN, Test FP, Test FN\n')
for i in range(1, 10):
    target_path = r"{0}\env\rule_{1}".format(path, i)
    o_t = np.loadtxt(target_path + r'\output_threshold.txt', dtype=float, delimiter=' ').reshape(1)
    o_w = np.loadtxt(target_path+r'\output_neuron_weight.txt', dtype=float, delimiter=' ').reshape((-1,1))
    h_t = np.loadtxt(target_path + r'\hidden_threshold.txt', dtype=float, delimiter=' ').reshape((1, -1))
    h_w = np.loadtxt(target_path + r'\hidden_neuron_weight.txt', dtype=float, delimiter=' ').reshape((-1, h_t.shape[1]))
    with tf.Graph().as_default():
        x_placeholder = tf.placeholder(tf.float64)
        hidden_thresholds = tf.Variable(h_t, dtype=tf.float64)
        hidden_weights = tf.Variable(h_w, dtype=tf.float64)
        hidden_layer = tf.tanh(tf.add(tf.matmul(x_placeholder, hidden_weights), hidden_thresholds))
        output_threshold = tf.Variable(o_t, dtype=tf.float64)
        output_weights = tf.Variable(o_w, dtype=tf.float64)
        output_layer = tf.add(tf.matmul(hidden_layer, output_weights), output_threshold)
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        training_data_result = np.loadtxt(
            target_path + r"\training_data_residual_predict_output_desire_output_desire_input.txt", dtype=float,
            delimiter=" ")
        outlier_index = int(int(input_amount) * 0.95)
        major_data_y = training_data_result[:outlier_index, 2].reshape((-1, 1))
        major_data_predict_y = training_data_result[:outlier_index, 1].reshape((-1, 1))
        alpha = min(major_data_predict_y[np.where(major_data_y == 1)])
        beta = max(major_data_predict_y[np.where(major_data_y == -1)])
        middle_point = (alpha + beta) / 2

        target_path = r"{0}\data\rule_{1}".format(path, i)
        mal_training_data = np.loadtxt(target_path + r'\mal_training_data.txt', dtype=float, delimiter=' ')
        mal_testing_data = np.loadtxt(target_path + r'\mal_testing_data.txt', dtype=float, delimiter=' ')
        benign_training_data = np.loadtxt(target_path + r'\benign_training_data.txt', dtype=float, delimiter=' ')
        benign_testing_data = np.loadtxt(target_path + r'\benign_testing_data.txt', dtype=float, delimiter=' ')
        mal_train_result = np.array(sess.run([output_layer], {x_placeholder: mal_training_data}))
        mal_test_result = np.array(sess.run([output_layer], feed_dict={x_placeholder: mal_testing_data}))
        benign_train_result = np.array(sess.run([output_layer], feed_dict={x_placeholder: benign_training_data}))
        benign_test_result = np.array(sess.run([output_layer], feed_dict={x_placeholder: benign_testing_data}))

        false_classified_mal_train_count = mal_train_result[np.where(mal_train_result >= middle_point)].shape[0]
        false_classified_mal_test_count = mal_test_result[np.where(mal_test_result >= middle_point)].shape[0]
        false_classified_benign_train_count = benign_train_result[np.where(benign_train_result < middle_point)].shape[0]
        false_classified_benign_test_count = benign_test_result[np.where(benign_test_result < middle_point)].shape[0]
    file.writelines('Rule {0}, {1}/{2}, {3}/{4}, {5}/{6}, {7}/{8}\n'.format(i, false_classified_benign_train_count, benign_training_data.shape[0], false_classified_mal_train_count, mal_training_data.shape[0], false_classified_benign_test_count, benign_testing_data.shape[0], false_classified_mal_test_count, mal_testing_data.shape[0]))
file.close()

# softmax
file = open(new_path + r"\softmax_train_test_analyze.txt", 'w')
file.writelines('Rule, Train FP, Train FN, Test FP, Test FN\n')
for i in range(1, 10):
    target_path = r"{0}\softmax\rule_{1}".format(path, i)
    bias = np.loadtxt(target_path + r'\two_class_bias.txt', dtype=float, delimiter=' ').reshape(2)
    weight = np.loadtxt(target_path+r'\two_class_weight.txt', dtype=float, delimiter=' ').reshape((-1,2))
    with tf.Graph().as_default():
        with tf.name_scope('placeholder_x'):
            x = tf.placeholder(tf.float64, name='x_placeholder')
        with tf.name_scope('placeholder_y'):
            y_ = tf.placeholder(tf.float64, name='y_placeholder')
        W = tf.Variable(weight, dtype=tf.float64, name='weight')
        b = tf.Variable(bias, dtype=tf.float64, name='bias')
        y = tf.nn.softmax(tf.matmul(x, W) + b, name='predict_y')
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()

        target_path = r"{0}\data\rule_{1}".format(path, i)
        mal_training_data = np.loadtxt(target_path + r'\mal_training_data.txt', dtype=float, delimiter=' ')
        mal_testing_data = np.loadtxt(target_path + r'\mal_testing_data.txt', dtype=float, delimiter=' ')
        benign_training_data = np.loadtxt(target_path + r'\benign_training_data.txt', dtype=float, delimiter=' ')
        benign_testing_data = np.loadtxt(target_path + r'\benign_testing_data.txt', dtype=float, delimiter=' ')

        y_benign_element = np.zeros(2)
        y_mal_element = np.zeros(2)
        y_benign_element[0] = 1
        y_mal_element[1] = 1
        mal_train_result = np.array(sess.run(correct_prediction, {x: mal_training_data, y_: np.tile(y_mal_element, mal_training_data.shape[0]).reshape(-1, 2)}))
        mal_test_result = np.array(sess.run(correct_prediction, feed_dict={x: mal_testing_data, y_: np.tile(y_mal_element, mal_testing_data.shape[0]).reshape(-1, 2)}))
        benign_train_result = np.array(sess.run(correct_prediction, feed_dict={x: benign_training_data, y_: np.tile(y_benign_element, benign_training_data.shape[0]).reshape(-1, 2)}))
        benign_test_result = np.array(sess.run(correct_prediction, feed_dict={x: benign_testing_data, y_: np.tile(y_benign_element, benign_testing_data.shape[0]).reshape(-1, 2)}))

        false_classified_mal_train_count = mal_train_result[np.where(~mal_train_result)].shape[0]
        false_classified_mal_test_count = mal_test_result[np.where(~mal_test_result)].shape[0]
        false_classified_benign_train_count = benign_train_result[np.where(~benign_train_result)].shape[0]
        false_classified_benign_test_count = benign_test_result[np.where(~benign_test_result)].shape[0]
    file.writelines('Rule {0}, {1}/{2}, {3}/{4}, {5}/{6}, {7}/{8}\n'.format(i, false_classified_benign_train_count, benign_training_data.shape[0], false_classified_mal_train_count, mal_training_data.shape[0], false_classified_benign_test_count, benign_testing_data.shape[0], false_classified_mal_test_count, mal_testing_data.shape[0]))
file.close()
