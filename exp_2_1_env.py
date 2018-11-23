# coding:utf-8
import tensorflow as tf
import numpy as np
import time
import math
import os

dir_path = os.path.dirname(os.path.realpath(__file__))

sample_rate = 0.8

# 所有sample 9種
for owl in range(6, 7):
    with tf.Graph().as_default():
        # read all owl data from file
        owl_rule = str(owl)
        print('rule {0}'.format(owl_rule))
        file_name = "rule_" + owl_rule
        file_input = file_name
        # mal_data = np.loadtxt(file_name + ".txt", dtype=float, delimiter=" ")
        # benign_data = np.loadtxt("owl_benign_samples.txt", dtype=float, delimiter=" ")
        # np.random.shuffle(mal_data)
        # np.random.shuffle(benign_data)
        # mal_data_amount = mal_data.shape[0]
        # mal_training_data = mal_data[:int(mal_data_amount * sample_rate)]
        # benign_training_data = benign_data[:int(mal_data_amount * sample_rate)]
        # mal_testing_data = mal_data[int(mal_data_amount * sample_rate):]
        # benign_testing_data = benign_data[int(mal_data_amount * sample_rate):]

        # 從事先抽好的train中拿資料
        mal_training_data = np.loadtxt(r"exp_2_1\data\rule_{0}\mal_training_data.txt".format(owl_rule), dtype=float,
                                       delimiter=" ")
        benign_training_data = np.loadtxt(r"exp_2_1\data\rule_{0}\benign_training_data.txt".format(owl_rule),
                                          dtype=float, delimiter=" ")

        # create folder to save training process
        new_path = r"{0}\exp_2_1".format(dir_path)
        if not os.path.exists(new_path):
            os.makedirs(new_path)
        new_path = r"{0}\env".format(new_path)
        if not os.path.exists(new_path):
            os.makedirs(new_path)
        new_path = r"{0}\rule_{1}".format(new_path, owl_rule)
        if not os.path.exists(new_path):
            os.makedirs(new_path)

        # np.savetxt(r'{0}\mal_training_data.txt'.format(new_path), mal_training_data, delimiter=' ')
        # np.savetxt(r'{0}\mal_testing_data.txt'.format(new_path), mal_testing_data, delimiter=' ')
        # np.savetxt(r'{0}\benign_training_data.txt'.format(new_path), benign_training_data, delimiter=' ')
        # np.savetxt(r'{0}\benign_testing_data.txt'.format(new_path), benign_testing_data, delimiter=' ')

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

        # execute_start_time = time.time()

        # Network Parameters
        input_node_amount = x_training_data.shape[1]
        hidden_node_amount = 1
        output_node_amount = 1
        learning_rate_eta = 0.005

        # Parameters
        every_stage_max_thinking_times = 100000
        data_size = x_training_data.shape[0]
        outlier_rate = 0.05
        # square_residual_tolerance = 0.5
        zeta = 0.05
        Lambda = 10000
        sigma_multiplier = 2

        # create file to save training process
        training_process = open(new_path + r"\_training_process.txt", 'w')

        # counters
        thinking_times_count = 0
        cramming_times_count = 0
        softening_thinking_times_count = 0
        pruning_success_times_count = 0

        # init = tf.global_variables_initializer()
        sess = tf.Session()
        # sess.run(init)

        with tf.name_scope('calculate_envelope_width'):
            # 算出envelope width: epsilon
            opt = sess.run(tf.matrix_solve_ls(x_training_data, y_training_data, fast=False, name='solve_matrix'))
            # 算出所有資料用此模型得到的輸出值
            opt_output = sess.run(tf.matmul(x_training_data, opt, name='matmul'))
            # 輸出值減掉實際的y值後取絕對值，得到此模型的差的矩陣
            opt_distance = sess.run(tf.abs(opt_output - y_training_data, name='abs'))
            # 取得差矩陣的平均值(用不到)以及變異數
            mean, var = tf.nn.moments(tf.stack(opt_distance, name='stack'), axes=[0], name='get_var')
            # stander deviation(全體資料的線性迴歸的標準差)
            sigma = sess.run(tf.sqrt(var, name='sqrt'))
            # envelope width(可以調整幾倍的標準差e.g. 2*sigma是95%的資料)
            epsilon = sigma_multiplier * sigma

            if epsilon < 0.0000001:
                epsilon = 0.0000001
            elif epsilon > 0.9:
                epsilon = 0.9

        with tf.name_scope('calculate_first_slfn_weights'):
            # 首先架構初始SLFN
            m = input_node_amount
            # 第一次取m+1筆資料算出first SLFN 的初始權重，算法是做矩陣列運算解聯立方程式，讓前m+1筆資料都可以完美符合這個模型
            # m+1筆資料(x,y)，m+1個變數(m個weight，1個hidden threshold)，解聯立方程式，得到正確答案
            # 取前m+1筆y資料，並且套公式給定output weight和threshold
            # 不知道為什麼output weight & threshold要這樣給，給weight 1 ,threshold 0不好嗎? 可問蔡老師
            first_slfn_output_weight = (np.max(y_training_data) - np.min(y_training_data) + 2.0).reshape(1, 1)
            first_slfn_output_threshold = (np.min(y_training_data) - 1.0).reshape(1)
            # print(first_slfn_output_weight)
            # print(first_slfn_output_threshold)
            desi_slice_y = y_training_data[:m+1]
            # print(desi_slice_y.shape)
            # 取得x經過運算後應該得到的hidden value(做tanh運算之前)
            yc = np.arctanh((desi_slice_y - first_slfn_output_threshold) / first_slfn_output_weight).reshape(m+1, 1)
            # print(yc.shape)
            # 對應給定的output weight & threshold，解hidden weight & threshold的聯立方程式
            desi_slice_x = x_training_data[:m+1]
            # 由於x原本只有m維，所以要加上1倍的threshold來變成m+1個變數，m+1筆資料，解方程式
            hidden_node_threshold_vector = tf.ones([m + 1, 1], dtype=tf.float64, name='one')
            xc = sess.run(tf.concat(axis=1, values=[desi_slice_x, hidden_node_threshold_vector]))
            # print(xc.shape)
            # 使用tf.matrix_solve_ls做矩陣運算解聯立方程式得到hidden weight & threshold
            answer = sess.run(tf.matrix_solve_ls(xc, yc, fast=False))
            # answer的前m個是hidden weight 最後一個是hidden threshold
            first_slfn_hidden_weight = answer[:m]
            first_slfn_hidden_threshold = answer[m:]

        # 架構第一個SLFN的tensor
        # placeholders
        with tf.name_scope('inputs'):
            x_placeholder = tf.placeholder(tf.float64, name='x_input')
            y_placeholder = tf.placeholder(tf.float64, name='y_input')

        # network architecture
        with tf.name_scope('hidden_layer'):
            hidden_thresholds = tf.Variable(first_slfn_hidden_threshold, dtype=tf.float64, name='hidden_threshold')
            hidden_weights = tf.Variable(first_slfn_hidden_weight, dtype=tf.float64, name='hidden_weight')
            hidden_layer = tf.tanh(tf.add(tf.matmul(x_placeholder, hidden_weights), hidden_thresholds))
        with tf.name_scope('output_layer'):
            output_threshold = tf.Variable(first_slfn_output_threshold, dtype=tf.float64, name='output_threshold')
            output_weights = tf.Variable(first_slfn_output_weight, dtype=tf.float64, name='hidden_weight')
            output_layer = tf.add(tf.matmul(hidden_layer, output_weights), output_threshold)

        # learning goal & optimizer
        with tf.name_scope('loss'):
            average_square_residual = tf.reduce_mean(tf.reduce_sum(tf.square(y_placeholder - output_layer), reduction_indices=[1]))
        with tf.name_scope('train'):
            train = tf.train.GradientDescentOptimizer(learning_rate_eta).minimize(average_square_residual)

        # saver
        saver = tf.train.Saver()

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        with tf.name_scope('calculate_alpha_T'):
            # alpha_T的算法，會用在加hidden node的時候
            beta_k_placeholder = tf.placeholder(tf.float64, name='beta_k')
            x_c_placeholder = tf.placeholder(tf.float64, name='x_c')
            x_k_placeholder = tf.placeholder(tf.float64, name='x_k')
            test = tf.sqrt(tf.reduce_sum(tf.square(beta_k_placeholder, name='square'), name='reduce_sum'), name='sqrt')
            alpha = tf.div(beta_k_placeholder, test, name='alpha')
            alpha_T = tf.transpose(alpha, name='transpose')
            Cal_table2 = tf.reduce_sum(tf.matmul(tf.subtract(x_c_placeholder, x_k_placeholder), alpha_T), name='test_alpha')

        with tf.name_scope('calculate_new_hidden_threshold'):
            # new hidden node threshold的算法
            alpha_T_placeholder = tf.placeholder(tf.float64, name='alpha_T')
            with tf.name_scope('hidden_threshold_1'):
                calculate_new_hidden_node_1_threshold = zeta - Lambda * tf.matmul(x_k_placeholder, alpha_T_placeholder)
            with tf.name_scope('hidden_threshold_2'):
                calculate_new_hidden_node_2_threshold = zeta + Lambda * tf.matmul(x_k_placeholder, alpha_T_placeholder)

        with tf.name_scope('calculate_new_output_weight'):
            # calculate new output weight
            y_k_minus_output_placeholder = tf.placeholder(tf.float64, name='y_k_minus_output')
            two = tf.cast(2.0, tf.float64, name='2')
            calculate_new_output_weight = y_k_minus_output_placeholder / (two * tf.cast(tf.tanh(zeta), tf.float64))

        # 如果想看所有default graph裡面的node 可以用下面這段code
        # for node in tf.get_default_graph().as_graph_def().node:
        #     print(node.name)
        execute_start_time = time.time()
        for n in range(m+2, int(data_size * (1 - outlier_rate) + 1)):
            print('-----stage: ' + str(n) + '-----')
            training_process.writelines('-----stage: ' + str(n) + '-----' + "\n")

            # pick n data of smallest residual
            predict_y = sess.run([output_layer], {x_placeholder: x_training_data, y_placeholder: y_training_data})
            square_residuals = np.square(predict_y[0] - y_training_data)
            # print(square_residuals)
            # concat residual & origin data, sort by residual, depart residual
            concat_x_and_y = np.concatenate((x_training_data, y_training_data), axis=1)
            concat_residual_and_x_y = np.concatenate((square_residuals, concat_x_and_y), axis=1)
            sort_result = concat_residual_and_x_y[np.argsort(concat_residual_and_x_y[:, 0])]
            x_training_data_sort_by_residual = np.delete(sort_result, (0, m + 1), axis=1)  # 去除0和m+1欄
            y_training_data_sort_by_residual = np.delete(sort_result, slice(0, m + 1), axis=1)  # 去除從0到m欄
            # print(concat_x_and_y)
            # print(concat_residual_and_x_y)
            # print(x_training_data_sort_by_residual)
            # print(y_training_data_sort_by_residual)

            # take first n row of data, this stage use these data to train the NN
            current_stage_x_training_data = x_training_data_sort_by_residual[:n]
            current_stage_y_training_data = y_training_data_sort_by_residual[:n]
            # print(current_stage_x_training_data)
            # print(current_stage_y_training_data)

            # calculate (y - predict_y) ^ 2 and check the residuals are smaller than tolerance
            predict_y = sess.run([output_layer], {x_placeholder: current_stage_x_training_data, y_placeholder: current_stage_y_training_data})

            # print(predict_y[0])
            # print(current_stage_y_training_data)
            # print(np.square(current_stage_y_training_data-predict_y[0]))
            current_stage_square_residuals = np.square(current_stage_y_training_data - predict_y[0])

            # print(epsilon.shape)
            # print(current_stage_square_residuals.shape)
            if all(square_residual < epsilon ** 2 for square_residual in current_stage_square_residuals):
                print('new training case can be classified without additional action.')
                training_process.writelines('new training case can be classified without additional action.' + "\n")
            else:
                print('new training case larger than epsilon, apply GradientDescent to change weights & thresholds.')
                training_process.writelines('new training case larger than epsilon, apply GradientDescent to change weights & thresholds.' + "\n")
                # BP
                print('start BP.')
                training_process.writelines('start BP.' + "\n")
                bp_failed = False
                saver.save(sess, r"{0}/model.ckpt".format(dir_path))
                last_max_squared_residual = 9999999
                for stage in range(every_stage_max_thinking_times):
                    sess.run(train, feed_dict={x_placeholder: current_stage_x_training_data,y_placeholder: current_stage_y_training_data})
                    thinking_times_count += 1

                    predict_y = sess.run([output_layer], {x_placeholder: current_stage_x_training_data, y_placeholder: current_stage_y_training_data})
                    current_stage_square_residuals = np.square(current_stage_y_training_data - predict_y[0])

                    current_stage_max_square_residual = max(current_stage_square_residuals)
                    if math.isnan(current_stage_max_square_residual):
                        print(current_stage_max_square_residual)
                        bp_failed = True
                        print('BP failed: after {0} times training, residual is nan.'.format(
                            (stage + 1)))
                        training_process.writelines(
                            'BP failed: after {0} times training, residual is nan.'.format(
                                (stage + 1)) + "\n")
                        # MUST restore before cramming(因為調權重可能會讓先前的資料違反condition L)
                        saver.restore(sess, r"{0}/model.ckpt".format(dir_path))
                        print('restore weights.')
                        training_process.writelines('restore weights.' + "\n")
                        break

                    if stage % 500 == 0 and stage != 0:
                        current_stage_max_square_residual = max(current_stage_square_residuals)
                        print(current_stage_max_square_residual)

                        if current_stage_max_square_residual - last_max_squared_residual < -0.05 * current_stage_max_square_residual:
                            last_max_squared_residual = current_stage_max_square_residual
                        else:
                            bp_failed = True
                            print('BP failed: after {0} times training, residual change too slow.'.format(
                                (stage + 1)))
                            training_process.writelines(
                                'BP failed: after {0} times training, residual change too slow.'.format(
                                    (stage + 1)) + "\n")
                            # MUST restore before cramming(因為調權重可能會讓先前的資料違反condition L)
                            saver.restore(sess, r"{0}/model.ckpt".format(dir_path))
                            print('restore weights.')
                            training_process.writelines('restore weights.' + "\n")
                            break
                    if all(square_residual < epsilon ** 2 for square_residual in current_stage_square_residuals):
                        print('BP {0} times, all this stage training data meet the condition square residual^2 < epsilon^2, thinking success!!!'.format((stage+1)))
                        training_process.writelines('BP {0} times, all this stage training data meet the condition square residual^2 < epsilon^2, thinking success!!!'.format((stage+1)) + "\n")
                        break
                    else:
                        if stage == (every_stage_max_thinking_times - 1):
                            bp_failed = True
                            print('BP failed: after {0} times training, residual still larger than tolerance.'.format((stage + 1)))
                            training_process.writelines('BP failed: after {0} times training, residual still larger than tolerance.'.format((stage + 1)) + "\n")
                            # MUST restore before cramming(因為調權重可能會讓先前的資料違反condition L)
                            saver.restore(sess, r"{0}/model.ckpt".format(dir_path))
                            print('restore weights.')
                            training_process.writelines('restore weights.' + "\n")

                if bp_failed:
                    # add two hidden nodes to make the new training case square residual less than tolerance
                    print('add two hidden nodes')
                    training_process.writelines('add two hidden nodes.' + "\n")
                    cramming_times_count += 1
                    hidden_node_amount += 2
                    # calculate relevant parameters
                    # 取得現有的weight&threshold陣列
                    current_hidden_weights, current_hidden_thresholds, current_output_weights, current_output_threshold = sess.run([hidden_weights, hidden_thresholds, output_weights, output_threshold], {x_placeholder: current_stage_x_training_data, y_placeholder: current_stage_y_training_data})
                    predict_y = sess.run([output_layer], {x_placeholder: current_stage_x_training_data, y_placeholder: current_stage_y_training_data})
                    # print('current hidden weights:')
                    # print(current_hidden_weights)
                    # print('current hidden thresholds:')
                    # print(current_hidden_thresholds)
                    # print('current output weights:')
                    # print(current_output_weights)
                    # print('current output threshold:')
                    # print(current_output_threshold)

                    x_c = current_stage_x_training_data[:n - 1]
                    x_k = current_stage_x_training_data[n - 1:]
                    y_k = current_stage_y_training_data[n - 1:]
                    # print(x_c.shape)
                    # print(x_k.shape)
                    # input()

                    # calculate new hidden weight
                    alpha_success = False
                    while not alpha_success:
                        beta_k = np.random.random_sample((1, m)) + 1
                        if sess.run([Cal_table2], {beta_k_placeholder: beta_k, x_c_placeholder: x_c, x_k_placeholder: x_k})[0] != 0:
                            alpha_success = True

                    current_stage_alpha_T = sess.run([alpha_T], {beta_k_placeholder: beta_k})[0]
                    # print(current_stage_alpha_T)
                    new_hidden_node_1_neuron_weights = Lambda * current_stage_alpha_T
                    new_hidden_node_2_neuron_weights = -Lambda * current_stage_alpha_T
                    # print('new hidden node 1 weights:')
                    # print(new_hidden_node_1_neuron_weights)
                    # print('new hidden node 2 weights:')
                    # print(new_hidden_node_2_neuron_weights)

                    # calculate new hidden threshold
                    new_hidden_node_1_threshold = sess.run([calculate_new_hidden_node_1_threshold], {x_k_placeholder: x_k, alpha_T_placeholder: current_stage_alpha_T})[0].reshape(1)
                    new_hidden_node_2_threshold = sess.run([calculate_new_hidden_node_2_threshold], {x_k_placeholder: x_k, alpha_T_placeholder: current_stage_alpha_T})[0].reshape(1)
                    # print('new hidden node 1 threshold:')
                    # print(new_hidden_node_1_threshold)
                    # print('new hidden node 2 threshold:')
                    # print(new_hidden_node_2_threshold)

                    # calculate new output weight
                    y_k_output = sess.run([output_layer], {x_placeholder: x_k, y_placeholder: y_k})
                    y_k_minus_output = y_k - y_k_output
                    new_output_weight = sess.run([calculate_new_output_weight], {y_k_minus_output_placeholder: y_k_minus_output})[0].reshape(1, 1)

                    # print('predict value of most recent training case: ' + str(predict_y[0][k - 1]))
                    # print('new output weight:')
                    # print(new_output_node_neuron_weight)

                    # combine weights & thresholds
                    new_hidden_weights_temp = np.append(current_hidden_weights, new_hidden_node_1_neuron_weights.reshape(input_node_amount, 1), axis=1)
                    new_hidden_weights = np.append(new_hidden_weights_temp, new_hidden_node_2_neuron_weights.reshape(input_node_amount, 1), axis=1)
                    # print(new_hidden_weights)
                    new_hidden_thresholds_temp = np.append(current_hidden_thresholds, new_hidden_node_1_threshold)
                    new_hidden_thresholds = np.append(new_hidden_thresholds_temp, new_hidden_node_2_threshold)
                    # print(new_hidden_thresholds)
                    new_output_weights_temp = np.append(current_output_weights, new_output_weight)
                    new_output_weights = np.append(new_output_weights_temp, new_output_weight).reshape(hidden_node_amount, 1)
                    # print(current_output_weights)
                    # print(new_output_weights)

                    # create new graph & session
                    with tf.Graph().as_default():  # Create a new graph, and make it the default.
                        with tf.name_scope('inputs'):
                            # placeholders
                            x_placeholder = tf.placeholder(tf.float64, name='x_input')
                            y_placeholder = tf.placeholder(tf.float64, name='y_input')

                        with tf.name_scope('hidden_layer'):
                            hidden_thresholds = tf.Variable(new_hidden_thresholds, name='hidden_threshold')
                            hidden_weights = tf.Variable(new_hidden_weights, name='hidden_weight')
                            hidden_layer = tf.tanh(tf.add(tf.matmul(x_placeholder, hidden_weights), hidden_thresholds))

                        with tf.name_scope('output_layer'):
                            # network architecture
                            output_threshold = tf.Variable(current_output_threshold, name='output_threshold')
                            output_weights = tf.Variable(new_output_weights, name='output_weight')
                            output_layer = tf.add(tf.matmul(hidden_layer, output_weights), output_threshold)

                        # learning goal & optimizer
                        with tf.name_scope('loss'):
                            average_square_residual = tf.reduce_mean(tf.reduce_sum(tf.square(y_placeholder - output_layer), reduction_indices=[1]))
                        with tf.name_scope('train'):
                            train = tf.train.GradientDescentOptimizer(learning_rate_eta).minimize(average_square_residual)

                        # saver
                        saver = tf.train.Saver()

                        with tf.name_scope('calculate_alpha_T'):
                            # alpha_T的算法，會用在加hidden node的時候
                            beta_k_placeholder = tf.placeholder(tf.float64, name='beta_k')
                            x_c_placeholder = tf.placeholder(tf.float64, name='x_c')
                            x_k_placeholder = tf.placeholder(tf.float64, name='x_k')
                            test = tf.sqrt(tf.reduce_sum(tf.square(beta_k_placeholder, name='square'), name='reduce_sum'), name='sqrt')
                            alpha = tf.div(beta_k_placeholder, test, name='alpha')
                            alpha_T = tf.transpose(alpha, name='transpose')
                            Cal_table2 = tf.reduce_sum(tf.matmul(tf.subtract(x_c_placeholder, x_k_placeholder), alpha_T), name='test_alpha')

                        with tf.name_scope('calculate_new_hidden_threshold'):
                            # new hidden node threshold的算法
                            alpha_T_placeholder = tf.placeholder(tf.float64, name='alpha_T')
                            with tf.name_scope('hidden_threshold_1'):
                                calculate_new_hidden_node_1_threshold = zeta - Lambda * tf.matmul(x_k_placeholder,
                                                                                                  alpha_T_placeholder)
                            with tf.name_scope('hidden_threshold_2'):
                                calculate_new_hidden_node_2_threshold = zeta + Lambda * tf.matmul(x_k_placeholder,
                                                                                                  alpha_T_placeholder)

                        with tf.name_scope('calculate_new_output_weight'):
                            # calculate new output weight
                            y_k_minus_output_placeholder = tf.placeholder(tf.float64, name='y_k_minus_output')
                            two = tf.cast(2.0, tf.float64, name='2')
                            calculate_new_output_weight = y_k_minus_output_placeholder / (two * tf.cast(tf.tanh(zeta), tf.float64))

                        init = tf.global_variables_initializer()
                        sess = tf.Session()
                        sess.run(init)

                        # # verify add hidden node effect
                        # predict_y = sess.run([output_layer], {x_placeholder: current_stage_x_training_data,
                        #                                       y_placeholder: current_stage_y_training_data})[0]
                        # print('predict y:')
                        # print(predict_y)
                        # print('origin y:')
                        # print(current_stage_y_training_data)

                        print('after add hidden node, new training case is in envelope')

        # close the recording file of training process
        training_process.close()

        # tf.train.SummaryWriter soon be deprecated, use following
        writer = tf.summary.FileWriter("C:/logfile", sess.graph)
        writer.close()

        # train end, get NN status
        curr_hidden_neuron_weight, curr_hidden_threshold, curr_output_neuron_weight, curr_output_threshold, curr_average_loss, curr_output = sess.run(
                    [hidden_weights, hidden_thresholds,
                     output_weights, output_threshold, average_square_residual,
                     output_layer],
                    {x_placeholder: x_training_data, y_placeholder: y_training_data})
        predict_y = sess.run([output_layer],
                                 {x_placeholder: x_training_data,
                                  y_placeholder: y_training_data})
        square_residuals = np.square(predict_y[0] - y_training_data.reshape((-1, 1)))
        # print(square_residuals)
        # concat residual & origin data, sort by residual, depart residual
        concat_y_and_x = np.concatenate((y_training_data.reshape((data_size, output_node_amount)), x_training_data), axis=1)
        concat_predict_and_y_x = np.concatenate((predict_y[0], concat_y_and_x), axis=1)
        concat_residual_and_predict_x_y = np.concatenate((square_residuals, concat_predict_and_y_x), axis=1)
        sort_result = concat_residual_and_predict_x_y[np.argsort(concat_residual_and_predict_x_y[:, 0])]

        np.savetxt(new_path + r"\hidden_neuron_weight.txt", curr_hidden_neuron_weight)
        np.savetxt(new_path + r"\hidden_threshold.txt", curr_hidden_threshold)
        np.savetxt(new_path + r"\output_neuron_weight.txt", curr_output_neuron_weight)
        np.savetxt(new_path + r"\output_threshold.txt", curr_output_threshold)
        np.savetxt(new_path + r"\training_data_residual_predict_output_desire_output_desire_input.txt", sort_result)

        file = open(new_path + r"\_training_detail.txt", 'w')
        file.writelines("learning_rate: " + str(learning_rate_eta) + "\n")
        file.writelines("input_node_amount: " + str(input_node_amount) + "\n")
        file.writelines("hidden_node_amount: " + str(hidden_node_amount) + "\n")
        file.writelines("output_node_amount: " + str(output_node_amount) + "\n")
        file.writelines("training_data_amount: " + str(data_size) + "\n")
        file.writelines("sigma: " + str(sigma) + "\n")
        file.writelines("sigma_multiplier: " + str(sigma_multiplier) + "\n")
        file.writelines("envelope_width_epsilon: " + str(epsilon) + "\n")
        file.writelines("outlier_rate: " + str(outlier_rate*100) + "%\n")
        file.writelines("average_loss_of_the_model: " + str(curr_average_loss) + "\n")
        file.writelines("thinking_times_count: " + str(thinking_times_count) + "\n")
        file.writelines("cramming_times_count: " + str(cramming_times_count) + "\n")
        file.writelines("softening_thinking_times_count: " + str(softening_thinking_times_count) + "\n")
        file.writelines("pruning_success_times_count: " + str(pruning_success_times_count) + "\n")
        file.writelines("total execution time: " + str(time.time() - execute_start_time) + " seconds" + "\n")
        file.close()
        print("thinking times: %s" % thinking_times_count)
        print("hidden node: %s nodes" % hidden_node_amount)
        print("--- execution time: %s seconds ---" % (time.time() - execute_start_time))
