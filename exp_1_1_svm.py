# -*- coding: utf-8 -*-
import numpy as np
import time
import os
from sklearn.svm import SVC
import pickle

dir_path = os.path.dirname(os.path.realpath(__file__))
kernel_arr = ['linear', 'rbf', 'poly', 'sigmoid']

for owl in range(1, 10):
    owl_rule = str(owl)
    for kernel in kernel_arr:
        new_path = r"{0}\exp_1_1".format(dir_path)
        if not os.path.exists(new_path):
            os.makedirs(new_path)
        new_path = r"{0}\svm_{1}".format(new_path, kernel)
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
        y_training_data = training_data[:, 0]
        # print(x_training_data.shape)
        # print(y_training_data.shape)

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
        y_testing_data_mal_part = np.tile(mal_y, x_testing_data_mal_part.shape[0])
        y_testing_data_benign_part = np.tile(benign_y, x_testing_data_benign_part.shape[0])
        # print(x_testing_data_mal_part.shape)
        # print(x_testing_data_benign_part.shape)
        # print(y_testing_data_mal_part.shape)
        # print(y_testing_data_benign_part.shape)
        # input(1)

        # parameters
        every_stage_max_thinking_times = 10000
        m = x_training_data.shape[1]
        data_size = training_data.shape[0]
        outlier_rate = 0.05
        # learning_rate = 0.01

        # create file to save training process
        training_process_log = open(new_path + r"\_two_class_training_process.txt", 'w')

        # counter
        bp_times_count = 0

        # classifier
        clf = SVC(kernel=kernel)

        start_time = time.time()
        for stage in range(m+2, int(data_size * (1 - outlier_rate) + 1)):
            print('-----stage: ' + str(stage) + '-----')
            training_process_log.writelines('-----stage: ' + str(stage) + '-----' + "\n")

            if stage == (m+2):
                current_stage_training_x = x_training_data[:m+2]
                current_stage_training_y = y_training_data[:m+2]
                # print(current_stage_training_x.shape)
                # print(current_stage_training_y.shape)
                clf.fit(current_stage_training_x, current_stage_training_y)
            else:  # 用order term sorting
                predict_y_of_all_data = clf.decision_function(x_training_data).reshape(-1, 1)
                order_term_of_all_data = -predict_y_of_all_data * y_training_data.reshape(-1, 1)

                concat_x_and_y = np.concatenate((x_training_data, y_training_data.reshape(-1, 1)), axis=1)
                concat_order_and_x_y = np.concatenate((order_term_of_all_data, concat_x_and_y), axis=1)
                # print(concat_entropy_and_x_y.shape)
                sort_result = concat_order_and_x_y[np.argsort(concat_order_and_x_y[:, 0])]
                x_training_data_sort_by_entropy = np.delete(sort_result, (0, m + 1), axis=1)  # 去除0和m+1欄
                y_training_data_sort_by_entropy = np.delete(sort_result, slice(0, m + 1), axis=1)  # 去除從0到m欄
                current_stage_training_x = x_training_data_sort_by_entropy[:stage]
                current_stage_training_y = y_training_data_sort_by_entropy[:stage].reshape(-1)

            current_stage_predict_class = clf.predict(current_stage_training_x)
            if all(current_stage_predict_class == current_stage_training_y) is True:
                print('all data in this stage correctly classified, do not need training.')
                training_process_log.writelines('all data in this stage correctly classified, do not need training.' + "\n")
            else:
                clf.fit(current_stage_training_x, current_stage_training_y)
                current_stage_predict_class = clf.predict(current_stage_training_x)
                if all(current_stage_predict_class == current_stage_training_y) is True:
                    print('after training, all data in this stage correctly classified.')
                    training_process_log.writelines('after training, all data in this stage correctly classified.' + "\n")
                else:
                    print('train failed, after tuning, all data in this stage cannot be correctly classified.'.format(every_stage_max_thinking_times))
                    training_process_log.writelines('train failed, after {0} tuning, all data in this stage cannot be correctly classified.\n'.format(every_stage_max_thinking_times))

        training_process_log.close()
        end_time = time.time()
        print('train end, save networks')
        with open(new_path + r'\svm_model.pickle', 'wb') as handle:
            pickle.dump(clf, handle, protocol=pickle.HIGHEST_PROTOCOL)

        predict_y_of_all_data = clf.decision_function(x_training_data).reshape(-1, 1)
        order_term_of_all_data = -predict_y_of_all_data * y_training_data.reshape(-1, 1)

        concat_x_and_y = np.concatenate((x_training_data, y_training_data.reshape(-1, 1)), axis=1)
        concat_order_and_x_y = np.concatenate((order_term_of_all_data, concat_x_and_y), axis=1)
        sort_result = concat_order_and_x_y[np.argsort(concat_order_and_x_y[:, 0])]
        np.savetxt(new_path + r"\training_data_order_x_y.txt", sort_result)

        file = open(new_path + r"\_two_class_training_detail.txt", 'w')
        file.writelines('input dimension: {0}\n'.format(m))
        file.writelines('training data amount: {0}\n'.format(data_size))
        file.writelines('outlier rate: {0}\n'.format(outlier_rate))
        file.writelines('thinking times count: {0}\n'.format(bp_times_count))
        file.writelines('execute time: {0} sec\n'.format(end_time - start_time))
        file.close()

        file = open(new_path + r"\_training_analyze.txt", 'w')
        predict_class_of_benign_train_data = clf.predict(x_training_data_benign_part)
        mask = (predict_class_of_benign_train_data == y_training_data_benign_part)
        benign_train_correct_count = predict_class_of_benign_train_data[mask].shape[0]
        file.writelines('benign accuracy: {0}/{1} , {2}\n'.format(benign_train_correct_count, x_training_data_benign_part.shape[0], (benign_train_correct_count / x_training_data_benign_part.shape[0])))
        predict_class_of_mal_train_data = clf.predict(x_training_data_mal_part)
        mask = (predict_class_of_mal_train_data == y_training_data_mal_part)
        mal_train_correct_count = predict_class_of_mal_train_data[mask].shape[0]
        file.writelines('mal accuracy: {0}/{1} , {2}\n'.format(mal_train_correct_count, x_training_data_mal_part.shape[0], (mal_train_correct_count / x_training_data_mal_part.shape[0])))
        file.close()

        file = open(new_path + r"\_testing_analyze.txt", 'w')
        predict_class_of_benign_test_data = clf.predict(x_testing_data_benign_part)
        mask = (predict_class_of_benign_test_data == y_testing_data_benign_part)
        benign_test_correct_count = predict_class_of_benign_test_data[mask].shape[0]
        file.writelines(
            'benign accuracy: {0}/{1} , {2}\n'.format(benign_test_correct_count, x_testing_data_benign_part.shape[0],
                                                      (benign_test_correct_count / x_testing_data_benign_part.shape[
                                                          0])))
        predict_class_of_mal_test_data = clf.predict(x_testing_data_mal_part)
        mask = (predict_class_of_mal_test_data == y_testing_data_mal_part)
        mal_test_correct_count = predict_class_of_mal_test_data[mask].shape[0]
        file.writelines(
            'mal accuracy: {0}/{1} , {2}\n'.format(mal_test_correct_count, x_testing_data_mal_part.shape[0],
                                                   (mal_test_correct_count / x_testing_data_mal_part.shape[0])))
        file.close()
