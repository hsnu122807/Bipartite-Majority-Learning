import numpy as np
import random

# a = np.array([1,2,3])
# b = np.array([1,2,3])
# c = a == b
# if all(c) is True:
#     print('123')
# print(a == b)

benign_samples = np.loadtxt(r"C:\Users\user\PycharmProjects\autoencoder\resistant_learning\19_owl_rules\benign_chrome_filezilla_sample.txt", dtype=float, delimiter=' ')
print(benign_samples.shape)
benign_samples = np.unique(benign_samples, axis=0)
print(benign_samples.shape)

data_dir_name = r"C:\Users\user\PycharmProjects\autoencoder\resistant_learning\19_owl_rules\owl_benign_samples"

file = open(data_dir_name + ".txt")
testing_data_benign_predict_wrong_count = 0
testing_data_benign_count = 0
while 1:
    r = random.randint(1, 10)
    for i in range(r):
        line = file.readline()
        if not line:
            break
    if not line:
        break
    data_arr = line.split(' ')
    data_arr = data_arr[:-1]
    # print(data_arr)
    # print(type(data_arr))
    data_arr = np.array(data_arr, dtype=float).reshape(1, -1)
    # print(type(data_arr))
    # print(data_arr.shape)
    benign_samples = np.concatenate([benign_samples, data_arr], axis=0)
    # print(benign_samples.shape)
    # input(1)
file.close()

np.savetxt('temp.txt', benign_samples, delimiter=' ')
print(benign_samples.shape)
benign_samples = np.unique(benign_samples, axis=0)
print(benign_samples.shape)
benign_samples = benign_samples[:19391]
print(benign_samples.shape)
np.savetxt('owl_benign_samples.txt', benign_samples, delimiter=' ')
