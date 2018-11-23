import numpy as np
import os
counter = 1
s = 0
dir_path = os.path.dirname(os.path.realpath(__file__))
file = open(dir_path + r"\rule_convert_record.txt", 'w')
for i in range(1,20):
    a = np.loadtxt(r'C:\Users\user\PycharmProjects\autoencoder\resistant_learning\19_owl_rules\owl_rule_'+str(i)+'.txt',dtype=str)
    no_label = a[:,:-1].astype(float)
    total_amount = no_label.shape[0]
    unique_amount = np.unique(no_label, axis=0).shape[0]
    print('rule {0}: total-{1}, unique-{2}'.format(i,total_amount,unique_amount))
    if unique_amount >= 1000:
        np.savetxt(dir_path+r'\rule_{0}.txt'.format(counter), np.unique(no_label, axis=0), delimiter=' ')
        file.writelines("rule {0} convert to rule {1} \n".format(i, counter))
        s += unique_amount
        counter += 1
    else:
        print('捨棄rule {0}'.format(i))

print('共有 {0} 個sample'.format(s))
file.close()
