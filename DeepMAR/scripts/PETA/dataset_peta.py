import os
import random
random.seed(1)

valid_label_file = open("peta_valid_labels.txt", 'r')

label_list = []
for line in valid_label_file:
    label_list.append(line.rstrip('\n'))

data_train_list = []
data_val_list = []
for root_, dirs_, files_ in os.walk("/mnt/ssd01/openDatasets/PETAdataset"):

    if "Label.txt" in files_:
        lfile = open(os.path.join(root_,"Label.txt"),'r')
        for line in lfile:
            sep = line.rstrip('\n').split(" ")
            img_id = sep[0]
            img_ml = sep[1:]

            binary_label = []
            for vl in label_list:
                if vl in img_ml:
                    binary_label.append('1')
                else:
                    binary_label.append('0')

            img_names = [s for s in files_ if s.startswith(img_id)]


            for in_ in img_names:
                fpath = os.path.join(root_, in_)+','
                fpath += ','.join(binary_label)
            if random.random() < 0.05:
                for in_ in img_names:
                    fpath = os.path.join(root_, in_)+','
                    fpath += ','.join(binary_label)
                    data_val_list.append(fpath)
            else:
                for in_ in img_names:
                    fpath = os.path.join(root_, in_)+','
                    fpath += ','.join(binary_label)
                    data_train_list.append(fpath)

trainfile = open('train_list.txt','w')
valfile = open('val_list.txt','w')
for data in random.sample(data_train_list, len(data_train_list)):
    trainfile.write(data+'\n')

for data in data_val_list:
    valfile.write(data+'\n')

#print (data_train_list)
#print (data_val_list)
exit()



"""
bmp
jpg
png
jpeg
"""