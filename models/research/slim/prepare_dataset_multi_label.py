import os, cv2
import random,shutil



def get_list_file_in_folder(dir, ext='jpg'):
    included_extensions = [ext]
    file_names = [fn for fn in os.listdir(dir)
                  if any(fn.endswith(ext) for ext in included_extensions)]
    return file_names

def get_list_dir_in_folder(dir):
    sub_dir = [o for o in os.listdir(dir) if os.path.isdir(os.path.join(dir, o))]
    return sub_dir

def read_file(file_path):
    with open(file_path) as f:
        lineList = f.readlines()
    return lineList

def save_file(file_path, content):
    with open(file_path, 'w') as file:
        file.write(content)

def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
        print ('Finish make dir:',dir)


def create_dataset(num_img=355):
    count=0
    src_dir='/media/atsg/Data/datasets/face_recognition/lfw'
    dst_dir='/home/atsg/PycharmProjects/gvh205/Caffe_Tools_Ubuntu/data/Face_custom/original/1'
    list_dir=get_list_dir_in_folder(src_dir)
    for sub_dir in list_dir:
        list_img = get_list_file_in_folder(os.path.join(src_dir, sub_dir))
        for img in list_img:
            if (count > num_img):
                continue
            src_file = os.path.join(src_dir, sub_dir, img)
            dst_file = os.path.join(dst_dir, img)
            image = cv2.imread(src_file)
            image = cv2.resize(image, (60, 60))
            cv2.imwrite(dst_file, image)
            # shutil.copy(src_file,dst_file)
            count += 1

    count=0
    src_dir='/home/atsg/PycharmProjects/gvh205/arm_proj/to_customer/GVH205_ARM_project_training_environment/dataset/getty_dataset2_resize300/train/dirty'
    dst_dir='/home/atsg/PycharmProjects/gvh205/Caffe_Tools_Ubuntu/data/Face_custom/original/0'
    list_img = get_list_file_in_folder(os.path.join(src_dir))
    for img in list_img:
        src_file = os.path.join(src_dir, img)
        dst_file = os.path.join(dst_dir, img)
        image = cv2.imread(src_file)
        image = cv2.resize(image, (60, 60))
        cv2.imwrite(dst_file, image)
        # shutil.copy(src_file,dst_file)
        count += 1
        if(count>num_img):
            break

def create_multi_label_dataset(shuffle=True, train_ratio=0.8):
    peta_file_dir='/home/atsg/PycharmProjects/gvh205_py3/tensorflow_slim/pytorch/scripts/PETA'
    peta_train_file=os.path.join(peta_file_dir,'train_list_v2.txt')
    peta_val_file=os.path.join(peta_file_dir,'val_list_v2.txt')


    data_dir = '/home/atsg/PycharmProjects/gvh205_py3/tensorflow_slim/pytorch/data/PETA_tfrecord'
    label_map_path = os.path.join(data_dir,'label_map.txt')
    label_train_path = os.path.join(data_dir,'label_train.txt')
    label_test_path = os.path.join(data_dir,'label_test.txt')
    image_train_path = os.path.join(data_dir,'image_train.txt')
    image_test_path = os.path.join(data_dir,'image_test.txt')

    process_file(peta_train_file,image_train_path, label_train_path)
    process_file(peta_val_file,image_test_path, label_test_path)
    convert_2_jpg(image_train_path)
    convert_2_jpg(image_test_path)

    # create_dir(train_dir)
    # create_dir(val_dir)
    # create_dir(lmdb_dir)
    #
    # classes=get_list_dir_in_folder(img_dir) #get class
    #
    # train_txt=''
    # val_txt=''
    # for dir in classes:
    #     print (dir)
    #     create_dir(os.path.join(train_dir, dir))
    #     create_dir(os.path.join(val_dir, dir))
    #     list_file=get_list_file_in_folder(os.path.join(img_dir, dir))
    #     total_img=len(list_file)
    #     num_train_img=int(train_ratio*total_img)
    #     count=0
    #     if(shuffle):
    #         random.shuffle(list_file)
    #     for file in list_file:
    #         src_file=os.path.join(img_dir, dir,file)
    #         if(count<num_train_img):
    #             dst_file=os.path.join(train_dir, dir,file)
    #             train_txt+=os.path.join(dir,file)+' '+dir+'\n'
    #         else:
    #             dst_file=os.path.join(val_dir, dir,file)
    #             val_txt+=os.path.join(dir,file)+' '+dir+'\n'
    #         shutil.move(src_file,dst_file)
    #         count+=1
    #
    # save_file(os.path.join(dataset_dir,'train.txt'),train_txt)
    # save_file(os.path.join(dataset_dir,'val.txt'),val_txt)

def process_file(input_file_path, output_image_file, output_label_file, num_class=31):
    lf = open(input_file_path, 'r')
    images=''
    labels=''
    for line in lf:
        sep = line.rstrip('\n').split(',')
        img_path = sep[0]
        images+=img_path+'\n'
        ldata = sep[1:]
        ldata = list(map(int, ldata))
        label=''
        for i in range(num_class):
            if (ldata[i]==1):
                label+=str(i)+' '
        label+='\n'
        label.replace(' \n','\n')
        labels+=label

    save_file(output_image_file, images)
    save_file(output_label_file, labels)
    print('Finish')

def convert_2_jpg(image_file_path, delete=False):
    lf = open(image_file_path, 'r')
    new_files = ''

    from PIL import Image
    for line in lf:
        new_line=line
        if '.bmp' in line:
            img = Image.open(line.replace('\n',''))
            new_line=line.replace('.bmp','.jpg')
            img.save(new_line.replace('\n',''))
            if (delete):
                os.remove(line.replace('\n',''))
        new_files+=new_line

    save_file(image_file_path, new_files)

if __name__ == "__main__":

    #create_train_val_txt(img_dir,ano_dir)
    #create_train_val_detection()
    create_multi_label_dataset()
    #merge_train_val_classification()
    #create_dataset()
    print('Finish')