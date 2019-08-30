import os, cv2

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


def create_multi_label_dataset(shuffle=True, train_ratio=0.8):
    peta_file_dir='/home/duycuong/PycharmProjects/research_py3/tensorflow_slim/data/PETA'
    peta_train_file=os.path.join(peta_file_dir,'train_list_v2.txt')
    peta_val_file=os.path.join(peta_file_dir,'val_list_v2.txt')


    data_dir = '/home/duycuong/PycharmProjects/research_py3/tensorflow_slim/data/PETA/tfrecord'
    label_train_path = os.path.join(data_dir,'label_train.txt')
    label_test_path = os.path.join(data_dir,'label_test.txt')
    image_train_path = os.path.join(data_dir,'image_train.txt')
    image_test_path = os.path.join(data_dir,'image_test.txt')

    process_file(peta_train_file,image_train_path, label_train_path)
    process_file(peta_val_file,image_test_path, label_test_path)
    convert_2_jpg(image_train_path, delete=False)
    convert_2_jpg(image_test_path, delete=False)

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
    print('Finish process file',input_file_path)

def convert_2_jpg(image_file_path, delete=False):
    lf = open(image_file_path, 'r')
    new_files = ''

    from PIL import Image
    for line in lf:
        new_line=line
        if '.bmp' in line:
            print ('convert bmp file',line,'to jpg')
            img = Image.open(line.replace('\n',''))
            new_line=line.replace('.bmp','.jpg')
            img.save(new_line.replace('\n',''))
            cv2.waitKey(10)
            if (delete):
                os.remove(line.replace('\n',''))
        new_files+=new_line

    save_file(image_file_path, new_files)
    print('Finish convert all images in',image_file_path,' to jpeg format!')

if __name__ == "__main__":
    create_multi_label_dataset()
    print('Finish create_multi_label_dataset')