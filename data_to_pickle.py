import os
import argparse
import pickle
import numpy as np
from PIL import Image as pil_image

def split_dataset_to_pickel(type):
    # 获取数据路径
    dataset_path = os.path.join(".\dataset", '%s.csv' %type)
    out_path = os.path.join(".\dataset", 'mini_imagenet_%s.txt' %type)
    with open(dataset_path,encoding = 'utf-8') as f:
        infos = np.loadtxt(f, str, delimiter = ",", skiprows = 1)
    #预定义数据结构 labels 为key 类下的图片为value
    data = dict()
    labels = set(infos[:,1])
    for label in labels:
        data[label] = []
    #存入图片数据   
    for info in infos:
        key = info[1]
        image_name = info[0]
        image_path = os.path.join(".\dataset", 'images',image_name)
        image_data = pil_image.open(image_path)
        image_data = image_data.resize((84, 84))
        # image_data = np.array(image_data, dtype='float32')
        # image_data = np.transpose(image_data, (2, 0, 1))
        data[key].append(image_data)
    #序列化
    item = pickle.dumps(data, protocol=True)
    with open(out_path, "wb") as f:
        f.write(item)
    return out_path

def main():
    parser = argparse.ArgumentParser(description='type')
    parser.add_argument('-t', '--type', default="train", help='type')
    args = parser.parse_args()
    print(args.type)
    result = split_dataset_to_pickel(args.type)
    print (result +'=============dump finish================')

if __name__ == '__main__':
    main()
