# coding=utf-8
'''
construct the json format examples for BERT
'''
import pandas as pd
import os
import json
# import math

label_dict = {'车站码头': 0,
             '街巷': 1,
             '居民小区': 2,
             '停车场': 3}


def file_based_dataset_prepare(xls_dir, save_path='./dataset', mode='train'):
    '''
    use this fn to construct json format example
    args:
        xls_dir: The excel file path
        mode: either var in range of 'train', 'validate' and 'test'
    '''
    mode = str(mode)
    if mode.lower() not in ('train', 'dev', 'test'):
        raise TypeError('{} is not in range of train, validate, test'.format(mode))
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    save_file = os.path.join(save_path, mode + '.json')
    index = 0
    f = open(save_file, 'w')
    sheet = pd.read_excel(xls_dir)    
    data = sheet.values
    for sub_data in data:
        for i in range(3):
            sub_data[i] = sub_data[i].strip('。')
        sentence = ';'.join(sub_data[:3])
        if mode.lower() in ('train', 'dev'):
            label_des = sub_data[3]
            assert label_des in label_dict.keys(), '{} is not in the labels description'.format(label_des)
            label = label_dict[label_des]
            json_data = {"label": str(label), "label_des": label_des, "sentence": sentence}
            json_str = json.dumps(json_data, ensure_ascii=False)
            print(json_str)
            f.write(json_str)
            f.write('\n')
        elif mode.lower() == 'test':
            json_data = {"id": index, "sentence": sentence} 
            json_str = json.dumps(json_data, ensure_ascii=False)
            print(json_str)
            f.write(json_str)
            f.write('\n')
            index += 1           
    f.close()


def local_based_dataset_prepare(datas, save_path='./dataset', mode='train'):
    mode = str(mode)
    if mode.lower() not in ('train', 'dev', 'test'):
        raise TypeError('{} is not in range of train, validate, test'.format(mode))
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    save_file = os.path.join(save_path, mode + '.json')
    index = 0
    f = open(save_file, 'w')
    dataset_list = []
    for sub_data in datas:
        for i in range(3):
            sub_data[i] = sub_data[i].strip('。')
        sentence = ';'.join(sub_data[:3])
        if mode.lower() in ('train', 'dev'):
            label_des = sub_data[3]
            assert label_des in label_dict.keys(), '{} is not in the labels description'.format(label_des)
            label = label_dict[label_des]
            json_data = {"label": str(label), "label_des": label_des, "sentence": sentence}
            json_str = json.dumps(json_data, ensure_ascii=False)
            dataset_list.append(json_str)
            print(json_str)
            f.write(json_str)
            f.write('\n')
        elif mode.lower() == 'test':
            json_data = {"id": index, "sentence": sentence} 
            json_str = json.dumps(json_data, ensure_ascii=False)
            dataset_list.append(json_str)
            print(json_str)
            f.write(json_str)
            f.write('\n')
            index += 1           
    f.close()
    return dataset_list


def Save_Label(label_dict, save_path):
    f = open(os.path.join(save_path, 'labels.json'), 'w')
    for key, value in label_dict.items():
        label_content = {"label": str(value), "label_des": key}
        json_str = json.dumps(label_content, ensure_ascii=False)
        f.write(json_str)
        f.write('\n')
    f.close()


if __name__ == '__main__':
    xls_dir = '/home/wanghongwei/WorkSpace/datasets/NLP/Glue-self/bike.xls'
    save_path = '/home/wanghongwei/WorkSpace/datasets/NLP/Glue-self/datasets/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    # file_based_dataset_prepare(xls_dir, save_path, mode='test')
    # file_based_dataset_prepare(xls_dir, save_path, mode='dev')
    # file_based_dataset_prepare(xls_dir, save_path, mode='train')
    Save_Label(label_dict, save_path)
