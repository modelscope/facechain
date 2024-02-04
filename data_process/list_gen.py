import os
import sys
import numpy as np
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--img_dir', type=str, help='List for single Face images')
    parser.add_argument('--list_num', type=int, default=8, help='The threshold for quality select')
    args = parser.parse_args() 
    return args

if __name__ == '__main__':
    args = parse_arguments()
    img_dir = args.img_dir
    list_num = args.list_num

    total_list = os.listdir(img_dir)
    list_len = int(len(total_list) / list_num)
    save_root = './filename_lists'

    if not os.path.exists(save_root):
        os.makedirs(save_root)


    for i in range(list_num):
        sub_list = total_list[i * list_len : (i + 1) * list_len]
        fileout = open(os.path.join(save_root, 'name_list_%s.txt')%(str(i)), 'w')
        for line in list(sub_list):
            fileout.write(os.path.join(img_dir.split('/')[-1], line) + '\n')
    print('Filename lists have been finished!')
    

        


