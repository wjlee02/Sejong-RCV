import json
import os
import torch

path = '/home/urp1/workspace/datasets/kaist'
filename = '/home/urp1/workspace/datasets/kaist/kaist_pd_urp/test-all-20.txt'

# total = list()
image_total = list()
f = open(filename, 'r')
lines = f.readlines()

for l in lines:
    l = l[:-1]
    image_path = l[:-7]
    image_name = l[-7:]
    image_total.append(path + "/images/" + image_path + "/lwir" + image_name + ".jpg")

output_folder = '/home/urp1/workspace/kaist_dataset'

with open(os.path.join(output_folder, 'TEST_lwir_images.json'), 'w') as j:
    json.dump(image_total, j)
# with open(os.path.join(output_folder, 'TRAIN_lwir_objects.json'), 'w') as j:
#     json.dump(total, j)