import json
import os
import time
import torch

path = '/home/urp1/workspace/datasets/kaist'
filename = '/home/urp1/workspace/datasets/kaist/kaist_pd_urp/test-all-20.txt'

#with open ('/home/urp1/workspace/out.json')

total = list()
image_total = list()
f = open(filename, 'r')
lines = f.readlines()

for l in lines:
    l = l[:-1]
    j = os.path.join(path, "kaist_pd_urp", l + '.json')
    with open (j, "r") as f2 :
        jp = json.load(f2)
    n_object = len(jp['annotation'])

    # tempdict = dict(bbox = list(), category_id = list(), is_crowd = list())
    # bbox = list()
    # category_id = list()        
    # is_crowd = list()


    # # object에 대한 정보
    # for obj in range(n_object):
    #     if jp['annotation'][obj]['category_id'] == -1:
    #         jp['annotation'][obj]['category_id'] = 0
            
    #     #bbox와 is_crowd는 모든 박스에 대해서 존재함    
    #     jp['annotation'][obj]['bbox'][2] = jp['annotation'][obj]['bbox'][0] + jp['annotation'][obj]['bbox'][2]
    #     jp['annotation'][obj]['bbox'][3] = jp['annotation'][obj]['bbox'][1] + jp['annotation'][obj]['bbox'][3]
        
    #     bbox.append(jp['annotation'][obj]['bbox'])          
    #     is_crowd.append(jp['annotation'][obj]['is_crowd'])          
    #     category_id.append(jp['annotation'][obj]['category_id'])          
            

    # 탐지한 박스가 있을 경우만 담음
    # if jp['annotation'][obj]['bbox'] != [0, 0, 0, 0]:
        
    # tempdict['bbox'] = bbox
    # tempdict['category_id'] = category_id
    # tempdict['is_crowd'] = is_crowd
    # total.append(tempdict)
    
    
    #image 경로
    image_path = l[:-7]
    image_name = l[-7:]
    image_total.append(path + "/images/" + image_path + "/visible" + image_name + ".jpg")

    #total['category_id'].append(jp['annotation'][obj]['category_id'])
    #total['is_crowd'].append(jp['annotation'][obj]['is_crowd'])
    
output_folder = '/home/urp1/workspace/kaist_dataset'

with open(os.path.join(output_folder, 'TEST_visible_images.json'), 'w') as j:
    json.dump(image_total, j)
# with open(os.path.join(output_folder, 'TRAIN_lwir_objects.json'), 'w') as j:
#     json.dump(total, j)
    
# import pdb; pdb.set_trace()
