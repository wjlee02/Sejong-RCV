import json
import os

def transform_format(input_data):
    transformed_data = {}
    for item in input_data:
        image_id = item['image_id']
        if image_id not in transformed_data:
            transformed_data[image_id] = {'bbox': [], 'category_id': [], 'is_crowd': []}

        transformed_data[image_id]['bbox'].append(item['bbox'])
        transformed_data[image_id]['category_id'].append(item['category_id'])
        transformed_data[image_id]['is_crowd'].append(item['is_crowd'])

    result = list(transformed_data.values())

    return result

dataset = []

outer_file_path = '/home/urp1/workspace/datasets/kaist/kaist_pd_urp/train-all-02.txt'
with open(outer_file_path) as outer_file:
    while True:
        line = outer_file.readline()
        if not line:
            break
        inner_file_path = os.path.join("/home/urp1/workspace/datasets/kaist/kaist_pd_urp", line.strip() + ".json")

        with open(inner_file_path) as inner_file:
            datasets = json.load(inner_file)
            # datasets : {'annotation': [{'bbox': [598, 223, 25, 64], 'category_id': 1, 'id': 0, 'image_id': 2761, 'is_crowd': 0}, 
            #                            {'bbox': [495, 219, 20, 43], 'category_id': 1, 'id': 0, 'image_id': 2761, 'is_crowd': 0}]}
            print(datasets)
            # for data in datasets:
            #     print(data)
                
            
            
            
            # annotations = data.get('annotation', [])
            # for annotation in annotations:
            
            
            #     shit.append(annotation)
# print(shit)                
# output = transform_format(shit)                    
# print(output)


