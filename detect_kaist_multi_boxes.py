from torchvision import transforms
from utils_kaist_2 import *
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model checkpoint
checkpoint = 'checkpoint_kaist_multi.pth.tar'
checkpoint = torch.load(checkpoint)
start_epoch = checkpoint['epoch'] + 1
print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
model = checkpoint['model']
model = model.to(device)
model.eval()

answer = []
annotated_image = []

# Transforms
resize = transforms.Resize((300, 300))
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

image_id = 0
true_path = '/home/urp1/workspace/official_Evaluation/kaist_annotations_test20.json'
with open(true_path, 'r') as f2:
    data = json.load(f2)

def detect(min_score, max_overlap, top_k, suppress=None):

    image = normalize(to_tensor(resize(original_image)))
    image = image.to(device)
    predicted_locs, predicted_scores = model(image.unsqueeze(0))
    det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, predicted_scores, min_score=min_score, max_overlap=max_overlap, top_k=top_k)
    
    true_boxes = list()
    for i in data['annotations']:
        if i['image_id'] == image_id:
            true_boxes.append(i['bbox'])
        elif i['image_id'] > image_id:
            break
    
    det_boxes = det_boxes[0].to('cpu')
    original_dims = torch.FloatTensor([original_image.width, original_image.height, original_image.width, original_image.height]).unsqueeze(0)
    det_boxes = det_boxes * original_dims
    det_labels = [rev_label_map[l] for l in det_labels[0].to('cpu').tolist()]
    if det_labels == ['background']:
        if true_boxes == []:
            return original_image

    annotated_image = original_image
    draw = ImageDraw.Draw(annotated_image)
    font = ImageFont.load_default()
                
    for j in range(0, len(det_labels[0])): # 한 이미지 내 박스 수
        if suppress is not None:
            if det_labels[i] in suppress:
                continue
        
        # Boxes
        box_location = det_boxes[i].tolist()
        draw.rectangle(xy=box_location, outline=label_color_map[det_labels[0]])
        draw.rectangle(xy=[l + 1. for l in box_location], outline=label_color_map[det_labels[0]])  # a second rectangle at an offset of 1 pixel to increase line thickness
        draw.rectangle(xy=[l + 2. for l in box_location], outline=label_color_map[det_labels[0]])  # a third rectangle at an offset of 1 pixel to increase line thickness
        draw.rectangle(xy=[l + 3. for l in box_location], outline=label_color_map[det_labels[0]])  # a fourth rectangle at an offset of 1 pixel to increase line thickness
        # Text
        text_size = font.getsize(det_labels[i].upper())
        text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
        textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0] + 4., box_location[1]]
        draw.rectangle(xy=textbox_location, fill=label_color_map[det_labels[i]])
        draw.text(xy=text_location, text='person', fill='white',font=font)
    
    for i in range(len(true_boxes)):
        true_location = true_boxes[i]
        true_location[2] = true_location[0] + true_location[2]
        true_location[3] = true_location[1] + true_location[3]
        draw.rectangle(xy=true_location, outline='#3CB44B')
    del draw
    return annotated_image
if __name__ == '__main__':
    path = '/home/urp1/workspace/datasets/kaist'
    save_path = '/home/urp1/workspace/thermal_image'
    filename = '/home/urp1/workspace/datasets/kaist/test-all-20.txt'
    f = open(filename, 'r')
    lines = f.readlines()
    for l in lines:
        #image 경로
        image_path = l[:-7]
        image_name = l[-7:-1]
        img_path = path + "/images/" + image_path + "lwir/" + image_name + ".jpg"
        original_image = Image.open(img_path, mode='r')
        original_image = original_image.convert('RGB')
        detect(original_image, min_score=0.2, max_overlap=0.5, top_k=200).save(save_path + str(image_id) + '.jpg',"JPEG")
        image_id += 1