from torchvision import transforms
from utils_kaist_lwir import *
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model checkpoint
checkpoint = 'checkpoint_kaist_3.pth.tar'
checkpoint = torch.load(checkpoint)
start_epoch = checkpoint['epoch'] + 1
print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
model = checkpoint['model']
model = model.to(device)
model.eval()

answer = []

# Transforms
resize = transforms.Resize((300, 300))
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def detect(min_score, max_overlap, top_k, suppress=None):
    """
    Detect objects in an image with a trained SSD300, and visualize the results.

    :param original_image: image, a "PIL Image"
    :param min_score: minimum threshold for a detected box to be considered a match for a certain class
    :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via Non-Maximum Suppression (NMS)
    :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
    :param suppress: classes that you know for sure cannot be in the image or you do not want in the image, a list
    :return: annotated image, a "PIL Image"
    """
    with open('/home/urp1/workspace/kaist_dataset/TEST_lwir_images.json') as j:
        test_imgs = json.load(j)
    
    for i, img_path in enumerate(tqdm(test_imgs)):
        original_image = Image.open(img_path, mode='r')
        original_image = original_image.convert('RGB') 
 
        # Transform
        image = normalize(to_tensor(resize(original_image))) # torch.Size([3, 300, 300])

        # Move to default device
        image = image.to(device)

        # Forward prop.
        predicted_locs, predicted_scores = model(image.unsqueeze(0))

        # Detect objects in SSD output
        det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, predicted_scores, min_score=min_score, max_overlap=max_overlap, top_k=top_k)
        # bbox, image_id, score + (category_id)
        # 순서는 image_id, category_id, bbox, score
        
        if len(det_labels[0]) > 1 :
            import pdb;pdb.set_trace()
  
        det_boxes = det_boxes[0].to('cpu')
        original_dims = torch.LongTensor((original_image.width, original_image.height, original_image.width, original_image.height)).unsqueeze(0)
        det_boxes = det_boxes * original_dims # 예측 상자를 원본 이미지의 크기로 변환
        # [xmin, ymin, xmax, ymax] 형식
        # 원래 모양으로 바꿔야함
        # [x, y, w, h] 형식
        
        # if(i==18):
        #     import pdb;pdb.set_trace()
        
        for j in range(0, len(det_labels[0])): # 한 이미지 내 박스 수
            
            det_boxes[j][2] = det_boxes[j][2] - det_boxes[j][0]
            det_boxes[j][3] = det_boxes[j][3] - det_boxes[j][1]
            
            answer.append({"image_id" : i, "category_id" : det_labels[0][j].tolist(),
                           "bbox" : det_boxes[j].tolist(), "score" : det_scores[0][j].tolist()})
                        
    with open('./output/Thermal_1.json', 'w') as f:
        json.dump(answer, f, indent=4)
#----------------------------------------------------------------------------------------------------------------------------------
    # Move detections to the CPU
    det_boxes = det_boxes[0].to('cpu')

    # Transform to original image dimensions
    original_dims = torch.FloatTensor(
        [original_image.width, original_image.height, original_image.width, original_image.height]).unsqueeze(0)
    det_boxes = det_boxes * original_dims # 예측 상자를 원본 이미지의 크기로 변환, [xmin, ymin, xmax, ymax] 형식

    # Decode class integer labels
    det_labels = [rev_label_map[l] for l in det_labels[0].to('cpu').tolist()]

    # If no objects found, the detected labels will be set to ['0.'], i.e. ['background'] in SSD300.detect_objects() in model.py
    if det_labels == ['background']:
        # Just return original image
        return original_image

    # Annotate
    annotated_image = original_image
    draw = ImageDraw.Draw(annotated_image)
    font = ImageFont.load_default()

    # Suppress specific classes, if needed
    for i in range(det_boxes.size(0)):
        if suppress is not None:
            if det_labels[i] in suppress:
                continue

        # Boxes
        box_location = det_boxes[i].tolist()
        draw.rectangle(xy=box_location, outline=label_color_map[det_labels[i]])
        draw.rectangle(xy=[l + 1. for l in box_location], outline=label_color_map[det_labels[i]])  # a second rectangle at an offset of 1 pixel to increase line thickness
        # draw.rectangle(xy=[l + 2. for l in box_location], outline=label_color_map[
        #     det_labels[i]])  # a third rectangle at an offset of 1 pixel to increase line thickness
        # draw.rectangle(xy=[l + 3. for l in box_location], outline=label_color_map[
        #     det_labels[i]])  # a fourth rectangle at an offset of 1 pixel to increase line thickness

        # Text
        text_size = font.getsize(det_labels[i].upper())
        text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
        textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0] + 4., box_location[1]]
        draw.rectangle(xy=textbox_location, fill=label_color_map[det_labels[i]])
        draw.text(xy=text_location, text=det_labels[i].upper(), fill='white', font=font)
    del draw

    return annotated_image


if __name__ == '__main__':
    # for i in range(0, 2252): # 흠.... image를 2000개 정도 불러와야하는데 img_path를 어떻게 넣어주지
    
    annotated_image = detect(min_score=0.2, max_overlap=0.5, top_k=200) # .show()는 터미널에서 안돌아감
    # annotated_image.save('/home/urp1/workspace/kaist/detected_images')   
    # /home/urp1/workspace/a-PyTorch-Tutorial-to-Object-Detection/annotated_image/00001.jpg