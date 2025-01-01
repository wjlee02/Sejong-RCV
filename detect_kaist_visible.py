from torchvision import transforms
from utils_kaist_2 import *
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model checkpoint
checkpoint = 'checkpoint_kaist_2.pth.tar'
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
    with open('/home/urp1/workspace/kaist_dataset/TEST_visible_images.json') as j:
        test_imgs = json.load(j)
    
    # i가 0일 때    
    for i, img_path in enumerate(tqdm(test_imgs)):
        # '/home/urp1/workspace/datasets/kaist/images/set06/V000/visible/I00019.jpg'
        original_image = Image.open(img_path, mode='r')
        # '<PIL.PngImagePlugin.PngImageFile image mode=RGB size=640x512 at 0x7FC5F40A07D0>'
        original_image = original_image.convert('RGB') 
        # '<PIL.Image.Image image mode=RGB size=640x512 at 0x7FC5F4079A50>'
            
        # Transform
        image = normalize(to_tensor(resize(original_image))) # torch.Size([3, 300, 300])

        # Move to default device
        image = image.to(device)

        # Forward prop.
        predicted_locs, predicted_scores = model(image.unsqueeze(0)) # ([1, 8732, 4]), ([1, 8732, 21])

        # Detect objects in SSD output
        det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, predicted_scores, min_score=min_score, max_overlap=max_overlap, top_k=top_k)
        # [[0., 0., 1., 1.]], [0], [0.]

        det_boxes = det_boxes[0].to('cpu')
        original_dims = torch.LongTensor((original_image.width, original_image.height, original_image.width, original_image.height)).unsqueeze(0)
        # tensor([[640, 512, 640, 512]])
        det_boxes = det_boxes * original_dims # 예측 상자를 원본 이미지의 크기로 변환 
        # [[  0.,   0., 640., 512.]]
        
        # det_labels = [rev_label_map[l] for l in det_labels[0].to('cpu').tolist()]

        if det_labels == [0]:
            return original_image
            # <PIL.Image.Image image mode=RGB size=640x512 at 0x7FC5FFA7E710>     
                
        annotated_image = original_image
        draw = ImageDraw.Draw(annotated_image)
        font = ImageFont.load_default()
                
        for j in range(0, len(det_labels[0])): # 한 이미지 내 박스 수
            if suppress is not None:
                if det_labels[i] in suppress:
                    continue
            
            det_boxes[j][2] = det_boxes[j][2] - det_boxes[j][0]
            det_boxes[j][3] = det_boxes[j][3] - det_boxes[j][1]
                
            box_location = det_boxes[j].tolist()
            draw.rectangle(xy=box_location, outline=label_color_map[det_labels[j]])
            draw.rectangle(xy=[l + 1. for l in box_location], outline=label_color_map[det_labels[j]])
        
            
        annotated_image.save('/home/urp1/workspace/kaist/detected_images/visible/set06/V000/I00???.jpg')
        
        del draw        
        # return annotated_image
            
            # answer.append({"image_id" : i, "category_id" : det_labels[0][j].tolist(),
                        #    "bbox" : det_boxes[j].tolist(), "score" : det_scores[0][j].tolist()})
        
        # if i == 18:
            # break
            
    # with open('./output/visible_4.json', 'w') as f:
    #     json.dump(answer, f, indent=4)

if __name__ == '__main__':    
    annotated_image = detect(min_score=0.2, max_overlap=0.5, top_k=200)    

    # filename = '/home/urp1/workspace/datasets/kaist/kaist_pd_urp/test-all-20.txt'
    # f = open(filename, 'r')
    # lines = f.readlines()
    # for i in lines:
    #     i_parts = i.split('/')
    #     # annotated_image.save(os.path.join('/home/urp1/workspace/kaist/detected_images/visible', i_parts[0], i_parts[1], i_parts[2] + '.jpg'))
