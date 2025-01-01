from torchvision import transforms
from utils_kaist_lwir import *
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model checkpoint
checkpoint = 'checkpoint_kaist_3.pth'
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

    with open('/home/urp1/workspace/kaist_dataset/TEST_visible_images.json') as k:
        test_imgs_rgb = json.load(k)
    
    with open('/home/urp1/workspace/kaist_dataset/TEST_lwir_images.json') as j:
        test_imgs_thermal = json.load(j)
    
    for i, (img_path_rgb, img_path_thermal) in enumerate(tqdm(zip(test_imgs_rgb, test_imgs_thermal), total=len(test_imgs_rgb))):
        
        original_image_rgb = Image.open(img_path_rgb, mode='r')
        original_image_rgb = original_image_rgb.convert('RGB')
        original_image_thermal = Image.open(img_path_thermal, mode='r')
        original_image_thermal = original_image_thermal.convert('RGB')
 
        # Transform
        image_rgb = normalize(to_tensor(resize(original_image_rgb)))
        image_thermal = normalize(to_tensor(resize(original_image_thermal)))
        
        # Move to default device
        image_rgb = image_rgb.to(device)
        image_thermal = image_thermal.to(device)
        
        # Forward prop.
        predicted_locs, predicted_scores = model(image_rgb.unsqueeze(0), image_thermal.unsqueeze(0))
        # Detect objects in SSD output
        det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, predicted_scores, min_score=min_score, max_overlap=max_overlap, top_k=top_k)

        det_boxes = det_boxes[0].to('cpu')
        original_dims = torch.LongTensor((original_image_thermal.width, original_image_thermal.height, original_image_thermal.width, original_image_thermal.height)).unsqueeze(0)
        det_boxes = det_boxes * original_dims 

        for j in range(0, len(det_labels[0])): # 한 이미지 내 박스 수
            
            det_boxes[j][2] = det_boxes[j][2] - det_boxes[j][0]
            det_boxes[j][3] = det_boxes[j][3] - det_boxes[j][1]
            
            answer.append({"image_id" : i, "category_id" : det_labels[0][j].tolist(),
                           "bbox" : det_boxes[j].tolist(), "score" : det_scores[0][j].tolist()})
                        
    with open('./output/multispectral_halfway_20.json', 'w') as f:
        json.dump(answer, f, indent=4)
#----------------------------------------------------------------------------------------------------------------------------------
    # 제출용 파일

if __name__ == '__main__':    
    annotated_image = detect(min_score=0.2, max_overlap=0.5, top_k=200)