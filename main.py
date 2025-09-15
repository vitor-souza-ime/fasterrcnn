import requests
from io import BytesIO
import torch
import torchvision
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

# Lista de classes do COCO
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
    'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
    'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# Função para carregar imagem da URL
def load_image_from_url(url):
    response = requests.get(url)
    image = Image.open(BytesIO(response.content)).convert("RGB")
    transform = transforms.Compose([transforms.ToTensor()])
    return transform(image).unsqueeze(0), image

# Função de detecção
def detect_objects(model, image_tensor):
    model.eval()
    with torch.no_grad():
        prediction = model(image_tensor)
    return prediction

# Função para desenhar caixas e rótulos com probabilidade
def draw_boxes(image, prediction, threshold=0.5):
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
        
    labels = prediction[0]['labels']
    boxes = prediction[0]['boxes']
    scores = prediction[0]['scores']

    for i in range(len(boxes)):
        if scores[i] > threshold:
            box = boxes[i].tolist()
            label = COCO_INSTANCE_CATEGORY_NAMES[labels[i]]
            score = scores[i].item()
            
            # Desenha a caixa
            draw.rectangle(box, outline="red", width=3)
            
            # Desenha o rótulo com a probabilidade acima da caixa
            text = f"{label} {score:.2f}"
            # Ajusta posição do texto
            text_bbox = draw.textbbox((0,0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            text_location = (box[0], max(0, box[1] - text_height))
            draw.rectangle([text_location, (text_location[0]+text_width, text_location[1]+text_height)], fill="red")
            draw.text(text_location, text, fill="white", font=font)
    
    return image

# Carregar modelo pré-treinado
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

# URL da imagem
image_url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSb0NDL2JJTH4vbJlGgtlrgXbffGTiNsAt_rQ&s"

# Carregar e processar a imagem
image_tensor, image = load_image_from_url(image_url)

# Realizar detecção
prediction = detect_objects(model, image_tensor)

# Desenhar caixas e rótulos com probabilidade
image_with_boxes = draw_boxes(image, prediction, threshold=0.5)

# Exibir imagem
plt.figure(figsize=(12,8))
plt.imshow(image_with_boxes)
plt.axis('off')
plt.show()
