from PIL import Image
import torchvision.transforms as transforms
import torch
from transformers import BertTokenizer
import json

def load_and_resize_image_pytorch(image_path, new_size=(224, 224)):
    # Open the image file
    image = Image.open(image_path).convert('RGB')  # Ensure image is in RGB format

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize(new_size),  # Resize the image
        transforms.ToTensor(),        # Convert the image to a tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])

    # Apply transformations
    processed_image = transform(image)
    return processed_image

def prepare_text(text):

    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    return input_ids, attention_mask

def get_question_by_image_id(data, image_id):

    for item in data['questions']:
        if item['image_id'] == image_id:
            return item
    return None



if __name__=="__main__":

    #PyTorch for image processing
    image_path = '/home/corina/Documents/computer_vision/final_proj/scene_img_abstract_v002_train2017/'
    image = 'abstract_v002_train2015_000000000087.png'
    processed_image_pytorch = load_and_resize_image_pytorch(image_path+image)

    # Check the new size and display the image using PyTorch tensor format
    print(f"Resized image shape: {processed_image_pytorch.shape}")
    # import matplotlib.pyplot as plt
    #plt.imshow(processed_image_pytorch.permute(1, 2, 0))  Rearrange the channels for display
    #plt.show()

    #path to JSON file
    file_path = '/home/corina/Documents/computer_vision/final_proj/abstract_images_balanced_binary/OpenEnded_abstract_v002_train2017_questions.json'
    # Open the JSON file for reading
    with open(file_path, 'r') as file:
        # Load JSON data from the file into a P
        data_train_abstract = json.load(file)

    # Initialize the tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # get text from question to tokenize
    item = get_question_by_image_id(data_train_abstract, 87)
    text = item['question']
    input_ids, attention_mask = prepare_text(text)

    print("Input IDs:", input_ids)
    print("Attention Mask:", attention_mask)

    ## inference

    from vilbert.vilbert import VILBertForVLTasks

    # Path to pre-trained model
    model_path = 'pretrained_model.bin'

    # Load the model
    model = VILBertForVLTasks.from_pretrained(model_path)
    model.eval()  # Set the model to inference mode

    # Load the image and text
    input_ids = input_ids.unsqueeze(0)
    attention_mask = attention_mask.unsqueeze(0)
    image_tensor = processed_image_pytorch.unsqueeze(0)

    with torch.no_grad():  # Turn off gradients for inference
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, visual_embeds=image_tensor)

