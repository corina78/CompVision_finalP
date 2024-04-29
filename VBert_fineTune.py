import os
import re
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torch.optim as optim
from transformers import VisualBertForQuestionAnswering, BertTokenizerFast
from modeling_frcnn import GeneralizedRCNN
from utils import Config
from PIL import Image
import torchvision.transforms as transforms
import pickle
import json
import random

def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def load_dict_from_file(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

class VQADataset(Dataset):
    def __init__(self, image_dir, questions_dict, annotations, tokenizer, transform=None):
        self.image_dir = image_dir
        self.questions_dict = questions_dict
        self.annotations = annotations
        self.tokenizer = tokenizer
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        self.image_files = {self.process_image_id(os.path.splitext(file)[0]): file
                            for file in os.listdir(image_dir)
                            if file.lower().endswith('.png')}

    def process_image_id(self, filename):
        # Extracts numeric ID from filename and returns it as an integer
        return int(re.sub(r'^0+', '', filename.split('_')[-1]))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        processed_image_id = list(self.image_files.keys())[idx]
        image_filename = self.image_files[processed_image_id]
        image_path = os.path.join(self.image_dir, image_filename)
        #print(image_path)

        question = self.questions_dict.get(processed_image_id)
        if not question:
            raise ValueError(f"No question found for image_id {processed_image_id}")

        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        #inputs = self.tokenizer(question['question'], return_tensors="pt", padding="max_length", max_length=30,
                                truncation=False)

        inputs = self.tokenizer(question['question'], return_tensors="pt", padding="max_length", max_length=20, truncation=True)
        answer = 1 if question['answer_label'].lower() == 'yes' else 0

        return image, inputs['input_ids'].squeeze(0), torch.tensor(answer, dtype=torch.long)


def evaluate_model(model, dataloader, annotations, reverse_truth=False):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, inputs, answers in dataloader:
            outputs = model(
                input_ids=inputs,
                attention_mask=torch.ones_like(inputs),
                token_type_ids=torch.zeros_like(inputs),
                visual_embeds=images,
                return_dict=True
            )
            _, predicted = torch.max(outputs.logits, 1)
            total += answers.size(0)

            # convert annotations (yes/no) to binary values
            annotations_batch = [1 if ans.lower() == 'yes' else 0 for ans in annotations]

            # Reverse truth values if specified (in order to properly evaluate negative questions)
            if reverse_truth:
                annotations_batch = [1 - ans for ans in annotations_batch]

            correct += (predicted == torch.tensor(annotations_batch)).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy on Test Set: {accuracy}%')


if __name__ == "__main__":
    # load models and model components
    frcnn_cfg = Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")
    frcnn = GeneralizedRCNN.from_pretrained("unc-nlp/frcnn-vg-finetuned", config=frcnn_cfg)
    bert_tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    visualbert_vqa = VisualBertForQuestionAnswering.from_pretrained("uclanlp/visualbert-vqa")

    # load data (local)
    image_dir = '/home/corina/Documents/computer_vision/final_proj/scene_img_abstract_v002_train2017/'
    annotations = load_json('/home/corina/Documents/computer_vision/final_proj/abstract_v002_train2017_annotations.json')
    questions = load_dict_from_file('/home/corina/Documents/computer_vision/final_proj/questions_baseline.pkl')
    # load data google cloud
    # image_dir = '/home/corina_rios/source/computer-vision4773/bucket-4773/scene_img_abstract_v002_train2017/'
    # annotations = load_json('/home/corina_rios/source/computer-vision4773/bucket-4773/abstract_v002_train2017_annotations.json')
    # questions = load_dict_from_file('/home/corina_rios/source/computer-vision4773/bucket-4773/questions_baseline.pkl')
    # questions_negated = load_dict_from_file('/home/corina_rios/source/computer-vision4773/bucket-4773/questions_negated.pkl')

    # Split dataset into test and validation sets
    dataset = VQADataset(image_dir, questions, annotations, bert_tokenizer)
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(0.5 * dataset_size)  # 50-50 split
    random.shuffle(indices)
    train_indices, test_indices = indices[split:], indices[:split]

    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=False)

    for param in frcnn.parameters():
        param.requires_grad = False
    for name, param in frcnn.named_parameters():
        if 'layer4' in name:
            param.requires_grad = True
    for param in visualbert_vqa.parameters():
        param.requires_grad = True

    optimizer = optim.Adam([
        {'params': filter(lambda p: p.requires_grad, frcnn.parameters()), 'lr': 1e-4},
        {'params': visualbert_vqa.parameters(), 'lr': 1e-5}
    ])
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(20):
    #for epoch in range(10):
    #for epoch in range(3):
        # Training loop
        for images, inputs, answers in train_dataloader:
            optimizer.zero_grad()
            outputs = visualbert_vqa(
                input_ids=inputs,
                attention_mask=torch.ones_like(inputs),
                token_type_ids=torch.zeros_like(inputs),
                visual_embeds=images,
                return_dict=True
            )
            loss = loss_fn(outputs.logits, answers)
            loss.backward()
            optimizer.step()
            print(f'Training - Epoch {epoch}, Loss: {loss.item()}')

        # validation loop into training loop
        with torch.no_grad():
            val_loss = 0.0
            total = 0
            correct = 0
            for images, inputs, answers in test_dataloader:
                outputs = visualbert_vqa(
                    input_ids=inputs,
                    attention_mask=torch.ones_like(inputs),
                    token_type_ids=torch.zeros_like(inputs),
                    visual_embeds=images,
                    return_dict=True
                )
                loss = loss_fn(outputs.logits, answers)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.logits, 1)
                total += answers.size(0)
                correct += (predicted == answers).sum().item()
            accuracy = 100 * correct / total
            print(f'Validation - Epoch {epoch}, Loss: {val_loss}, Accuracy: {accuracy}%')

    # Evaluate against annotations on test set
    evaluate_model(visualbert_vqa, test_dataloader, annotations)
    #evaluate-model(visualbert_vqa, test_dataloader, annotations, reverse_truth=True)
