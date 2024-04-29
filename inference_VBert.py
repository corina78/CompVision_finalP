import os
import re
from IPython.display import Image, display
from IPython.display import Image, display
import PIL.Image
import io
import torch
import numpy as np
from processing_image import Preprocess
from visualizing_image import SingleImageViz
from modeling_frcnn import GeneralizedRCNN
from utils import Config
import utils
from transformers import VisualBertForQuestionAnswering, BertTokenizerFast
import pickle

def load_dict_from_file(file_path):
    with open(file_path, 'rb') as file:
        # Use pickle to load the serialized dictionary
        return pickle.load(file)

def find_question_for_image(image_id, question_dict):
    for item in question_dict:
        if item['image_id'] == image_id:
            return item['question']
    return None

# Path to the images and questions
image_dir = '/home/corina/Documents/computer_vision/final_proj/scene_img_abstract_v002_train2017/'
questions_file_path='/home/corina/Documents/computer_vision/final_proj/questions_baseline.pkl'
questions_dict = load_dict_from_file(questions_file_path)
OBJ_URL = "https://raw.githubusercontent.com/airsplay/py-bottom-up-attention/master/demo/data/genome/1600-400-20/objects_vocab.txt"
ATTR_URL = "https://raw.githubusercontent.com/airsplay/py-bottom-up-attention/master/demo/data/genome/1600-400-20/attributes_vocab.txt"
VQA_URL = "https://dl.fbaipublicfiles.com/pythia/data/answers_vqa.txt"


predictions = {}
# Process each image in the directory
for image_filename in os.listdir(image_dir):
    # Skip files that aren't images
    if not image_filename.lower().endswith('.png'):
        continue

    # Extract image_id from filename (e.g., '000000000008' -> 8)
    image_id = int(re.sub(r'^0+', '', image_filename.split('_')[-1].split('.')[0]))

    # Find the corresponding question for the image_id
    question = find_question_for_image(image_id, questions_dict)
    if not question:
        print(f"No question found for image_id {image_id}")
        continue

    # Construct the full image URL
    URL = os.path.join(image_dir, image_filename)


    # for visualizing output
    def showarray(a, fmt="jpeg"):
        a = np.uint8(np.clip(a, 0, 255))
        f = io.BytesIO()
        PIL.Image.fromarray(a).save(f, fmt)
        display(Image(data=f.getvalue()))


    # load object, attribute, and answer labels
    objids = utils.get_data(OBJ_URL)
    attrids = utils.get_data(ATTR_URL)
    vqa_answers = utils.get_data(VQA_URL)

    # load models and model components
    frcnn_cfg = Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")

    frcnn = GeneralizedRCNN.from_pretrained("unc-nlp/frcnn-vg-finetuned", config=frcnn_cfg)

    image_preprocess = Preprocess(frcnn_cfg)

    bert_tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    visualbert_vqa = VisualBertForQuestionAnswering.from_pretrained("uclanlp/visualbert-vqa")

    # image viz
    frcnn_visualizer = SingleImageViz(URL, id2obj=objids, id2attr=attrids)
    # run frcnn
    images, sizes, scales_yx = image_preprocess(URL)
    output_dict = frcnn(
        images,
        sizes,
        scales_yx=scales_yx,
        padding="max_detections",
        max_detections=frcnn_cfg.max_detections,
        return_tensors="pt",
    )
    # add boxes and labels to the image

    frcnn_visualizer.draw_boxes(
        output_dict.get("boxes"),
        output_dict.pop("obj_ids"),
        output_dict.pop("obj_probs"),
        output_dict.pop("attr_ids"),
        output_dict.pop("attr_probs"),
    )
    showarray(frcnn_visualizer._get_buffer())

    test_questions_for_url1 = [question]

    # test_questions_for_url1 = [
    # "Where is this scene?",
    # "what is the man riding?",
    # "What is the man wearing?",
    # "What is the color of the horse?"
    # ]
    # test_questions_for_url2 = [
    #    "Where is the cat?",
    #    "What is near the disk?",
    #    "What is the color of the table?",
    #    "What is the color of the cat?",
    #    "What is the shape of the monitor?",
    # ]

    # Very important that the boxes are normalized
    # normalized_boxes = output_dict.get("normalized_boxes")
    features = output_dict.get("roi_features")  # Very important that the boxes are normalized

    for test_question in test_questions_for_url1:
        test_question = [test_question]

        inputs = bert_tokenizer(
            test_question,
            padding="max_length",
            max_length=20,
            truncation=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        output_vqa = visualbert_vqa(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            visual_embeds=features,
            visual_attention_mask=torch.ones(features.shape[:-1]),
            token_type_ids=inputs.token_type_ids,
            output_attentions=False,
        )
        # get prediction
        pred_vqa = output_vqa["logits"].argmax(-1)
        print("Question:", test_question)
        print("prediction from VisualBert VQA:", vqa_answers[pred_vqa])
        predictions[image_id] = vqa_answers[pred_vqa]

# Save results to a file
with open('predictions.pkl', 'wb') as handle:
    pickle.dump(predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)
