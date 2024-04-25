import argparse
import torch

import sys
sys.path.append('/home/wangpuyi/Q-Align')
from q_align.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from q_align.conversation import conv_templates, SeparatorStyle
from q_align.model.builder import load_pretrained_model
from q_align.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer

import json
from tqdm import tqdm
from collections import defaultdict

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"



def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def main(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)
    
    
    import json
    import pandas as pd

    json_path = os.path.join(args.json_path, args.json_name)
    if not os.path.exists(json_path):
        data = pd.read_csv(args.csv_path)
        data.to_json(json_path, orient='records')
                  
    img_root = args.img_root

    jsons = [
        json_path,
    ]
    # json_prefix = "playground/data/test_jsons/"
    # jsons = [
    #     json_prefix + "test_.json",
    # ]

    os.makedirs(f"results/{args.model_path}/", exist_ok=True)
    os.makedirs(args.tensor_root, exist_ok=True)


    conv_mode = "mplug_owl2"
    question_type = args.question_type
    
    if question_type == 1:
        inp = "How would you rate the quality of this image?"
            
        conv = conv_templates[conv_mode].copy()
        inp =  inp + "\n" + DEFAULT_IMAGE_TOKEN
        conv.append_message(conv.roles[0], inp)
        image = None
            
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt() + " The quality of the image is"
        
        toks = ["good", "poor", "high", "fair", "low", "excellent", "bad", "fine", "moderate",  "decent", "average", "medium", "acceptable"]
        print(toks)
        ids_ = [id_[1] for id_ in tokenizer(toks)["input_ids"]]
        print(ids_)
    elif question_type == 2: #common sense / coherence
        inp = "Evaluate if the image quality is compromised due to violations of coherence."
        conv = conv_templates[conv_mode].copy()
        inp =  inp + "\n" + DEFAULT_IMAGE_TOKEN
        conv.append_message(conv.roles[0], inp)

        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        # toks = ['extreme', 'profound', 'significant', 'considerable', 'substantial', 'moderate', 'noticeable', 'mild', 'slight', 'bare', 'negligible', 'high', 'low']
        # ids_ = [id_[1] for id_ in tokenizer(toks)["input_ids"]]
        # print(toks)
        # print(ids_)
    elif question_type == 3: # semantic content
        inp = 'Evaluate the input image to determine if its quality is compromised due to a lack of meaningful semantic content.'
        conv = conv_templates[conv_mode].copy()
        inp =  inp + "\n" + DEFAULT_IMAGE_TOKEN
        conv.append_message(conv.roles[0], inp)

        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        # toks = ['extreme', 'profound', 'significant', 'considerable', 'substantial', 'moderate', 'noticeable', 'mild', 'slight', 'bare', 'negligible', 'high', 'low']
        # ids_ = [id_[1] for id_ in tokenizer(toks)["input_ids"]]
        # print(toks)
        # print(ids_)

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(args.device)
    
    for json_ in jsons:
        with open(json_) as f:
            iqadata = json.load(f)  

            image_tensors = []
            batch_data = []
            if question_type == 1:
                json_ = json_.replace("combined/", "combined-")
                with open(f"results/{args.model_path}/{json_.split('/')[-1]}", "w") as wf:
                    wf.write("[\n")
            for i, llddata in enumerate(tqdm(iqadata, desc="Evaluating [{}]".format(json_.split("/")[-1]))):
                try:
                    filename = llddata["name"]
                except:
                    filename = llddata["img_path"]
                llddata["logits"] = defaultdict(float) # 添加一个 "logits" 键 原本存在name, prompt, mos
                imag_path = os.path.join(img_root, filename)
                image = load_image(imag_path) # load image here
                def expand2square(pil_img, background_color):
                        width, height = pil_img.size
                        if width == height:
                            return pil_img
                        elif width > height:
                            result = Image.new(pil_img.mode, (width, width), background_color)
                            result.paste(pil_img, (0, (width - height) // 2))
                            return result
                        else:
                            result = Image.new(pil_img.mode, (height, height), background_color)
                            result.paste(pil_img, ((height - width) // 2, 0))
                            return result
                image = expand2square(image, tuple(int(x*255) for x in image_processor.image_mean))
                image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().to(args.device)

                image_tensors.append(image_tensor)
                batch_data.append(llddata)

                


                if i % 8 == 7 or i == len(iqadata) - 1: # 相当于控制batch_size为8                    
                    with torch.inference_mode():
                        output_logits = model(input_ids.repeat(len(image_tensors), 1),
                            images=torch.cat(image_tensors, 0))["logits"] #(batch_size, sequence_length, config.vocab_size) torch.Size([8, 148, 32000]) sequence随着输入问题改变而改变
                        # print("i:", i," output_logits:", output_logits.shape)
                        output_tensors = model(input_ids.repeat(len(image_tensors), 1),
                            images=torch.cat(image_tensors, 0), output_hidden_states=True)['hidden_states'][-1]
                        output_tensors = torch.mean(output_tensors, dim = 1, keepdim=True) #torch.Size([8, 1, 4096])
                        # print("i:", i," outputs:", outputs)
                        # print("len1:", len(outputs), "len2:", len(outputs[0]), "len3:", len(outputs[0][0]), "len4:", len(outputs[0][0][0])) #list of 33 tensors torch.Size([8, 148, 4096])
                    for j, data in enumerate(batch_data):
                        try:
                            img_name = data["name"]
                        except:
                            img_name = data["img_path"]
                        if question_type == 2:
                            tensor = output_tensors[j] #torch.Size([1, 4096])
                            if img_name.endswith(".jpg"):
                                save_path = os.path.join(args.tensor_root, img_name.replace(".jpg", "") + "_coherence.pt")
                            elif img_name.endswith(".png"):
                                save_path = os.path.join(args.tensor_root, img_name.replace(".png", "") + "_coherence.pt")
                            else:
                                raise ValueError("img_name should end with .jpg or .png")
                             #common sense
                            torch.save(tensor, save_path) 
                            # print("tensor: ", tensor.shape, "save_path: ", save_path)
                        elif question_type == 3:
                            tensor = output_tensors[j]
                            if img_name.endswith(".jpg"):
                                save_path = os.path.join(args.tensor_root, img_name.replace(".jpg", "") + "_semantic_content.pt")
                            elif img_name.endswith(".png"):
                                save_path = os.path.join(args.tensor_root, img_name.replace(".png", "") + "_semantic_content.pt")
                            else:
                                raise ValueError("img_name should end with .jpg or .png")
                            torch.save(tensor, save_path)


                            
                    image_tensors = []
                    batch_data = []



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # MAGAer13/mplug-owl2-llama2-7b /data/wangpuyi_data/mplug_owl2
    parser.add_argument("--model_path", type=str, default="MAGAer13/mplug-owl2-llama2-7b") 
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:3")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--image-aspect-ratio", type=str, default='pad')
   
    parser.add_argument("--csv_path", type=str, default="playground/data/csv_files/AIGCQA_30k.csv")
    parser.add_argument("--tensor_root", type=str, default="/data/wangpuyi_data/mplug_tensor/AIGCQA_30k")
    parser.add_argument("--json_name", type=str, default="AIGCQA_30k.json")
    parser.add_argument("--json_path", type=str, default="playground/data/test_jsons")
    parser.add_argument("--question_type", type=int, default=2)
    parser.add_argument("--img_root", type=str, default="/data/wangpuyi_data/AIGCQA-30k")
    args = parser.parse_args()
    main(args)