

import torch
import torch.nn as nn

import numpy as np
import sys, os
import librosa
import argparse

sys.path.append("/workspace/WPBERT/WPBERT_LM")

from WPBERT_LM import WPBertForMaskedLM
from CharBERT.modeling.configuration_bert import BertConfig
from modify_audio import embed

def search_codebook(codebook, key):
    codebook_size = codebook.size()[0]
    
    for i in range(0, codebook_size):
        if torch.equal(key, codebook[i]):
            return i

    return -1    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--pretrain_path", default=None, type=str, required=True, \
        help="The path of trained Language Model")
    parser.add_argument("--model_name", default=None, type=str, required=True, \
        help = "The name of model that will be included in write path")
    parser.add_argument("--metric", default=None, type=str, required=True, \
        help = "THe metric you want to evaluate. Should be one of Lexical / Syntactic")
    parser.add_argument("--number", default=0, type=int, \
        help="The dev/{number/} you want to evaluate. Should choose on 0, 1, 2, 3")

    parser.add_argument("--gpu", default=0, type=int, \
        help="The gpu number you want to use, 0 or 1")

    args = parser.parse_args()
    

    datapath = f'/data/babymind/eval_dev/{args.metric}/dev{args.number}.txt'
    pretrain_path = args.pretrain_path

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    path = f'/data/babymind/{args.model_name}/{args.metric}'
    if not os.path.exists(path):
        os.makedirs(path)
    writepath = f'/data/babymind/{args.model_name}/{args.metric}/dev{args.number}.txt'
    original_path = f'/data/babymind/eval_data/{args.metric}/dev/'
    
    embedding = embed(device = device)
    
    config = BertConfig(max_positional_embeddings = 512, type_vocab_size=1, vocab_size=1024, \
        num_hidden_layers=8, num_attention_heads=8, hidden_size=512, intermediate_size=2048)
    model = WPBertForMaskedLM.from_pretrained(pretrained_model_name_or_path = pretrain_path, config=config)
    model.to(device)
    model.eval()
    
    read = open(datapath, 'r')
    write = open(writepath, 'w')
    cnt = 0
    
    while True:
        file_path = read.readline()
        if not file_path: break
        file_path =file_path.replace("\n", "")
        audio, sr = librosa.load(file_path, None)   
        input_embed, phone_embed, start_ids, end_ids = embedding.forward(audio)
        input_embed = input_embed.unsqueeze(0)
        phone_embed = phone_embed.unsqueeze(0)
        start_ids = start_ids.unsqueeze(0)
        end_ids = end_ids.unsqueeze(0)
        
        log_prob = 0
        
        for i in range(1, input_embed.size()[1]-1): #to exclude [SEP], [CLS]
            masked_input = input_embed.clone().detach()
            masked_phone = phone_embed.clone().detach()
            masked_start = start_ids.clone().detach()
            masked_end = end_ids.clone().detach()
            original_token = masked_input[0][i].clone().detach()
            
            masked_input[0][i] = embedding.mask
            answer = search_codebook(embedding.token_codebook, original_token)
            
            '''
            print(masked_input.shape)
            print(masked_phone.shape)
            print(masked_start.shape)
            print(masked_start)
            print(masked_end.shape)
            print(masked_end)
            '''
            
            
            with torch.no_grad():
                outputs = model(inputs_embeds = masked_input, char_input_ids = masked_phone,\
                    start_ids = masked_start, end_ids = masked_end,  masked_lm_labels=None)
            prediction_scores = outputs[0].squeeze(dim=0)[i]
            prediction_scores = nn.functional.softmax(prediction_scores, dim=0)
            prob = prediction_scores[answer]
            score = torch.log(prob).cpu().numpy()
            log_prob += score 
            
        log_prob = log_prob/(input_embed.size()[1]-2)
        log_prob = str(log_prob)
        name = file_path.replace(original_path, "")
        name = name.replace(".wav", " ")
        write.write(name+log_prob+"\n")
        
        cnt+=1
        if cnt%1000==0: print(f"{args.metric} %d done"%cnt)
        
    write.close()
                

                
    
    
            
                
                
    
