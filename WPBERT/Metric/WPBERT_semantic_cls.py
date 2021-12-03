

import torch
import torch.nn as nn
import shutil

import numpy as np
import sys, os
import librosa
import argparse

sys.path.append("/workspace/WPBERT/WPBERT_LM")

from WPBERT_LM import WPBertForMaskedLM
from CharBERT.modeling.configuration_bert import BertConfig
from modify_audio import embed

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--pretrain_path", default=None, type=str, require=True, \
        help="The path of trained Language Model")
    parser.add_argument("--hidden", default=None, type=int, required=True,\
        help="The hidden layer you want to get representations. Should be one of seq, seqchar, char")
    parser.add_argument("--repr", default=None, type=str, required=True, \
        help="The representation type you want to get")
    parser.add_argument("--gpu", default=0, type=1, \
        help="The gpu number you want to use")

    args = parser.parse_args()  

    datapath = '/data/babymind/eval_data/semantic/dev'
    pretrain_path = args.pretrain_path
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    embedding = embed(device = device)
    
    config = BertConfig(max_positional_embeddings = 512, type_vocab_size=1, vocab_size=1024, \
        num_hidden_layers=8, num_attention_heads=8, hidden_size=512, intermediate_size=2048, output_hidden_states=True)
    model = WPBertForMaskedLM.from_pretrained(pretrained_model_name_or_path = pretrain_path, config=config)
    model.to(device)
    model.eval()
    for (path, dirs, files) in os.walk(datapath):
        for file_name in files:
            if file_name.find('.wav')!=-1:
                file_path = os.path.join(path, file_name)
                audio, sr = librosa.load(file_path, None)   
                input_embed, phone_embed, start_ids, end_ids = embedding.forward(audio)
                input_embed = input_embed.unsqueeze(0)
                phone_embed = phone_embed.unsqueeze(0)
                start_ids = start_ids.unsqueeze(0)
                end_ids = end_ids.unsqueeze(0)
                
                
                with torch.no_grad():
                    outputs = model(inputs_embeds = input_embed, char_input_ids = phone_embed,\
                        start_ids = start_ids, end_ids = end_ids, masked_lm=False,\
                            masked_lm_labels=None)

                sequence_repr = outputs[2].squeeze(dim=0)
                tmp = sequence_repr.cpu().numpy()
                result = np.array([tmp, tmp])       
                
                dir_path = path.replace("eval_data", "WPBERT_LM_semantic/cls")
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)
                
                new_path = file_path.replace(".wav", ".txt")
                new_path = new_path.replace("eval_data", "WPBERT_LM_semantic/cls")
               

                np.savetxt(new_path, result)
                

                
    
    
            
                
                
    
