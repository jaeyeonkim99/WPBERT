import sys, os

sys.path.append('/workspace/WPBERT/Shared_files')

from ResDAVEnetVQ.models.AudioModels import ResDavenet
import ResDAVEnetVQ.run_utils as RDV
import ResDAVEnetVQ.dataloaders.utils as utils
import torch
import torch.nn as nn
import joblib

import numpy as np

class embed(): 
    token_pretrain_path = "/data/babymind/pretrained_models/ResDAVEnet323"
    phone_pretrain_path = "/data/babymind/pretrained_models/ResDAVEnet02"
    #device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    audio_conf = {'window_stride': 0.01, 'use_raw_length': True}

    def __init__(self, device):
        self.device = device
        self.token_encoder = RDV.load_audio_model_and_state(exp_dir=self.token_pretrain_path)
        self.token_encoder.to(self.device)
        self.token_encoder.eval()
        for param in self.token_encoder.parameters():
            param.requires_grad_(False)

        self.phone_encoder = RDV.load_audio_model_and_state(exp_dir=self.phone_pretrain_path)
        self.phone_encoder.to(self.device)
        self.phone_encoder.eval()

        for param in self.phone_encoder.parameters():
            param.requires_grad_(False)
        
        self.token_codebook = self.token_encoder.get_embedding('quant3').cpu().numpy()

        self.phone_codebook = self.phone_encoder.get_embedding('quant2').cpu().numpy()

        

        #make embedding for mask using random
        #fixed seed, so that same maske, cls, seq for the training
        np.random.seed(50) 
        self.mask= torch.from_numpy(np.random.uniform(0, 5, size=256)).float().to(self.device)
        self.cls = torch.from_numpy(np.random.uniform(0, 5, size=256)).float().to(self.device)
        self.sep =  torch.from_numpy(np.random.uniform(0, 5, size=256)).float().to(self.device)

        return

    
    def search_codebook(self, codebook, key):
        codebook_size = codebook.shape[0]

        for i in range(0, codebook_size):
            if np.allclose(key, codebook[i], 0, 1e-2):
                return i

        return -1

    def forward(self, audio):
        vq_layer = 'quant3'
        vq_layer2 = 'quant2'

        #get token embedding
        token_input, n_frame = utils.compute_spectrogram(audio, 16000, self.audio_conf)
        token_input = token_input.unsqueeze(0).to(self.device)
        (_, q_out, preq_out, onehot) = self.token_encoder.get_vq_outputs(token_input, vq_layer, True)
        tmp = q_out.squeeze()            
        token_embedding = tmp.transpose(0,1)   

        #get phone embedding
        (_, q_out2, preq_out2, onehot2) = self.phone_encoder.get_vq_outputs(token_input, vq_layer2, True)
        tmp2 = q_out2.squeeze()            
        phone_embedding = tmp2.transpose(0,1)

        #merge
        zero = torch.zeros(256, dtype=float).to(self.device)
        token_result = [self.cls]
        phone_result = [zero]
        start_ids = [0]
        end_ids = [0]
        result_entry_phone = []
        result_entry_token = token_embedding[0]
            
            
        for i in range(0, len(token_embedding)):                    
            if i==0:
                if 2*i<len(phone_embedding):
                    result_entry_phone.append(phone_embedding[2*i])

                if 2*i+1 <len(phone_embedding) and not torch.equal(result_entry_phone[-1], phone_embedding[2*i+1]):
                    result_entry_phone.append(phone_embedding[2*i+1])
                continue
            
            if not torch.equal(result_entry_token, token_embedding[i]):
                token_result.append(result_entry_token)                    
                result_entry_token = token_embedding[i]
                
                start_ids.append(len(phone_result))
                phone_result.extend(result_entry_phone)
                end_ids.append(len(phone_result)-1)
                
                if 2*i<len(phone_embedding):
                    result_entry_phone=[]
                    result_entry_phone.append(phone_embedding[2*i])
                else: 
                    tmp = result_entry_phone[-1]
                    result_entry_phone = []
                    result_entry_phone.append(tmp)
                
                if 2*i+1 <len(phone_embedding) and not torch.equal(result_entry_phone[-1], phone_embedding[2*i+1]):
                    result_entry_phone.append(phone_embedding[2*i+1])
            else:
                
                if 2*i<len(phone_embedding) and not torch.equal(result_entry_phone[-1], phone_embedding[2*i]):
                    result_entry_phone.append(phone_embedding[2*i])
                if  2*i+1 <len(phone_embedding) and not torch.equal(result_entry_phone[-1], phone_embedding[2*i+1]):
                    result_entry_phone.append(phone_embedding[2*i+1])    

        #for last entry                
        token_result.append(result_entry_token)
        start_ids.append(len(phone_result))
        phone_result.extend(result_entry_phone)
        end_ids.append(len(phone_result)-1)

        #for [SEP]
        token_result.append(self.sep)
        start_ids.append(len(phone_result))
        phone_result.append(zero)
        end_ids.append(len(phone_result)-1)

        token_result = torch.stack(token_result).cpu().numpy()
        phone_result = torch.stack(phone_result).cpu().numpy()


        word_integer = []
        phone_integer = []

        for i in range(0, token_result.shape[0]):
            if i==0:
                word_integer.append(1024)
                continue
            elif i==token_result.shape[0]-1:
                word_integer.append(1025)
                continue

            word_integer.append(self.search_codebook(self.token_codebook, token_result[i]))

        for i in range(0, phone_result.shape[0]):
            if i==0:
                phone_integer.append(1024)
                continue
            elif i==token_result.shape[0]-1:
                phone_integer.append(1024)
                continue

            phone_integer.append(self.search_codebook(self.phone_codebook, phone_result[i]))

        return torch.LongTensor(word_integer).to(self.device), torch.LongTensor(phone_integer).to(self.device), \
            torch.LongTensor(start_ids).to(self.device), torch.LongTensor(end_ids).to(self.device)






                

              

    