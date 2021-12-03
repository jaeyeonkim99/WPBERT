import sys, os

sys.path.append('/workspace/WPBERT/Shared_files')

from ResDAVEnetVQ.models.AudioModels import ResDavenet
import ResDAVEnetVQ.run_utils as RDV
import ResDAVEnetVQ.dataloaders.utils as utils
import torch
import torch.nn as nn
import joblib

import numpy as np
from cpc_feature_reader_MARG import CpcFeatureReader

class embed(): 
    token_pretrain_path = "/data/babymind/pretrained_models/ResDAVEnet323"
    phone_pretrain_path = "/data/babymind/cpc_kmeans/cpc_recon_BCS_500k.pt"
    kmeans_path = "/data/kmeans/kmeans300.bin"
    audio_conf = {'window_stride': 0.01, 'use_raw_length': True}

    def __init__(self, device):
        self.device = device
        self.token_encoder = RDV.load_audio_model_and_state(
            exp_dir="/data/babymind/pretrained_models/ResDAVEnet323").to(self.device)
        self.token_encoder.eval()
        self.token_codebook = self.token_encoder.get_embedding(
            'quant3').cpu().numpy()

        np.random.seed(50)
        self.cls = torch.from_numpy(np.random.uniform(
            0, 5, size=256)).float().to(self.device)
        self.sep = torch.from_numpy(np.random.uniform(
            0, 5, size=256)).float().to(self.device)

        self.phone_padding = 1024

        self.phone_encoder = CpcFeatureReader(
            checkpoint_path=self.phone_pretrain_path, layer=None, device=self.device)
        self.kmeans_model = joblib.load(open(self.kmeans_path, "rb"))
        self.kmeans_model.verbose = False

        return
    
    def search_codebook(self, codebook, key):
        codebook_size = codebook.shape[0]
        if np.array_equal(self.cls, key): return 1024
        elif np.array_equal(self.sep, key): return 1025

        for i in range(0, codebook_size):
            if np.allclose(key, codebook[i], 0, 1e-2):
                return i

        return -1

    def forward(self, audio):
        vq_layer = 'quant3'

        # get token embedding
        token_input, n_frame = utils.compute_spectrogram(
            audio, 16000, self.audio_conf)
        token_input = token_input.unsqueeze(0).to(self.device)
        (_, q_out, preq_out, onehot) = self.token_encoder.get_vq_outputs(
            token_input, vq_layer, True)
        tmp = q_out.squeeze()
        token_embedding = tmp.transpose(0, 1).to(self.device)

        # get CPC feature
        cpc_embedding = self.phone_encoder.get_feats(audio)
        cpc_embedding = cpc_embedding.cpu().numpy()

        # Quantize Using Kmeans
        phone_embedding = self.kmeans_model.predict(cpc_embedding)

        # Align CPC embedding to VQ3 embedding
        length_diff = 4*token_embedding.size()[0] - phone_embedding.shape[0]

        if length_diff == 3:
            phone_embedding = np.insert(
                phone_embedding, 0, phone_embedding[0], axis=0)
            phone_embedding = np.append(phone_embedding, np.array(
                [phone_embedding[-1], phone_embedding[-1]]))
        elif length_diff == 2:
            phone_embedding = np.insert(
                phone_embedding, 0, phone_embedding[0], axis=0)
            phone_embedding = np.append(phone_embedding, phone_embedding[-1])
        elif length_diff == 1:
            phone_embedding = np.append(phone_embedding, phone_embedding[-1])
        elif length_diff == 4:
            phone_embedding = np.insert(
                phone_embedding, 0, phone_embedding[0], axis=0)
            phone_embedding = np.insert(
                phone_embedding, 0, phone_embedding[0], axis=0)
            phone_embedding = np.append(phone_embedding, np.array(
                [phone_embedding[-1], phone_embedding[-1]]))

        if phone_embedding.shape[0] > token_embedding.size()[0]*4:
            phone_embedding = phone_embedding[:token_embedding.size()[0]*4]

        if phone_embedding.shape[0] != token_embedding.size()[0]*4:
            print(token_embedding.shape)
            print(phone_embedding.shape)
            return np.array([0]), np.array([0]), np.array([0]), np.array([0])

        # merge 
        token_result = [self.cls]
        phone_result = [self.phone_padding]
        start_ids = [0]
        end_ids = [0]
        result_entry_phone = [phone_embedding[0]]
        result_entry_token = token_embedding[0]

        for i in range(0, len(token_embedding)):

            if not torch.equal(result_entry_token,token_embedding[i]):
                token_result.append(result_entry_token)
                result_entry_token = token_embedding[i]

                start_ids.append(len(phone_result))
                phone_result.extend(result_entry_phone)
                end_ids.append(len(phone_result)-1)

                result_entry_phone = []
                result_entry_phone.append(phone_embedding[4*i])

                for j in range(1, 4):
                    if result_entry_phone[-1]!=phone_embedding[4*i+j]:
                        result_entry_phone.append(phone_embedding[4*i+j])
            else:
                for j in range(0, 4):
                    if i == 0 and j == 0:
                        continue
                    if result_entry_phone[-1]!=phone_embedding[4*i+j]:
                        result_entry_phone.append(phone_embedding[4*i+j])

        # for last entry
        token_result.append(result_entry_token)
        start_ids.append(len(phone_result))
        phone_result.extend(result_entry_phone)
        end_ids.append(len(phone_result)-1)

        # for [SEP]
        token_result.append(self.sep)
        start_ids.append(len(phone_result))
        phone_result.append(self.phone_padding)
        end_ids.append(len(phone_result)-1)
        
        input_embed = torch.stack(token_result).cpu().numpy()
        phone_result = torch.LongTensor(phone_result).to(self.device)
        start_ids = torch.LongTensor(start_ids).to(self.device)
        end_ids = torch.LongTensor(end_ids).to(self.device)

        input_integer = []
        for i in range(0, input_embed.shape[0]):
            if i==0: input_integer.append(1024)
            elif i==input_embed.shape[0]-1: input_integer.append(1025)
            else:    
                input_integer.append(self.search_codebook(
                self.token_codebook, input_embed[i]))

        token_result = torch.LongTensor(input_integer).to(self.device)
      
        return token_result, phone_result, start_ids, end_ids




                

              

    