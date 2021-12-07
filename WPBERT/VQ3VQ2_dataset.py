import torch
from torch.utils.data import Dataset

import numpy as np
import os
import collections
import random

class VQ3VQ2_dataset(Dataset):


    def __init__(self, datapath, phone_type, logger = None, masked_lm = True, mlm_prob = 0.15, rng=None): 
        #datapath: get absolute path of directory as input
        super().__init__()
        self.datapath = datapath
        self.masked_lm  = masked_lm
        self.mlm_prob = mlm_prob
        self.path = []

        self.mask = 1026
        self.cls = 1024
        self.sep = 1025
        self.padding =1027

        self.phone_padding = 1024

        if rng==None: 
            self.rng = random.Random(100)
        else: self.rng = rng 

        cnt = 0
        for (path, dirs, files) in os.walk(datapath):
            for file_name in files:
                if file_name.find('.npz')!=-1:
                    file_path = os.path.join(path, file_name)
                    file_path = file_path.replace(datapath, "")
                    self.path.append(file_path)

                    if logger!=None:
                        cnt+=1
                        if cnt%20000==0: logger.info("Read %d files", cnt)
    

    def __getitem__(self, index):
        
        file_path = self.datapath + self.path[index]
        
        data = np.load(file_path)
        input_embed = data['input_embed']
        char_embed = data['char_embed']
        start_ids = data['start_ids']
        end_ids = data['end_ids']

        data.close()

        input_embed = torch.LongTensor(input_embed)
        char_embed = torch.LongTensor(char_embed)
        start_ids = torch.LongTensor(start_ids)
        end_ids = torch.LongTensor(end_ids)


        #Make Attention Mask
        word_attention_mask = torch.LongTensor([1]*input_embed.shape[0])

        #creat masking
        if self.masked_lm == True:
            #masking for MLM
            input_embed, masked_lm_label = self.create_masked_lms(input_embed, self.mlm_prob, self.rng)
            masked_lm_label = torch.LongTensor(masked_lm_label)

        else: masked_lm_label = None

        return (input_embed, char_embed, start_ids, end_ids, masked_lm_label, word_attention_mask)


    def __len__(self):
        return len(self.path)

    def search_codebook(self, codebook, key):
        codebook_size = codebook.size()[0]
        
        for i in range(0, codebook_size):
            if torch.equal(key, codebook[i]):
                return i

        return -1

    def create_masked_lms(self, tokens, masked_lm_prob, rng):
        MaskedLmInstance = collections.namedtuple("MaskedLmInstance",["index", "label"])
        
        cand_indexes =list(range(0, len(tokens)))
        
        rng.shuffle(cand_indexes)

        output_tokens = tokens.clone().detach()

        num_to_predict = int(round(len(tokens) * masked_lm_prob))

        masked_lms = []
        
        for index in cand_indexes:
            if len(masked_lms) >= num_to_predict:
                break

            if tokens[index]==self.cls or tokens[index]==self.sep:
                continue
            
            masked_token = None
            #80% of the time, replace with mask_embed(correspond to [MASK])
            if rng.random() < 0.8:
                masked_token = self.mask
            else:
                # 20% of the time, keep original
                masked_token = tokens[index]
                                
                    
            output_tokens[index] = masked_token
            masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))
                
        assert len(masked_lms) <= num_to_predict
        masked_lms = sorted(masked_lms, key=lambda x: x.index)

        masked_lm_labels = [-1] * len(tokens)
        for p in masked_lms:
            masked_lm_labels[p.index] = p.label
            
        return output_tokens, masked_lm_labels       


def VQ2_embed_collate_fn(batch):

    word_padding = 1027
    phone_padding = 1024

    batch_token_result = []
    batch_phone_result = []
    batch_start_ids = []
    batch_end_ids = []
    batch_masked_lm = []

    #Attention Mask
    batch_word_am =[]

    batch_size = len(batch)

    for item in batch:
        batch_token_result.append(item[0])
        batch_phone_result.append(item[1])        
        batch_start_ids.append(item[2])
        batch_end_ids.append(item[3])

        if item[4]!=None:
            batch_masked_lm.append(item[4])
        
        batch_word_am.append(item[5])

    max_token_length = 0
    max_phone_length = 0
    
    for i in range(0, batch_size):
        if len(batch_token_result[i])>max_token_length:
            max_token_length = len(batch_token_result[i])
        if len(batch_phone_result[i])>max_phone_length:
            max_phone_length = len(batch_phone_result[i])


    for i in range(0, batch_size):

        padd_amount2 = max_phone_length-batch_phone_result[i].size()[0]
        new2 = torch.LongTensor([phone_padding]*padd_amount2)
        batch_phone_result[i] = torch.cat([batch_phone_result[i], new2], dim=0)        
        padd_amount = max_token_length-batch_token_result[i].size()[0]

        if padd_amount>0:
            new = torch.LongTensor([word_padding]*padd_amount)
            batch_token_result[i] = torch.cat([batch_token_result[i], new], dim=0)

            #Attention Mask
            batch_word_am[i] = torch.cat([batch_word_am[i], torch.LongTensor([0]*padd_amount)])

            #start ids & end ids
            padd_start = batch_end_ids[i][-1]
            padding = torch.LongTensor([padd_start]*padd_amount)
            batch_start_ids[i] = torch.cat([batch_start_ids[i], padding], dim=0)
            batch_end_ids[i] = torch.cat([batch_end_ids[i], padding], dim=0)
            
            
            if len(batch_masked_lm)!=0:
                #masked_lmbatch_masked_lm 
                batch_masked_lm[i] = torch.cat([batch_masked_lm[i], torch.LongTensor([-1]*padd_amount)], dim=0)
                            
    return_token = torch.stack(batch_token_result)
    return_phone = torch.stack(batch_phone_result)
            
    #for one-hot encoding, change to long-tensor
    batch_start_ids = torch.stack(batch_start_ids)
    batch_end_ids = torch.stack(batch_end_ids)

    batch_word_am = torch.stack(batch_word_am)

        
    if len(batch_masked_lm)!=0:
        batch_masked_lm = torch.stack(batch_masked_lm)
        
    
    return return_token, return_phone, batch_start_ids, batch_end_ids, batch_masked_lm, batch_word_am
