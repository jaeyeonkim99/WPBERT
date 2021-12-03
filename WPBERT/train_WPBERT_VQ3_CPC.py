# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""


from __future__ import absolute_import, division, print_function    

import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil
import copy
import time
import traceback

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
import collections
import sys
import io
from datetime import datetime

#sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')

sys.path.append("/workspace/WPBERT/Integer/WPBERT_REP_WP_Modified/*")
sys.path.append("/workspace/WPBERT/Integer/WPBERT_REP_WP_Modified/CharBERT/*")

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import tqdm, trange

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    get_linear_schedule_with_warmup,
)

#Added

from embed_dataset import Embed_dataset
from WPBERT import WPBertForMaskedLM
from CharBERT.modeling.configuration_bert import BertConfig
from embed_dataset import embed_collate_fn

from apex.parallel import DistributedDataParallel as DDP

logger = logging.getLogger(__name__)

#Set Environmental variables
os.environ["CUDA_VISIBLE_DEVICES"] = '4, 5'
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '29500'

'''
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
logger.addHandler(stream_handler)
'''

train_path = '/data/babymind/LibriSpeech_embed_integer/train'
validate_path = '/data/babymind/LibriSpeech_embed_integer/dev'
output_dir = "/data/babymind/WPBERT_models/RES_INTEGER/WPBERT_REP_WP_BERT"

def load_and_cache_examples(datapath):
    dataset = Embed_dataset(datapath, logger=logger)
    return dataset


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, validation_dataset, model):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    args.val_batch_size = args.per_gpu_val_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    validate_sampler = RandomSampler(validation_dataset) if args.local_rank == -1 else DistributedSampler(validation_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, collate_fn = embed_collate_fn, num_workers=32, pin_memory=True)
    validate_dataloader =DataLoader(validation_dataset, sampler = validate_sampler, batch_size =args.val_batch_size,  collate_fn = embed_collate_fn, num_workers=32, pin_memory = True)

    #for validation
    validation_losses = []

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    # Check if saved optimizer or scheduler states exist
    '''
    if os.path.isfile(os.path.join(args.model_name_or_path, 'optimizer.pt')) and os.path.isfile(os.path.join(args.model_name_or_path, 'scheduler.pt')):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, 'optimizer.pt')))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, 'scheduler.pt')))
    '''
    
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = DDP(model, delay_allreduce=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0

    '''
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        # set global_step to gobal_step of last saved checkpoint from model path
        #global_step = int(args.model_name_or_path.split('-')[-1].split('/')[0])
        epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
    '''
    
    tr_loss, logging_loss = 0.0, 0.0
    
    model.zero_grad()

    train_iterator = trange(epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)


    logger.info("Starting training at")
    logger.info(datetime.now())

    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        
        logger.info("Starting Epoch %d", epoch)
        try: 
            for step, batch in enumerate(epoch_iterator):
                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                inputs_embeds, char_input_ids, start_ids, end_ids, masked_lm_labels, attention_mask = batch

                inputs_embeds = inputs_embeds.to(args.device)
                char_input_ids = char_input_ids.to(args.device)
                start_ids = start_ids.to(args.device)
                end_ids = end_ids.to(args.device)
                if masked_lm_labels!=None:
                    masked_lm_labels = masked_lm_labels.to(args.device)
                attention_mask = attention_mask.to(args.device)

                '''
                print(inputs_embeds.shape)
                print(char_input_ids.shape)
                print(start_ids.shape)
                print(end_ids.shape)
                print(masked_lm_labels.shape)
                print(attention_mask.shape)
                ''' 

                model.train()
                outputs = model(inputs_embeds = inputs_embeds, char_input_ids = char_input_ids, start_ids = start_ids, \
                    end_ids = end_ids, masked_lm_labels = masked_lm_labels, attention_mask = attention_mask)

                if args.output_debug:
                    print(f'mask_lm loss: {outputs[0]}')
                
                loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                tr_loss += loss.item()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad() 
                    global_step += 1

                    if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                        # Log metrics
                        tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                        tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args.logging_steps, global_step)
                        logging_loss = tr_loss

                if args.max_steps > 0 and global_step > args.max_steps:
                    #epoch_iterator.close()
                    break
            if args.max_steps > 0 and global_step > args.max_steps:
                train_iterator.close()
                break
            
            
            #validation for every epoch
            validation_loss = 0
            model.eval()            
            with torch.no_grad():
                validate_iterator = tqdm(validate_dataloader, desc="Validation", disable=args.local_rank not in [-1, 0])
                for idx, batch in enumerate(validate_iterator):
                    inputs_embeds, char_input_ids, start_ids, end_ids, masked_lm_labels, attention_mask = batch
                    inputs_embeds = inputs_embeds.to(args.device)
                    char_input_ids = char_input_ids.to(args.device)
                    start_ids = start_ids.to(args.device)
                    end_ids = end_ids.to(args.device)
                    if masked_lm_labels!=None:
                        masked_lm_labels = masked_lm_labels.to(args.device)
                    attention_mask = attention_mask.to(args.device)

                    outputs = model(inputs_embeds = inputs_embeds, char_input_ids = char_input_ids, start_ids = start_ids, \
                        end_ids = end_ids, masked_lm_labels = masked_lm_labels, attention_mask = attention_mask)
                    
                    loss = outputs[0]                    
                    if args.n_gpu > 1:
                        loss = loss.mean()  # mean() to average on multi-gpu parallel training

                    validation_loss+=loss.item()

            if args.local_rank in [-1, 0]:
                tb_writer.add_scalar('validation loss', validation_loss, epoch)
            logger.info("Validation loss for epoch %d: %f", epoch, validation_loss)
            validation_losses.append(validation_loss)
            
            #validation: to avoid overfitting
            if len(validation_losses)>=3:
                if validation_losses[-3]<validation_losses[-2] and validation_losses[-3]<validation_losses[-1]:
                    logger.warn("Break out the loop: last epoch: %d", epoch+1)
                    break

            if len(validation_losses)>=2:
                if args.local_rank in [-1, 0] and validation_losses[-1]<=validation_losses[-2]:
                    checkpoint_prefix = 'checkpoint'
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, '{}-{}'.format(checkpoint_prefix, epoch))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                        model_to_save.save_pretrained(output_dir)

                        torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                        logger.info("Saving model checkpoint to %s", output_dir)

                        #_rotate_checkpoints(args, checkpoint_prefix)

                        torch.save(optimizer.state_dict(), os.path.join(output_dir, 'optimizer.pt'))
                        torch.save(scheduler.state_dict(), os.path.join(output_dir, 'scheduler.pt'))
                        logger.info("Saving optimizer and scheduler states to %s", output_dir)
            else: 
                if args.local_rank in [-1, 0]:
                    checkpoint_prefix = 'checkpoint'
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, '{}-{}'.format(checkpoint_prefix, epoch))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                        model_to_save.save_pretrained(output_dir)

                        torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                        logger.info("Saving model checkpoint to %s", output_dir)

                        #_rotate_checkpoints(args, checkpoint_prefix)

                        torch.save(optimizer.state_dict(), os.path.join(output_dir, 'optimizer.pt'))
                        torch.save(scheduler.state_dict(), os.path.join(output_dir, 'scheduler.pt'))
                        logger.info("Saving optimizer and scheduler states to %s", output_dir)


        except Exception:
            checkpoint_prefix = 'errorpoint'
            output_dir = os.path.join(args.output_dir, '{}-{}'.format(checkpoint_prefix, args.local_rank))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                model_to_save.save_pretrained(output_dir)

                torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                logger.info("Saving model checkpoint to %s", output_dir)

                #_rotate_checkpoints(args, checkpoint_prefix)

                torch.save(optimizer.state_dict(), os.path.join(output_dir, 'optimizer.pt'))
                torch.save(scheduler.state_dict(), os.path.join(output_dir, 'scheduler.pt'))
                logger.info("Saving optimizer and scheduler states to %s", output_dir)
                
                txt_dir = output_dir + '/errorpoint.txt'
                f = open(txt_dir, 'w')
                f.write("Error made in epoch %d\n"%epoch)
                err = traceback.format_exc()
                f.write(err)
                f.close()                
            break
        
    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step

def main():
    parser = argparse.ArgumentParser()

    
    parser.add_argument("--output_dir", default=output_dir, type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")

    parser.add_argument("--mlm", default = True, type = bool,
                        help="Train with masked-language modeling loss")
   
    parser.add_argument("--mlm_probability", type=float, default=0.15,
                        help="Ratio of tokens to mask for masked language modeling loss")
    
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")

    parser.add_argument("--output_debug", action='store_true',
                        help="Whether to output the debug information.")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
                        #256 bert
                        #32 charbert
                        #512k token Zerospeech

    parser.add_argument("--per_gpu_val_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for validate.")
                        
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=1e-4, type=float, #CharBERT 5e-5, BERT: 1e-4 
                        help="The initial learning rate for Adam.") #ok???
    parser.add_argument("--weight_decay", default=0.01, type=float, #원래 0
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=40.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
            

    parser.add_argument("--warmup_steps", default=10000, type=int,  #원래 0
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=50
    ,
                        help="Log every X updates steps.")

    parser.add_argument('--save_total_limit', type=int, default=None,
                        help='Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default')
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--fp16', type=bool, default=True, 
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
   
    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))
  
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl',  rank= args.local_rank, world_size = 2)
        args.n_gpu = 1
    args.device = device

    

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)
    
    log_path = output_dir + '/log.txt'
    file_handler =logging.FileHandler(log_path)
    logger.addHandler(file_handler)
    
    # Set seed
    set_seed(args)

    #Instantiate model
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

     
    config = BertConfig(max_positional_embeddings = 512, type_vocab_size=1, vocab_size=1028, \
        num_hidden_layers=12, num_attention_heads=12, hidden_size=768, intermediate_size=3072)
    model = WPBertForMaskedLM(config=config)    
    model.to(args.device)

    cnt = 0
    for p in list(model.bert.char_embeddings.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        cnt+=nn

    logger.info(cnt)        

    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training/evaluation parameters %s", args)
    
    # Training
    if args.do_train:
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache

        logger.info("loading train_dataset")
        try: 
            train_dataset = load_and_cache_examples(train_path)
        except Exception:
            err = traceback.format_exc()
            logger.error(err)
            
        logger.info("loading validate dataset")
        validation_dataset = load_and_cache_examples(validate_path)

        if args.local_rank == 0:
            torch.distributed.barrier()
        try:
            global_step, tr_loss = train(args, train_dataset, validation_dataset, model)
            logger.info("Ending training at")
            logger.info(datetime.now())
            logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
        except Exception:
            err = traceback.format_exc()
            logger.error(err)

    
    # Saving best-practices: if you use save_pretrained for the model and tokenizer, you can reload them using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

        # Load a trained model and vocabulary that you have fine-tuned
        model = WPBertForMaskedLM.from_pretrained(args.output_dir)
        model.to(args.device)

    return 


if __name__ == "__main__":
    main()
