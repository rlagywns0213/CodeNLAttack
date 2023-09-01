# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.
from __future__ import absolute_import
import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
# os.environ["CUDA_VISIBLE_DEVICES"]= "0"  # Set the GPU 2 to use
import copy
import json
import time
import sys
import codecs
import pickle
import torch
import json
import random
import logging
import argparse
import numpy as np
from io import open
from itertools import cycle
import torch.nn as nn
from model import Seq2Seq
from tqdm import tqdm, trange
from bleu import _bleu, attack_bleu
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
#from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
#                          RobertaConfig, RobertaModel, RobertaTokenizer)
#MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer)}

from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          XLMRobertaConfig, XLMRobertaModel, XLMRobertaTokenizer, RobertaForMaskedLM)
MODEL_CLASSES = {'roberta': (XLMRobertaConfig, XLMRobertaModel, XLMRobertaTokenizer)}

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

filter_words = ['a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'ain', 'all', 'almost',
                'alone', 'along', 'already', 'also', 'although', 'am', 'among', 'amongst', 'an', 'and', 'another',
                'any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere', 'are', 'aren', "aren't", 'around', 'as',
                'at', 'back', 'been', 'before', 'beforehand', 'behind', 'being', 'below', 'beside', 'besides',
                'between', 'beyond', 'both', 'but', 'by', 'can', 'cannot', 'could', 'couldn', "couldn't", 'd', 'didn',
                "didn't", 'doesn', "doesn't", 'don', "don't", 'down', 'due', 'during', 'either', 'else', 'elsewhere',
                'empty', 'enough', 'even', 'ever', 'everyone', 'everything', 'everywhere', 'except', 'first', 'for',
                'former', 'formerly', 'from', 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'he', 'hence',
                'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'him', 'himself', 'his',
                'how', 'however', 'hundred', 'i', 'if', 'in', 'indeed', 'into', 'is', 'isn', "isn't", 'it', "it's",
                'its', 'itself', 'just', 'latter', 'latterly', 'least', 'll', 'may', 'me', 'meanwhile', 'mightn',
                "mightn't", 'mine', 'more', 'moreover', 'most', 'mostly', 'must', 'mustn', "mustn't", 'my', 'myself',
                'namely', 'needn', "needn't", 'neither', 'never', 'nevertheless', 'next', 'no', 'nobody', 'none',
                'noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'o', 'of', 'off', 'on', 'once', 'one', 'only',
                'onto', 'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'per',
                'please', 's', 'same', 'shan', "shan't", 'she', "she's", "should've", 'shouldn', "shouldn't", 'somehow',
                'something', 'sometime', 'somewhere', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs',
                'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore', 'therein',
                'thereupon', 'these', 'they', 'this', 'those', 'through', 'throughout', 'thru', 'thus', 'to', 'too',
                'toward', 'towards', 'under', 'unless', 'until', 'up', 'upon', 'used', 've', 'was', 'wasn', "wasn't",
                'we', 'were', 'weren', "weren't", 'what', 'whatever', 'when', 'whence', 'whenever', 'where',
                'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while',
                'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'with', 'within', 'without', 'won',
                "won't", 'would', 'wouldn', "wouldn't", 'y', 'yet', 'you', "you'd", "you'll", "you're", "you've",
                'your', 'yours', 'yourself', 'yourselves']
filter_words = set(filter_words)

class Example(object):
    """A single training/test example."""
    def __init__(self,
                 idx,
                 source,
                 target,
                 ):
        self.idx = idx
        self.source = source
        self.target = target

class Feature(object):
    def __init__(self, idx, source,target):
        self.idx = idx
        self.source = source
        self.target = target
        self.final_adverse = source
        self.query = 0
        self.change = 0
        self.success = 0
        self.sim = 0.0
        self.time = 0.0
        self.original_bleu = 0.0
        self.attack_bleu = 0.0
        self.most_gap = 0.0
        self.changes = []
        self.model_pred = ''

def open_json(data):
    with open(data, 'r') as f:
        json_data = json.load(f)
        return json_data

def read_examples_json():
    df_all = open_json('/home/rlagywns0213/22_hj/text-to-text_3090/code/attack_result/bleu_beam_1080.json')
    examples=[]
    idx = 0
    for i in df_all:
        if i['gold_target'] == i['pred']:
            examples.append(
            Example(
                    idx = idx,
                    source=i['source'].strip(),
                    target=i['gold_target'].strip(),
                    )
            )
            idx+=1  
    # with codecs.open(src_filename, 'r', 'utf-8') as f1, codecs.open(trg_filename, 'r', 'utf-8') as f2:
    #         for line1,line2 in zip(f1, f2):
    #             examples.append(
    #             Example(
    #                     idx = idx,
    #                     source=line1.strip(),
    #                     target=line2.strip(),
    #                     ) 
    #             )
    #             idx+=1
    return examples

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 example_id,
                 source_ids,
                 target_ids,
                 source_mask,
                 target_mask,

    ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.source_mask = source_mask
        self.target_mask = target_mask       
        


def convert_examples_to_features(examples, tokenizer, args,stage=None):
    features = []
    for example_index, example in enumerate(examples):
        #source
        source_tokens = tokenizer.tokenize(example.source)[:args.max_source_length-2]
        source_tokens =[tokenizer.cls_token]+source_tokens+[tokenizer.sep_token]
        source_ids =  tokenizer.convert_tokens_to_ids(source_tokens) 
        source_mask = [1] * (len(source_tokens))
        padding_length = args.max_source_length - len(source_ids)
        source_ids+=[tokenizer.pad_token_id]*padding_length
        source_mask+=[0]*padding_length
 
        #target
        if stage=="test":
            target_tokens = tokenizer.tokenize("None") #test 할때는 None값을 target_token으로 : ['▁No', 'ne']
        else:
            target_tokens = tokenizer.tokenize(example.target)[:args.max_target_length-2]
        target_tokens = [tokenizer.cls_token]+target_tokens+[tokenizer.sep_token] # ['<s>', '▁No', 'ne', '</s>']
        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
        target_mask = [1] *len(target_ids)
        padding_length = args.max_target_length - len(target_ids)
        target_ids+=[tokenizer.pad_token_id]*padding_length
        target_mask+=[0]*padding_length   
   
        if example_index < 5:
            if stage=='train':
                logger.info("*** Example ***")
                logger.info("idx: {}".format(example.idx))

                logger.info("source_tokens: {}".format([x.replace('\u0120','_') for x in source_tokens]))
                logger.info("source_ids: {}".format(' '.join(map(str, source_ids))))
                logger.info("source_mask: {}".format(' '.join(map(str, source_mask))))
                
                logger.info("target_tokens: {}".format([x.replace('\u0120','_') for x in target_tokens]))
                logger.info("target_ids: {}".format(' '.join(map(str, target_ids))))
                logger.info("target_mask: {}".format(' '.join(map(str, target_mask))))
       
        features.append(
            InputFeatures(
                 example_index,
                 source_ids,
                 target_ids,
                 source_mask,
                 target_mask,
            )
        )
    return features


def _truncate_seq_pair(tokens_a, tokens_b,tokens_c, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    
    while True:
        total_length = len(tokens_a) + len(tokens_b)+len(tokens_c)
        if total_length <= max_length:
            break
        if len(tokens_a) >= len(tokens_b) and len(tokens_a)>=len(tokens_c):
            tokens_a.pop()
        elif len(tokens_b) >= len(tokens_a) and len(tokens_b)>=len(tokens_c):
            tokens_b.pop()
        else:
            tokens_c.pop()

def set_seed(args):
    """set random seed."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def get_sim_embed(embed_path, sim_path):
    id2word = {}
    word2id = {}

    with open(embed_path, 'r', encoding='utf-8') as ifile:
        for line in ifile:
            word = line.split()[0]
            if word not in id2word:
                id2word[len(id2word)] = word
                word2id[word] = len(id2word) - 1

    cos_sim = np.load(sim_path) # textfoolr에서 돌린 cos_sim_counter_fitting 무엇을 뜻하는걸까?
    return cos_sim, word2id, id2word

def _tokenize(seq, tokenizer):
    seq = seq.replace('\n', '').lower()
    words = seq.split(' ')

    sub_words = []
    keys = []
    index = 0
    for word in words:
        sub = tokenizer.tokenize(word)
        sub_words += sub
        keys.append([index, index + len(sub)])
        index += len(sub)

    return words, sub_words, keys

def _get_masked(words):
    len_text = len(words)
    masked_words = []
    for i in range(len_text - 1):
        masked_words.append(words[0:i] + ['[UNK]'] + words[i + 1:])
    # list of words
    return masked_words

def get_important_scores(words,gold_target, tgt_model, orig_bleu, tokenizer, batch_size, args):
    masked_words = _get_masked(words)
    texts = [' '.join(words) for words in masked_words]  # list of text of masked words
    all_examples = []

    #source_to_feature
    masked_features = []
    for example_index, text in enumerate(texts):
        source_tokens = tokenizer.tokenize(text)[:args.max_source_length-2]
        source_tokens =[tokenizer.cls_token]+source_tokens+[tokenizer.sep_token]
        source_ids =  tokenizer.convert_tokens_to_ids(source_tokens) 
        source_mask = [1] * (len(source_tokens))
        padding_length = args.max_source_length - len(source_ids)
        source_ids+=[tokenizer.pad_token_id]*padding_length
        source_mask+=[0]*padding_length

        #target_for_attack
        target_tokens = tokenizer.tokenize("None")
        target_tokens = [tokenizer.cls_token]+target_tokens+[tokenizer.sep_token] # ['<s>', '▁No', 'ne', '</s>']
        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
        target_mask = [1] *len(target_ids)
        padding_length = args.max_target_length - len(target_ids)
        target_ids+=[tokenizer.pad_token_id]*padding_length
        target_mask+=[0]*padding_length

        masked_features.append(InputFeatures(
                 example_index,
                 source_ids,
                 target_ids,
                 source_mask,
                 target_mask,
            ))
    
    all_source_ids = torch.tensor([f.source_ids for f in masked_features], dtype=torch.long) #8000,256
    all_source_mask = torch.tensor([f.source_mask for f in masked_features], dtype=torch.long) #8000,256
    eval_data = TensorDataset(all_source_ids,all_source_mask)   
    # Calculate bleu
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
    tgt_model.eval() 
    p=[]
    for batch in eval_dataloader:
        batch = tuple(t.to(args.device) for t in batch)
        source_ids,source_mask= batch                  
        with torch.no_grad():
            preds = tgt_model(source_ids=source_ids,source_mask=source_mask, args = args)  
            for pred in preds:
                t=pred[0].cpu().numpy()
                t=list(t)
                if 0 in t:
                    t=t[:t.index(0)]
                text = tokenizer.decode(t,clean_up_tokenization_spaces=False)
                p.append(text)
    masked_bleu_results = []
    for beam_pred in p:
        masked_bleu = round(attack_bleu(gold_target, [beam_pred]),2)
        masked_bleu_results.append(masked_bleu)
    important_scores = torch.tensor(masked_bleu_results) - orig_bleu

    return important_scores


def get_substitues(substitutes, tokenizer, mlm_model, use_bpe, substitutes_score=None, threshold=3.0, args=''):
    # substitues L,k
    # from this matrix to recover a word
    words = []
    sub_len, k = substitutes.size()  # sub-len, k

    if sub_len == 0:
        return words
        
    elif sub_len == 1:
        for (i,j) in zip(substitutes[0], substitutes_score[0]):
            if threshold != 0 and j < threshold:
                break
            words.append(tokenizer._convert_id_to_token(int(i)))
    else:
        if use_bpe == 1:
            words = get_bpe_substitues(substitutes, tokenizer, mlm_model, args)
        else:
            return words
    #
    # print(words)
    return words

def get_bpe_substitues(substitutes, tokenizer, mlm_model,args):
    # substitutes L, k

    substitutes = substitutes[0:12, 0:4] # maximum BPE candidates

    # find all possible candidates 

    all_substitutes = []
    for i in range(substitutes.size(0)):
        if len(all_substitutes) == 0:
            lev_i = substitutes[i]
            all_substitutes = [[int(c)] for c in lev_i]
        else:
            lev_i = []
            for all_sub in all_substitutes:
                for j in substitutes[i]:
                    lev_i.append(all_sub + [int(j)])
            all_substitutes = lev_i

    # all substitutes  list of list of token-id (all candidates)
    c_loss = nn.CrossEntropyLoss(reduction='none')
    word_list = []
    # all_substitutes = all_substitutes[:24]
    all_substitutes = torch.tensor(all_substitutes) # [ N, L ]
    all_substitutes = all_substitutes[:24].to(args.device)
    # print(substitutes.size(), all_substitutes.size())
    N, L = all_substitutes.size()
    word_predictions = mlm_model(all_substitutes)[0] # N L vocab-size
    ppl = c_loss(word_predictions.view(N*L, -1), all_substitutes.view(-1)) # [ N*L ] 
    ppl = torch.exp(torch.mean(ppl.view(N, L), dim=-1)) # N  
    _, word_list = torch.sort(ppl)
    word_list = [all_substitutes[i] for i in word_list]
    final_words = []
    for word in word_list:
        tokens = [tokenizer._convert_id_to_token(int(i)) for i in word]
        text = tokenizer.convert_tokens_to_string(tokens)
        final_words.append(text)
    return final_words

def attack(feature, tgt_model, mlm_model, tokenizer, args, k, batch_size, cos_mat=None, w2i={}, i2w={}, use_bpe=1, threshold_pred_score=0.3):

    #source_to_feature
    source_tokens = tokenizer.tokenize(feature.source)[:args.max_source_length-2]
    source_tokens =[tokenizer.cls_token]+source_tokens+[tokenizer.sep_token]
    source_ids =  tokenizer.convert_tokens_to_ids(source_tokens) 
    source_mask = [1] * (len(source_tokens))
    padding_length = args.max_source_length - len(source_ids)
    source_ids+=[tokenizer.pad_token_id]*padding_length
    source_mask+=[0]*padding_length
    
    #target_for_attack
    target_tokens = tokenizer.tokenize("None")
    target_tokens = [tokenizer.cls_token]+target_tokens+[tokenizer.sep_token] # ['<s>', '▁No', 'ne', '</s>']
    target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
    target_mask = [1] *len(target_ids)
    padding_length = args.max_target_length - len(target_ids)
    target_ids+=[tokenizer.pad_token_id]*padding_length
    target_mask+=[0]*padding_length   

    source_ids = torch.tensor(source_ids).unsqueeze(0).to(args.device)
    source_mask = torch.tensor(source_mask).unsqueeze(0).to(args.device)
    target_ids = torch.tensor(target_ids).unsqueeze(0).to(args.device)
    target_mask = torch.tensor(target_mask).unsqueeze(0).to(args.device)
    
    preds = tgt_model(source_ids=source_ids,source_mask=source_mask, args = args)
    
    beam_pred=[]
    for pred in preds:
        t=pred[0].cpu().numpy() #beam 중에서 top-1
        t=list(t)
        if 0 in t:
            t=t[:t.index(0)]
        text = tokenizer.decode(t,clean_up_tokenization_spaces=False)
        beam_pred.append(text)
    original_bleu_score = round(attack_bleu(feature.target, beam_pred),2)
    feature.original_bleu = original_bleu_score
    # if int(orig_label) != feature.label:
    #     feature.success = 3
    #     return feature
    # MLM-process
    words, sub_words, keys = _tokenize(feature.source, tokenizer)
    sub_words = [tokenizer.cls_token]+sub_words[:512 - 2]+[tokenizer.sep_token]  
    # 왜 sub_words 기준으로 attack 을 찾는거지? : tokenize하면 subword로 찢어지니까 대체값을 tokenizer기준으로 찾는것
    input_ids_ = torch.tensor([tokenizer.convert_tokens_to_ids(sub_words)]).to(args.device)
    word_predictions = mlm_model(input_ids_)[0].squeeze() # seq-len(sub) vocab #25, 250002
    word_pred_scores_all, word_predictions = torch.topk(word_predictions, k, -1)  # seq-len k

    word_predictions = word_predictions[1:len(sub_words) + 1, :] # 본인 빼고
    word_pred_scores_all = word_pred_scores_all[1:len(sub_words) + 1, :]

    important_scores = get_important_scores(words, feature.target, tgt_model, original_bleu_score, 
                                            tokenizer, batch_size, args)
    feature.query += int(len(words))
    list_of_index = sorted(enumerate(important_scores), key=lambda x: x[1], reverse=False) #많이 감소할수록 좋은 것
    # print(list_of_index)
    final_words = copy.deepcopy(words)

    for top_index in list_of_index:
        if feature.change > int(0.4 * (len(words))): # 단어가 40% 바뀌면
            feature.success = 1  # exceed
            return feature

        tgt_word = words[top_index[0]]
        if tgt_word in filter_words:
            continue
        # if keys[top_index[0]][0] > max_seq_length - 2:
        #     continue

        substitutes = word_predictions[keys[top_index[0]][0]:keys[top_index[0]][1]]  # L, k
        word_pred_scores = word_pred_scores_all[keys[top_index[0]][0]:keys[top_index[0]][1]]

        substitutes = get_substitues(substitutes, tokenizer, mlm_model, use_bpe, word_pred_scores, threshold_pred_score, args)

        most_gap = 0.0
        candidate = None

        for substitute_ in substitutes:
            substitute = substitute_

            if substitute == tgt_word:
                continue  # filter out original word
            if '##' in substitute:
                continue  # filter out sub-word

            if substitute in filter_words:
                continue
            if substitute in w2i and tgt_word in w2i:
                if cos_mat[w2i[substitute]][w2i[tgt_word]] < 0.4:
                    continue
            temp_replace = final_words
            temp_replace[top_index[0]] = substitute
            
            temp_text = " ".join(temp_replace)

            #source_to_feature
            temp_tokens = tokenizer.tokenize(temp_text)[:args.max_source_length-2]
            temp_tokens =[tokenizer.cls_token]+temp_tokens+[tokenizer.sep_token]
            temp_ids =  tokenizer.convert_tokens_to_ids(temp_tokens) 
            temp_mask = [1] * (len(temp_tokens))
            padding_length = args.max_source_length - len(temp_ids)
            temp_ids+=[tokenizer.pad_token_id]*padding_length
            temp_mask+=[0]*padding_length

            temp_ids = torch.tensor(temp_ids).unsqueeze(0).to(args.device)
            temp_mask = torch.tensor(temp_mask).unsqueeze(0).to(args.device)
            
            temp_preds = tgt_model(source_ids=source_ids,source_mask=source_mask, args = args)
            feature.query += 1
            temp_beam_pred=[]
            for pred in temp_preds:
                t=pred[0].cpu().numpy() #beam 중에서 top-1
                t=list(t)
                if 0 in t:
                    t=t[:t.index(0)]
                text = tokenizer.decode(t,clean_up_tokenization_spaces=False)
                temp_beam_pred.append(text)
            temp_bleu_score = round(attack_bleu(feature.target, temp_beam_pred),2)
            print(tgt_word , substitute, temp_bleu_score)
            if temp_bleu_score < args.success_bleu:
                feature.change += 1
                final_words[top_index[0]] = substitute
                feature.changes.append([keys[top_index[0]][0], substitute, tgt_word])
                feature.final_adverse = temp_text
                feature.success = 4 #제대로 attack
                feature.model_pred = temp_beam_pred[0]
                feature.attack_bleu = temp_bleu_score
                return feature
            else:
                if original_bleu_score > temp_bleu_score: #감소한다면 gap 개선
                    gap = original_bleu_score - temp_bleu_score
                    if gap > most_gap:
                        most_gap = gap
                        feature.most_gap = most_gap
                        candidate = substitute
                        feature.model_pred = temp_beam_pred[0]
                        feature.attack_bleu = temp_bleu_score
        if most_gap > 0:
            feature.change += 1
            feature.changes.append([keys[top_index[0]][0], candidate, tgt_word])
            final_words[top_index[0]] = candidate

    feature.final_adverse = " ".join(final_words)
    feature.success = 2
    return feature


def dump_features(features, output_json):
    outputs = []
    for feature in features:
        outputs.append({'idx' : feature.idx,
                        'source': feature.source,
                        'gold_target': feature.target,
                        'success': feature.success,
                        'change': feature.change,
                        'num_word': len(feature.source.split(' ')),
                        'query': feature.query,
                        'changes': feature.changes,
                        'adv': feature.final_adverse,
                        'original_bleu': feature.original_bleu,
                        'attack_bleu': feature.attack_bleu,
                        'most_gap' : feature.most_gap,
                        'model_pred' : feature.model_pred,
                        'time': feature.time,
                        })                       
    json.dump(outputs, open(output_json, 'w'), indent=2)
    import pickle
    with open('features.pkl','wb') as f:
        pickle.dump(outputs,f)
    print('finished dump')

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters  
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type: e.g. roberta")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model: e.g. roberta-base" )
    parser.add_argument("--tokenizer_name", default="", required=True,
                        help="Pretrained tokenizer name or path if not the same as model_name")    
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--load_model_path", default=None, type=str, 
                        help="Path to trained model: Should contain the .bin files" )    
    ## Other parameters
    parser.add_argument("--train_filename", default=None, type=str,
                        help="The train filenames (source and target files).")
    parser.add_argument("--dev_filename", default=None, type=str,
                        help="The dev filename. (source and target files).")
    parser.add_argument("--test_filename", default=None, type=str,
                        help="The test filename. (source and target files).")  
    
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")

    parser.add_argument("--max_source_length", default=64, type=int,
                        help="The maximum total source sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_target_length", default=32, type=int,
                        help="The maximum total target sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--using_pretrain_model", action='store_true',
                        help="Initialize Transformer encoder with pre-trained model") 
    
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available") 
    
    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--beam_size", default=1, type=int,
                        help="beam size for beam search")    
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--eval_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--train_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")   
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

      ### for attack ###
    parser.add_argument("--use_sim_mat", type=int, default=1)
    parser.add_argument("--start", type=int, default = 0)
    parser.add_argument("--end", type=int, default = 1000)
    parser.add_argument("--k", default = 48, type=int)
    parser.add_argument("--threshold_pred_score", default = 0, type=float)
    parser.add_argument("--use_bpe", type=int,default = 1 ) #원래는 default : 1 (즉 bertattack bpe 1)
    parser.add_argument("--success_bleu", type=int,default = 70) #원래는 default : 1 (즉 bertattack bpe 1)
    args = parser.parse_args()
    # print arguments
    args = parser.parse_args()
    logger.info(args)

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda:3" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 1
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda:3", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1))
    args.device = device
    # Set seed
    set_seed(args)
    # make dir if output_dir not exist
    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)
        
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type] #roberta
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name,do_lower_case=args.do_lower_case)
    
    #budild model
    if args.using_pretrain_model: #학습할 떄는 XLM-Roberta 들고와서 더 fine-tuning시키는 것
        encoder = model_class.from_pretrained(args.model_name_or_path,config=config)    
    else:    
        encoder = model_class(config)  # pretrained안된 encoder 가져옴
    decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads) #768차원짜리 Transformer 기반 decoder layer (#_attention_heads:12)
    decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
    model=Seq2Seq(encoder=encoder,decoder=decoder,config=config,
                  beam_size=args.beam_size,max_length=args.max_target_length,
                  sos_id=tokenizer.cls_token_id,eos_id=tokenizer.sep_token_id)
    
    if args.load_model_path is not None:
        logger.info("reload model from {}".format(args.load_model_path))
        model.load_state_dict(torch.load(args.load_model_path, map_location=args.device))
        # model.load_state_dict(torch.load(args.load_model_path, map_location=args.device)) #seq2seq 모델 이미 학습시킨것 가져옴(evaluation)
        
    model.to(args.device)
    
    mlm_model = RobertaForMaskedLM.from_pretrained(args.tokenizer_name) #roberta-base 에서 tokenizer xlm_roberta로 변경
    mlm_model.to(args.device)
    print('loading sim-embed')
    if args.use_sim_mat == 1:
        cos_mat, w2i, i2w = get_sim_embed('/home/rlagywns0213/counter_fit/counter-fitted-vectors.txt', '/home/rlagywns0213/counter_fit/cos_sim_counter_fitting.npy')
    else:
        cos_mat, w2i, i2w = None, {}, {}

    print('finish get-sim-embed')

    features_output = []
    
    print("Attacking")

    eval_examples = read_examples_json()
    with torch.no_grad():
        for feature in eval_examples[args.start:10]:
            id, source, gold = feature.idx, feature.source, feature.target
            feat = Feature(id, source, gold)
            print('number {} '.format(id), end='')
            start_time = time.time()
            feat = attack(feat, model, mlm_model, tokenizer, args, args.k, batch_size=32,cos_mat=cos_mat, w2i=w2i, i2w=i2w, use_bpe=args.use_bpe,threshold_pred_score=args.threshold_pred_score)
            feat.time = round(time.time() - start_time,2)
            print( "changes : ", feat.changes, "change : ", feat.change, "query : ", feat.query, "success : ", feat.success, "time : ", feat.time)
            if feat.success > 2:
                print('success \n', end='')
            else:
                print('failed \n ', end='')
            features_output.append(feat)
    dump_features(features_output, f'attack_result/1123_bleu100_{args.success_bleu}_bpe_{args.use_bpe}.json')
    print('end')                        
    
                
if __name__ == "__main__":
    main()