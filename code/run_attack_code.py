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

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import random
import json
import copy
import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter
from tqdm import tqdm, trange

from transformers import (WEIGHTS_NAME, get_linear_schedule_with_warmup, AdamW,
                          RobertaConfig,
                          RobertaModel,
                          RobertaForMaskedLM,
                          RobertaTokenizer)

from models import Model
from utils import acc_and_f1, TextDataset, Attack_Dataset
import multiprocessing
import pickle
import time

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

cpu_cont = multiprocessing.cpu_count()

logger = logging.getLogger(__name__)

MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer)}

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self, code_tokens, code_ids, nl_tokens, nl_ids, label):
        self.code_tokens = code_tokens
        self.code_ids = code_ids
        self.nl_tokens = nl_tokens
        self.nl_ids = nl_ids
        self.label = label
        # self.idx = idx

class Feature(object):
    def __init__(self, idx, nl,code, label):
        self.idx = idx
        self.label = label
        self.seq = nl
        self.final_adverse = nl
        self.code = code
        self.query = 0
        self.change = 0
        self.success = 0
        self.sim = 0.0
        self.time = 0.0
        self.changes = []

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

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

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

def get_important_scores(words,code_tokens, code_ids, tgt_model, orig_prob, orig_label, tokenizer, batch_size, max_seq_length, args):
    masked_words = _get_masked(words)
    texts = [' '.join(words) for words in masked_words]  # list of text of masked words
    all_examples = []

    for text in texts:
        nl_tokens = tokenizer.tokenize(text)[:max_seq_length-2]
        nl_tokens = [tokenizer.cls_token]+nl_tokens+[tokenizer.sep_token]
        nl_ids = tokenizer.convert_tokens_to_ids(nl_tokens)
        padding_length = max_seq_length - len(nl_ids)
        nl_ids += [tokenizer.pad_token_id]*padding_length
        all_examples.append(InputFeatures(code_tokens, code_ids.squeeze().tolist(), nl_tokens, nl_ids, int(orig_label.squeeze())))
       
    attack_dataset = Attack_Dataset(all_examples)
    eval_sampler = SequentialSampler(attack_dataset) 
    eval_dataloader = DataLoader(attack_dataset, sampler=eval_sampler, batch_size=batch_size, num_workers=0, pin_memory=True)
    leave_1_probs = []
    for batch in eval_dataloader:
            code_inputs = batch[0].to(args.device)
            nl_inputs = batch[1].to(args.device)
            labels = batch[2].to(args.device)
            with torch.no_grad():
                logits, loss, new_label = tgt_model(code_inputs, nl_inputs, labels)
                leave_1_probs.append(logits)
    leave_1_probs = torch.cat(leave_1_probs, dim=0)
    if orig_label == 1:
        important_scores = (orig_prob - leave_1_probs).data.cpu().numpy() 
    else:
        important_scores = (leave_1_probs - orig_prob).data.cpu().numpy()  # orig_prob이 0에 가까움 

    return important_scores

def get_substitues(substitutes, tokenizer, mlm_model, use_bpe, substitutes_score=None, threshold=3.0):
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
            words = get_bpe_substitues(substitutes, tokenizer, mlm_model)
        else:
            return words
    #
    # print(words)
    return words


def get_bpe_substitues(substitutes, tokenizer, mlm_model):
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
    all_substitutes = all_substitutes[:24].to('cuda')
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

def attack(feature, tgt_model, mlm_model, tokenizer, args, k, batch_size, max_length=512, cos_mat=None, w2i={}, i2w={}, use_bpe=1, threshold_pred_score=0.3, max_seq_length = 200):
    # MLM-process
    
    words, sub_words, keys = _tokenize(feature.seq, tokenizer)
    # original label
    code = feature.code
    code_tokens = tokenizer.tokenize(code)[:max_seq_length-2]
    code_tokens = [tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
    code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
    padding_length = max_seq_length - len(code_ids)
    code_ids += [tokenizer.pad_token_id]*padding_length

    nl = feature.seq
    nl_tokens = tokenizer.tokenize(nl)[:max_seq_length-2]
    nl_tokens = [tokenizer.cls_token]+nl_tokens+[tokenizer.sep_token]
    nl_ids = tokenizer.convert_tokens_to_ids(nl_tokens)
    padding_length = max_seq_length - len(nl_ids)
    nl_ids += [tokenizer.pad_token_id]*padding_length

    code_ids = torch.tensor(code_ids).unsqueeze(0).to(args.device)
    nl_ids = torch.tensor(nl_ids).unsqueeze(0).to(args.device)
    label = torch.tensor(feature.label).unsqueeze(0).to(args.device)
    logits, loss, orig_label = tgt_model(code_ids, nl_ids, label) 
    current_prob = logits.squeeze() #logits 값 항상 0~1 사이

    if int(orig_label) != feature.label:
        feature.success = 3
        return feature

    sub_words = [tokenizer.cls_token]+sub_words[:max_seq_length - 2]+[tokenizer.sep_token]  # 왜 sub_words 기준으로 attack 을 찾는거지?
    input_ids_ = torch.tensor([tokenizer.convert_tokens_to_ids(sub_words)])
    word_predictions = mlm_model(input_ids_.to(args.device))[0].squeeze()  # seq-len(sub) vocab # 27, 50265
    word_pred_scores_all, word_predictions = torch.topk(word_predictions, k, -1)  # seq-len k

    word_predictions = word_predictions[1:len(sub_words) + 1, :] # 본인 빼고
    word_pred_scores_all = word_pred_scores_all[1:len(sub_words) + 1, :]

    important_scores = get_important_scores(words, code_tokens, code_ids, tgt_model, current_prob, orig_label, 
                                            tokenizer, batch_size, max_seq_length, args)
    feature.query += int(len(words))
    list_of_index = sorted(enumerate(important_scores), key=lambda x: x[1], reverse=True)
    # print(list_of_index)
    final_words = copy.deepcopy(words)

    for top_index in list_of_index:
        if feature.change > int(0.4 * (len(words))): # 단어가 40% 바뀌면
            feature.success = 1  # exceed
            return feature

        tgt_word = words[top_index[0]]
        if tgt_word in filter_words:
            continue
        if keys[top_index[0]][0] > max_seq_length - 2:
            continue


        substitutes = word_predictions[keys[top_index[0]][0]:keys[top_index[0]][1]]  # L, k
        word_pred_scores = word_pred_scores_all[keys[top_index[0]][0]:keys[top_index[0]][1]]

        substitutes = get_substitues(substitutes, tokenizer, mlm_model, use_bpe, word_pred_scores, threshold_pred_score)


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
            temp_tokens = tokenizer.tokenize(temp_text)[:max_seq_length-2]
            temp_tokens = [tokenizer.cls_token]+temp_tokens+[tokenizer.sep_token]
            temp_ids = tokenizer.convert_tokens_to_ids(temp_tokens)
            padding_length = max_seq_length - len(temp_ids)
            temp_ids += [tokenizer.pad_token_id]*padding_length

            input_ids = torch.tensor(temp_ids).unsqueeze(0).to(args.device)
            temp_prob, _, temp_label = tgt_model(code_ids, input_ids, label) 

            if int(temp_label) != int(orig_label):
                feature.change += 1
                final_words[top_index[0]] = substitute
                feature.changes.append([keys[top_index[0]][0], substitute, tgt_word])
                feature.final_adverse = temp_text
                feature.success = 4 #제대로 attack
                return feature
            else:
                gap = abs(current_prob - temp_prob.squeeze())
                if gap > most_gap:
                    most_gap = gap
                    candidate = substitute

        if most_gap > 0:
            feature.change += 1
            feature.changes.append([keys[top_index[0]][0], candidate, tgt_word])
            current_prob = current_prob - most_gap
            final_words[top_index[0]] = candidate

    feature.final_adverse = " ".join(final_words)
    feature.success = 2
    return feature
    

def evaluate(features):
    do_use = 0
    use = None
    sim_thres = 0
    # evaluate with USE

    if do_use == 1:
        cache_path = ''
        import tensorflow as tf
        import tensorflow_hub as hub
    
        class USE(object):
            def __init__(self, cache_path):
                super(USE, self).__init__()

                self.embed = hub.Module(cache_path)
                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True
                self.sess = tf.Session()
                self.build_graph()
                self.sess.run([tf.global_variables_initializer(), tf.tables_initializer()])

            def build_graph(self):
                self.sts_input1 = tf.placeholder(tf.string, shape=(None))
                self.sts_input2 = tf.placeholder(tf.string, shape=(None))

                sts_encode1 = tf.nn.l2_normalize(self.embed(self.sts_input1), axis=1)
                sts_encode2 = tf.nn.l2_normalize(self.embed(self.sts_input2), axis=1)
                self.cosine_similarities = tf.reduce_sum(tf.multiply(sts_encode1, sts_encode2), axis=1)
                clip_cosine_similarities = tf.clip_by_value(self.cosine_similarities, -1.0, 1.0)
                self.sim_scores = 1.0 - tf.acos(clip_cosine_similarities)

            def semantic_sim(self, sents1, sents2):
                sents1 = [s.lower() for s in sents1]
                sents2 = [s.lower() for s in sents2]
                scores = self.sess.run(
                    [self.sim_scores],
                    feed_dict={
                        self.sts_input1: sents1,
                        self.sts_input2: sents2,
                    })
                return scores[0]

            use = USE(cache_path)


    acc = 0
    origin_success = 0
    total = 0
    total_q = 0
    total_change = 0
    total_word = 0
    total_time = 0.0
    for feat in features:
        if feat.success > 2:

            if do_use == 1:
                sim = float(use.semantic_sim([feat.seq], [feat.final_adverse]))
                if sim < sim_thres:
                    continue
            
            acc += 1
            total_q += feat.query
            total_change += feat.change
            total_word += len(feat.seq.split(' '))
            total_time += feat.time
            if feat.success == 3: # 이미 모델 예측값이 틀린 경우 origin_success + 1
                origin_success += 1

        total += 1

    suc = float(acc / total)

    query = float(total_q / acc)
    change_rate = float(total_change / total_word)

    origin_acc = 1 - origin_success / total
    after_atk = 1 - suc

    print('# of atk {:.1f},# of success_3 {:.1f}, acc/aft-atk-acc {:.6f}/ {:.6f}, query-num {:.4f}, change-rate {:.4f}, total_time {:.2f}'.format(acc,origin_success, origin_acc, after_atk, query, change_rate, total_time))

def dump_features(features, output):
    outputs = []

    for feature in features:
        outputs.append({'idx' : feature.idx,
                        'label': feature.label,
                        'success': feature.success,
                        'change': feature.change,
                        'num_word': len(feature.seq.split(' ')),
                        'query': feature.query,
                        'changes': feature.changes,
                        'seq_a': feature.seq,
                        'adv': feature.final_adverse,
                        'code': feature.code
                        })
    output_json = output
    json.dump(outputs, open(output_json, 'w'), indent=2)
    import pickle
    with open('features.pkl','wb') as f:
        pickle.dump(outputs,f)
    print('finished dump')

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default='./data_codebert/', type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--file_name", default='python_dev_data.json', type=str,
                        help="The input training data file (a text file).")

    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--model_type", default="roberta", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--pn_weight", type=float, default=1.0,
                        help="Ratio of positive examples in the sum of bce loss")
    parser.add_argument("--encoder_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--checkpoint_path", default=None, type=str,
                        help="The checkpoint path of model to continue training.")

    parser.add_argument("--mlm", action='store_true',
                        help="Train with masked-language modeling loss instead of language modeling.")
    parser.add_argument("--mlm_probability", type=float, default=0.15,
                        help="Ratio of tokens to mask for masked language modeling loss")

    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    parser.add_argument("--max_seq_length", default=200, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")

    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument("--pred_model_dir", default=None, type=str,
                        help='model for prediction')
    parser.add_argument("--test_result_dir", default='test_results.tsv', type=str,
                        help='path to store test result')
    parser.add_argument("--prediction_file", default='predictions.txt', type=str,
                        help='path to save predictions result, note to specify task name')


    ### for attack ###
    parser.add_argument("--use_sim_mat", type=int, default=0)
    parser.add_argument("--start", type=int, default = 0)
    parser.add_argument("--end", type=int, default = 1000)
    parser.add_argument("--k", default = 48, type=int)
    parser.add_argument("--threshold_pred_score", default = 0, type=float)
    parser.add_argument("--use_bpe", type=int,default = 1 ) #원래는 default : 1 (즉 bertattack bpe 1)
    args = parser.parse_args()

    max_seq_length = args.max_seq_length


    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    # Set seed
    set_seed(args.seed)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.start_epoch = 0
    args.start_step = 0

    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    if os.path.exists(checkpoint_last) and os.listdir(checkpoint_last):
        # args.encoder_name_or_path = os.path.join(checkpoint_last, 'pytorch_model.bin')
        args.config_name = os.path.join(checkpoint_last, 'config.json')
        idx_file = os.path.join(checkpoint_last, 'idx_file.txt')
        with open(idx_file, encoding='utf-8') as idxf:
            args.start_epoch = int(idxf.readlines()[0].strip()) + 1

        step_file = os.path.join(checkpoint_last, 'step_file.txt')
        if os.path.exists(step_file):
            with open(step_file, encoding='utf-8') as stepf:
                args.start_step = int(stepf.readlines()[0].strip())

        logger.info("reload model from {}, resume from {} epoch".format(checkpoint_last, args.start_epoch))

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.encoder_name_or_path,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    config.num_labels = 2
    tokenizer_tgt = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.encoder_name_or_path,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
    if max_seq_length <= 0:
        max_seq_length = tokenizer_tgt.max_len_single_sentence  # Our input block size will be the max possible for the model
    max_seq_length = min(max_seq_length, tokenizer_tgt.max_len_single_sentence)
    if args.encoder_name_or_path:
        model = model_class.from_pretrained(args.encoder_name_or_path,
                                            from_tf=bool('.ckpt' in args.encoder_name_or_path),
                                            config=config,
                                            cache_dir=args.cache_dir if args.cache_dir else None)
    else:
        model = model_class(config)

    print("Attacking")
    
    ### model load ###
    tgt_model = Model(model, config, tokenizer_tgt, args)
    mlm_model = RobertaForMaskedLM.from_pretrained('roberta-base') 
    logger.info("Training/evaluation parameters %s", args)
   
    tgt_model.load_state_dict(torch.load(os.path.join(args.output_dir, 'pytorch_model.bin')))
    tokenizer_tgt = tokenizer_tgt.from_pretrained(args.output_dir)
    tgt_model.to(args.device)
    mlm_model.to(args.device)
    print('loading sim-embed')
    if args.use_sim_mat == 1:
        cos_mat, w2i, i2w = get_sim_embed('counter_fit/counter-fitted-vectors.txt', 'counter_fit/cos_sim_counter_fitting.npy')
    else:
        cos_mat, w2i, i2w = None, {}, {}

    print('finish get-sim-embed')
    features_output = []

    except_list=[]
    ### data load ###
    attack_data_path = os.path.join(args.data_dir, args.file_name)
    with open(attack_data_path, 'r') as f:
            data = json.load(f)

    with torch.no_grad():
        for index, feature in enumerate(data[args.start:20]):
            if index in [229, 326]:
                continue
            else:
                try:
                    nl, code, label = feature['doc'], feature['code'], feature['label']
                    feat = Feature(feature['idx'], nl,code,label)
                    print('number {} '.format(feature['idx']), end='')
                    start_time = time.time()
                    feat = attack(feat, tgt_model, mlm_model, tokenizer_tgt, args, args.k, batch_size=32, max_length=512,
                                cos_mat=cos_mat, w2i=w2i, i2w=i2w, use_bpe=args.use_bpe,threshold_pred_score=args.threshold_pred_score, max_seq_length=max_seq_length)
                    feat.time = round(time.time() - start_time,2)
                    
                    print( "changes : ", feat.changes, "change : ", feat.change, "query : ", feat.query, "success : ", feat.success, "time : ", feat.time)
                    if feat.success > 2:
                        print('success \n', end='')
                    else:
                        print('failed \n ', end='')
                    features_output.append(feat)
                except:
                    except_list.append(index)
        print("error : ", except_list)
    evaluate(features_output)

    if args.use_bpe == 1:
        dump_features(features_output, 'data_attack/python_dev(bpe)_20.json')
    else:
        dump_features(features_output, 'data_attack/python_dev_20.json')
        
    # with torch.no_grad():
        # for index, feature in enumerate()
    # results = attack(args, tgt_model, tokenizer)

if __name__ == "__main__":
    main()