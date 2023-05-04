from arguments import get_eval_args_parser
from collections import Counter
from data_prompt import REPromptDataset
from modeling import get_model, get_tokenizer
from optimizing import get_optimizer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from templating import get_temps
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler
from tqdm import tqdm, trange
from utils import progress_bar_log

import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

def f1_score(output, label, rel_num, na_num):
    correct_by_relation = Counter()
    guess_by_relation = Counter()
    gold_by_relation = Counter()

    for i in range(len(output)):
        guess = output[i]
        gold = label[i]

        if guess == na_num:
            guess = 0
        elif guess < na_num:
            guess += 1

        if gold == na_num:
            gold = 0
        elif gold < na_num:
            gold += 1

        if gold == 0 and guess == 0:
            continue
        if gold == 0 and guess != 0:
            guess_by_relation[guess] += 1
        if gold != 0 and guess == 0:
            gold_by_relation[gold] += 1
        if gold != 0 and guess != 0:
            guess_by_relation[guess] += 1
            gold_by_relation[gold] += 1
            if gold == guess:
                correct_by_relation[gold] += 1
    
    f1_by_relation = Counter()
    recall_by_relation = Counter()
    prec_by_relation = Counter()
    for i in range(1, rel_num):
        recall = 0
        if gold_by_relation[i] > 0:
            recall = correct_by_relation[i] / gold_by_relation[i]
        precision = 0
        if guess_by_relation[i] > 0:
            precision = correct_by_relation[i] / guess_by_relation[i]
        if recall + precision > 0 :
            f1_by_relation[i] = 2 * recall * precision / (recall + precision)
        recall_by_relation[i] = recall
        prec_by_relation[i] = precision

    micro_f1 = 0
    if sum(guess_by_relation.values()) != 0 and sum(correct_by_relation.values()) != 0:
        recall = sum(correct_by_relation.values()) / sum(gold_by_relation.values())
        prec = sum(correct_by_relation.values()) / sum(guess_by_relation.values())    
        micro_f1 = 2 * recall * prec / (recall+prec)

    return micro_f1, f1_by_relation

def evaluate(model, dataset, dataloader, output_dir='.'):
    model.eval()
    scores = []
    all_labels = []
    progress = progress_bar_log(log)
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            logits = model(**batch)
            progress.check(i, len(dataloader))
            res = []
            for i in dataset.prompt_id_2_label:
                _res = 0.0
                for j in range(len(i)):
                    _res += logits[j][:, i[j]]                
                _res = _res.detach().cpu()
                res.append(_res)
            logits = torch.stack(res, 0).transpose(1,0)
            labels = batch['labels'].detach().cpu().tolist()
            all_labels+=labels
            scores.append(logits.cpu().detach())
        scores = torch.cat(scores, 0)
        scores = scores.detach().cpu().numpy()
        all_labels = np.array(all_labels)
        np.save(output_dir+"/scores.npy", scores)
        np.save(output_dir+"/all_labels.npy", all_labels)

        pred = np.argmax(scores, axis = -1)
        mi_f1, ma_f1 = f1_score(pred, all_labels, dataset.num_class, dataset.NA_NUM)
        return mi_f1, ma_f1

args = get_eval_args_parser()

logging.basicConfig(filename=args.output_dir+'/output.log', level=logging.DEBUG)
log = logging.getLogger(__name__)

log.info(f'Logger start: {os.uname()[1]}')
tokenizer = get_tokenizer(special=[])
temps = get_temps(tokenizer)
scores = np.load(args.output_dir+"/scores.npy")
all_labels = np.load(args.output_dir+"/all_labels.npy")
log.info(scores.shape)
log.info(all_labels.shape)


test_dataset = REPromptDataset.load(
    path = args.output_dir, 
    name = "test", 
    temps = temps,
    tokenizer = tokenizer,
    rel2id = args.data_dir + "/" + "rel2id.json")
predictions = np.argmax(scores, axis=1)
N = len(test_dataset.rel2id)
# log.info(f'all_labels[:50]] {all_labels[:50]}')
# log.info(f'predictions[:50]] {predictions[:50]}')
cm = confusion_matrix(all_labels, predictions, labels=range(N))

rel2idlist = [None] * len(test_dataset.rel2id)
for rel, i in test_dataset.rel2id.items():
    rel2idlist[i] = rel

errorsummary = {}
for i in range(len(test_dataset.rel2id)):
    for j in range(len(test_dataset.rel2id)):
        if i == j: continue
        errorsummary[(i, j)] = cm[i][j]

ans = sorted(errorsummary.items(), key=lambda x:x[1], reverse=True)
# for item in ans:
    # log.info(f'{rel2idlist[item[0][0]]} -> {rel2idlist[item[0][1]]} : {item[1]}' )
mosterror = {i : {} for i in range(N)}
TOP = 10
for i, item in enumerate(ans[:TOP]):
    mosterror[item[0][0]][item[0][1]] = i
    filename = f"{args.output_dir}/{i}.txt"
    if os.path.exists(filename):  os.remove(filename)

## Get the tokenizer
tokenizer = get_tokenizer(special=[])
for i, data in tqdm(enumerate(test_dataset)):
    # dict_keys(['input_ids', 'token_type_ids', 'attention_mask', 'labels', 'input_flags', 'mlm_labels'])
    input_ids = [t for t in data['input_ids'] if t != tokenizer.pad_token_id]
    # log.info(tokenizer.decode(input_ids, skip_special_tokens=False))
    label = int(data['labels'].numpy())
    # log.info(rel2idlist[data['labels'].numpy()])
    # log.info(rel2idlist[predictions[i]])
    if predictions[i] in mosterror[label]:
        with open(f"{args.output_dir}/{mosterror[label][predictions[i]]}.txt", "a") as f:
            f.write(tokenizer.decode(input_ids, skip_special_tokens=False)+'\n')
            f.write(rel2idlist[label]+'\n')
            f.write(rel2idlist[predictions[i]]+'\n')

tokenizer = get_tokenizer(special=[])
LIMIT = 10
map_data = {i :[] for i in range(LIMIT)}
def extract_strings(input_string):
    start_index = input_string.find('the<mask>') + len('the<mask>')
    end_index = input_string.find('<mask><mask><mask>', start_index)
    if start_index == -1 or end_index == -1:
        return None
    else:
        x = input_string[start_index:end_index].strip()

    start_index = input_string.find('<mask><mask><mask>the<mask>') + len('<mask><mask><mask>the<mask>')
    end_index = input_string.find('</s>', start_index)
    if start_index == -1 or end_index == -1:
        return None
    else:
        y = input_string[start_index:end_index].strip()
    return (x,y)
for i, data in tqdm(enumerate(test_dataset)):
    # dict_keys(['input_ids', 'token_type_ids', 'attention_mask', 'labels', 'input_flags', 'mlm_labels'])
    input_ids = [t for t in data['input_ids'] if t != tokenizer.pad_token_id]
    # log.info(tokenizer.decode(input_ids, skip_special_tokens=False))
    label = int(data['labels'].numpy())
    if label in map_data:
        pair = extract_strings(tokenizer.decode(input_ids, skip_special_tokens=False))
        print(pair)
        map_data[label].append(pair)
    log.info(rel2idlist[data['labels'].numpy()])

    # log.info(rel2idlist[predictions[i]])
