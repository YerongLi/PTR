import logging
import torch
import numpy as np
from torch.utils.data import Dataset
import json
from tqdm import tqdm
from arguments import get_args

class DictDataset(Dataset):
    """A dataset of tensors that uses a dictionary for key-value mappings"""

    def __init__(self, **tensors):
        tensors.values()
        assert all(next(iter(tensors.values())).size(0) == tensor.size(0) for tensor in tensors.values())
        self.tensors = tensors

    def __getitem__(self, index):
        return {key: tensor[index] for key, tensor in self.tensors.items()}

    def __len__(self):
        return next(iter(self.tensors.values())).size(0)

    def cuda(self):
        for key in self.tensors:
            self.tensors[key] = self.tensors[key].cuda()

    def cpu(self):
        for key in self.tensors:
            self.tensors[key] = self.tensors[key].cpu()


class REPromptDataset(DictDataset):

    def __init__(self, path = None, name = None, rel2id = None, tokenizer = None, temps = None, features = None):

        with open(rel2id, "r") as f:
            self.rel2id = json.loads(f.read())
        if not 'NA' in self.rel2id:
            self.NA_NUM = self.rel2id['no_relation']
        else:  
            self.NA_NUM = self.rel2id['NA']


        self.num_class = len(self.rel2id)
        self.temps = temps
        self.get_labels(tokenizer)

        if features is None:
            self.args = get_args()
            with open(path+"/" + name, "r") as f:
                features = []
                for line in f.readlines():
                    line = line.rstrip()
                    if len(line) > 0:
                        features.append(eval(line))            
            features = self.list2tensor(features, tokenizer)

        super().__init__(**features)
    
    def get_labels(self, tokenizer):
        total = {}
        self.temp_ids = {}

        for name in self.temps:
            last = 0
            self.temp_ids[name] = {}
            self.temp_ids[name]['label_ids'] = []
            self.temp_ids[name]['mask_ids'] = []

            for index, temp in enumerate(self.temps[name]['temp']):
                _temp = temp.copy()
                # logging.info(_temp)
                # INFO:root:['the', '<mask>']
                # INFO:root:['<mask>', '<mask>', '<mask>']
                
                # logging.info('tempslabels.index')
                # logging.info(self.temps[name]['labels'])
                # INFO:root:[('person',), ('was', 'charged', 'with'), ('event',)]
                

                _labels = self.temps[name]['labels'][index]
                _labels_index = []

                for i in range(len(_temp)):
                    if _temp[i] == tokenizer.mask_token:
                        _temp[i] = _labels[len(_labels_index)]
                        _labels_index.append(i)

                original = tokenizer.encode(" ".join(temp), add_special_tokens=False)
                final =  tokenizer.encode(" ".join(_temp), add_special_tokens=False)
                # logging.info('original and final')
                # logging.info(original)
                # logging.info(final)

                # INFO:root:original and final
                # INFO:root:[50264, 50264, 50264]
                # INFO:root:[7325, 1340, 19]
                # INFO:root:original and final
                # INFO:root:[627, 50264]
                # INFO:root:[627, 515]
                # INFO:root:original and final
                # INFO:root:[627, 50264]
                # INFO:root:[627, 621]
                
                assert len(original) == len(final)
                self.temp_ids[name]['label_ids'] += [final[pos] for pos in _labels_index]
                # logging.info("self.temp_ids[name]['label_ids']")
                # logging.info(tokenizer.decode(self.temp_ids[name]['label_ids']))
                # INFO:root: person
                # INFO:root: personwas charged with
                # INFO:root: personwas charged with event

                # INFO:root: person
                # INFO:root: personwas died on
                # INFO:root: personwas died on date

                # logging.info('labels_index')
                # logging.info(_labels_index)
                # INFO:root:[1]
                # INFO:root:labels_index
                # INFO:root:[1]
                # INFO:root:labels_index
                # INFO:root:[0, 1, 2]
                # INFO:root:labels_index
                # INFO:root:[1]
                # INFO:root:labels_index
                # INFO:root:[1]
                # INFO:root:labels_index
                # INFO:root:[0, 1, 2]
                # INFO:root:labels_index
                # INFO:root:[1]

                for pos in _labels_index:
                    if not last in total:
                        total[last] = {}
                    # logging.info('last')
                    # logging.info(last)
                    # logging.info(final[pos])
                    total[last][final[pos]] = 1
                    last+=1
                self.temp_ids[name]['mask_ids'].append(original)
        # logging.info(total)
        # INFO:root:{0: {621: 1, 1651: 1, 10014: 1}, 1: {7325: 1, 18: 1, 354: 1}, 2: {1340: 1, 962: 1, 4790: 1, 2421: 1, 5221: 1, 5407: 1, 1270: 1, 25385: 1, 1207: 1, 919: 1, 8850: 1, 334: 1, 3200: 1, 21771: 1, 17117: 1, 4095: 1, 920: 1, 29853: 1, 26241: 1, 998: 1, 1046: 1, 21821: 1, 2034: 1}, 3: {19: 1, 15: 1, 11: 1, 9: 1, 30: 1, 16: 1, 21: 1, 34: 1, 7: 1}, 4: {515: 1, 1248: 1, 247: 1, 621: 1, 343: 1, 194: 1, 1270: 1, 1651: 1, 6825: 1, 346: 1, 46471: 1, 10014: 1}}
        # logging.info(len(total))
        # for key in range(len(total)):
        #     l = list(total[key].keys())
        #     logging.info(tokenizer.decode(l))
        # INFO:root:5
        # INFO:root: person organization entity
        # INFO:root:was'sis
        # INFO:root: charged died founded born invested relative title dissolved living member employer school employee sibling spouse parent child alias nationality website age irrelevant located
        # INFO:root: with on in of by is was has to
        # INFO:root: event date country person city state title organization religion number url entity


        # print (total)
        self.set = [(list)((sorted)(set(total[i]))) for i in range(len(total))]
        print ("=================================")
        # for i in self.set:
        #     print (i)
        print ("=================================")
        logging.info('self.set after the sorting')
        logging.info(self.set)

        for name in self.temp_ids:
            for j in range(len(self.temp_ids[name]['label_ids'])):
                self.temp_ids[name]['label_ids'][j] = self.set[j].index(
                    self.temp_ids[name]['label_ids'][j])
                
        # logging.info('selftemp_ids')
        # logging.info(self.temp_ids['per:country_of_death'])
        # INFO:root:{'label_ids': [0, 2, 3, 2, 1], 'mask_ids': [[627, 50264], [50264, 50264, 50264], [627, 50264]]}
        
        # logging.info('selftemp_ids')
        # logging.info(self.temp_ids['org:political/religious_affiliation']) 
        # INFO:root:selftemp_ids
        # INFO:root:{'label_ids': [1, 2, 1, 1, 9], 'mask_ids': [[627, 50264], [50264, 50264, 50264], [627, 50264]]}


        # logging.info('selftemp_ids')
        # for j, item in enumerate(self.temp_ids['per:country_of_death']['label_ids']):
        #     logging.info(tokenizer.decode(self.set[j][self.temp_ids['per:country_of_death']['label_ids'][j]]))
        # INFO:root:selftemp_ids
        # INFO:root: person
        # INFO:root:was
        # INFO:root: died
        # INFO:root: in
        # INFO:root: country

        self.prompt_id_2_label = torch.zeros(len(self.temp_ids), len(self.set)).long()
        
        for name in self.temp_ids:
            for j in range(len(self.prompt_id_2_label[self.rel2id[name]])):
                self.prompt_id_2_label[self.rel2id[name]][j] = self.temp_ids[name]['label_ids'][j]
        # logging.info('self.prompt_id_2_label')
        # logging.info(self.prompt_id_2_label)
        # INFO:root:self.prompt_id_2_label
        # INFO:root:tensor([[ 2,  1, 19,  0, 10],
        #         [ 1,  0,  1,  6,  8],
        #         [ 0,  0, 18,  6,  5],
        #         [ 0,  0, 17,  6,  5],
        #         [ 1,  2,  9,  2,  1],
        #         [ 0,  2,  3,  2,  1],
        #         [ 0,  0, 12,  6,  5],
        #         [ 0,  2,  6,  2,  0],
        #         [ 1,  0, 16,  6,  5],
        #         [ 1,  2, 20,  2,  6],
        #         [ 1,  0, 16,  8,  3],
        #         [ 0,  2,  3,  2,  0],
        #         [ 0,  0, 21,  6,  1],
        #         [ 0,  0,  2,  6,  5],
        #         [ 1,  2,  1,  1,  9],

        # logging.info('selfset')
        # logging.info(self.set)
        # INFO:root:selfset
        # INFO:root:[[621, 1651, 10014], [18, 354, 7325], [334, 919, 920, 962, 998, 1046, 1207, 1270, 1340, 2034, 2421, 3200, 4095, 4790, 5221, 5407, 8850, 17117, 21771, 21821, 25385, 26241, 29853], [7, 9, 11, 15, 16, 19, 21, 30, 34], [194, 247, 343, 346, 515, 621, 1248, 1270, 1651, 6825, 10014, 46471]]


        # logging.info('self.prompt_id_2_label length')
        # logging.info(len(self.prompt_id_2_label))
        # for entry in self.prompt_id_2_label:
        #     logging.info(entry.numpy())
        #     logging.info(tokenizer.decode(entry.numpy()))
        self.prompt_id_2_label = self.prompt_id_2_label.long().cuda()
        # logging.info('self.set last statement')
        # logging.info(self.set)
        self.prompt_label_idx = [
            torch.Tensor(i).long() for i in self.set
        ]

        # logging.info('selfprompt')
        # logging.info(self.prompt_label_idx)
        # INFO:root:selfprompt
        # INFO:root:[tensor([  621,  1651, 10014]), tensor([  18,  354, 7325]), tensor([  334,   919,   920,   962,   998,  1046,  1207,  1270,  1340,  2034,
        #          2421,  3200,  4095,  4790,  5221,  5407,  8850, 17117, 21771, 21821,
        #         25385, 26241, 29853]), tensor([ 7,  9, 11, 15, 16, 19, 21, 30, 34]), tensor([  194,   247,   343,   346,   515,   621,  1248,  1270,  1651,  6825,
        #         10014, 46471])]

    def save(self, path = None, name = None):
        path = path + "/" + name  + "/"
        np.save(path+"input_ids", self.tensors['input_ids'].numpy())
        np.save(path+"token_type_ids", self.tensors['token_type_ids'].numpy())
        np.save(path+"attention_mask", self.tensors['attention_mask'].numpy())
        np.save(path+"labels", self.tensors['labels'].numpy())
        np.save(path+"mlm_labels", self.tensors['mlm_labels'].numpy())
        np.save(path+"input_flags", self.tensors['input_flags'].numpy())
        # np.save(path+"prompt_label_idx_0", self.prompt_label_idx[0].numpy())
        # np.save(path+"prompt_label_idx_1", self.prompt_label_idx[1].numpy())
        # np.save(path+"prompt_label_idx_2", self.prompt_label_idx[2].numpy())

    @classmethod
    def load(cls, path = None, name = None, rel2id = None, temps = None, tokenizer = None):
        path = path + "/" + name  + "/"
        features = {}
        features['input_ids'] = torch.Tensor(np.load(path+"input_ids.npy")).long()
        features['token_type_ids'] = torch.Tensor(np.load(path+"token_type_ids.npy")).long()
        features['attention_mask'] = torch.Tensor(np.load(path+"attention_mask.npy")).long()
        features['labels'] = torch.Tensor(np.load(path+"labels.npy")).long()
        features['input_flags'] = torch.Tensor(np.load(path+"input_flags.npy")).long()
        features['mlm_labels'] = torch.Tensor(np.load(path+"mlm_labels.npy")).long()
        res = cls(rel2id = rel2id, features = features, temps = temps, tokenizer = tokenizer)
        # res.prompt_label_idx = [torch.Tensor(np.load(path+"prompt_label_idx_0.npy")).long(),
        #     torch.Tensor(np.load(path+"prompt_label_idx_1.npy")).long(),
        #     torch.Tensor(np.load(path+"prompt_label_idx_2.npy")).long()
        # ]
        return res

    def list2tensor(self, data, tokenizer):
        res = {}
        res['input_ids'] = []
        res['token_type_ids'] = []
        res['attention_mask'] = []
        res['input_flags'] = []
        res['mlm_labels'] = []
        res['labels'] = []

        for index, i in enumerate(tqdm(data)):

            input_ids, token_type_ids, input_flags = self.tokenize(i, tokenizer)
            attention_mask = [1] * len(input_ids)
            padding_length = self.args.max_seq_length - len(input_ids)

            if padding_length > 0:
                input_ids = input_ids + ([tokenizer.pad_token_id] * padding_length)
                attention_mask = attention_mask + ([0] * padding_length)
                token_type_ids = token_type_ids + ([0] * padding_length)
                input_flags = input_flags + ([0] * padding_length)
            
            assert len(input_ids) == self.args.max_seq_length
            assert len(attention_mask) == self.args.max_seq_length
            assert len(token_type_ids) == self.args.max_seq_length
            assert len(input_flags) == self.args.max_seq_length

            label = self.rel2id[i['relation']]
            res['input_ids'].append(np.array(input_ids))
            res['token_type_ids'].append(np.array(token_type_ids))
            res['attention_mask'].append(np.array(attention_mask))
            res['input_flags'].append(np.array(input_flags))
            res['labels'].append(np.array(label))
            mask_pos = np.where(res['input_ids'][-1] == tokenizer.mask_token_id)[0]
            mlm_labels = np.ones(self.args.max_seq_length) * (-1)
            mlm_labels[mask_pos] = 1
            res['mlm_labels'].append(mlm_labels)
        for key in res:
            res[key] = np.array(res[key])
            res[key] = torch.Tensor(res[key]).long()
        return res

    def tokenize(self, item, tokenizer):

        sentence = item['token']
        pos_head = item['h']
        pos_tail = item['t']
        rel_name = item['relation']

        temp = self.temps[rel_name]

        sentence = " ".join(sentence)
        sentence = tokenizer.encode(sentence, add_special_tokens=False)
        e1 = tokenizer.encode(" ".join(['was', pos_head['name']]), add_special_tokens=False)[1:]
        e2 = tokenizer.encode(" ".join(['was', pos_tail['name']]), add_special_tokens=False)[1:]

        # prompt =  [tokenizer.unk_token_id, tokenizer.unk_token_id] + \
        prompt = self.temp_ids[rel_name]['mask_ids'][0] + e1 + \
                 self.temp_ids[rel_name]['mask_ids'][1] + \
                 self.temp_ids[rel_name]['mask_ids'][2] + e2 
        #  + \
        #  [tokenizer.unk_token_id, tokenizer.unk_token_id]

        flags = []
        last = 0
        for i in prompt:
            # if i == tokenizer.unk_token_id:
            #     last+=1
            #     flags.append(last)
            # else:
            flags.append(0)
        
        tokens = sentence + prompt
        flags = [0 for i in range(len(sentence))] + flags
        # tokens = prompt + sentence
        # flags =  flags + [0 for i in range(len(sentence))]        
        
        tokens = self.truncate(tokens, 
                               max_length = self.args.max_seq_length - tokenizer.num_special_tokens_to_add(False))
        flags = self.truncate(flags, 
                               max_length = self.args.max_seq_length - tokenizer.num_special_tokens_to_add(False))

        input_ids = tokenizer.build_inputs_with_special_tokens(tokens)
        token_type_ids = tokenizer.create_token_type_ids_from_sequences(tokens)
        input_flags = tokenizer.build_inputs_with_special_tokens(flags)
        input_flags[0] = 0
        input_flags[-1] = 0
        assert len(input_ids) == len(input_flags)
        assert len(input_ids) == len(token_type_ids)
        return input_ids, token_type_ids, input_flags

    def truncate(self, seq, max_length):
        if len(seq) <= max_length:
            return seq
        else:
            print ("=========")
            return seq[len(seq) - max_length:]