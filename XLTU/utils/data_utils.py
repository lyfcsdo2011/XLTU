import os
import logging
import pandas as pd
import random
import torch 
from torch.utils.data import TensorDataset, Dataset, BatchSampler

class InputExample(object):
    """
    A single training/test example for the sequence labeling task or the binary classification task.
    """

    def __init__(self, guid, text_a, text_b=None, label=None, task=0):
        """
        Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
            task: 0 or 1. 0 indicates the sequence labeling task, while 1 indicates 
            the binary classification task. 
        
        e.g.:
            # sequence labeling task:
                valid-0
                CRICKET - LEICESTERSHIRE TAKE OVER AT TOP AFTER INNINGS VICTORY
                None
                ['O', 'O', 'B-ORG', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
                0

            # binary classification task:
                train-0
                They have so many different things to try today.
                None
                1
                1
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.task = task

class InputFeatures(object):
    """
    A single set of features of data.
    """
    def __init__(self, input_ids, label_id, label_mask=None, task = 0):
        self.input_ids = input_ids
        self.label_id = label_id
        self.label_mask = label_mask # only used by the sequence labeling task
        self.task = task # 0 indicates the sequence labeling task, while 1 indicates the binary classification task

class BCProcessor:
    """
    Processor for the binary classification data set.
    """
    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_file(os.path.join(data_dir, "train.tsv")), "train")
    
    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_file(os.path.join(data_dir, "valid.tsv")), "valid")
    
    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_file(os.path.join(data_dir, "test.tsv")), "test")

    def _read_file(self, filename):
        import csv
        df = pd.read_csv(filename, sep='\t', quoting=csv.QUOTE_NONE)
        sentences = df.iloc[:,1].tolist()
        labels = df.iloc[:,0].tolist()
        tasks = [self.get_task_id()] * len(labels)
        data = zip(sentences, labels, tasks)
        data = list(data)
        return data

    # set_type: string. 'train', 'valid', or 'test'.
    def _create_examples(self, data, set_type):
        examples = []

        for i, (sentence, label, task) in enumerate(data):
            guid = "%s-%s" % (set_type, i)
            text_a = sentence
            text_b = None
            label = label
            task = task
            examples.append(InputExample(
                guid=guid, text_a=text_a, text_b=text_b, label=label, task=task))
        return examples

    @staticmethod
    def get_task_id():
        return 1

class SLProcessor:
    """
    Processor for the sequence labeling data set.
    """
    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_file(os.path.join(data_dir, "train.txt")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_file(os.path.join(data_dir, "valid.txt")), "valid")

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_file(os.path.join(data_dir, "test.txt")), "test")

    def get_labels(self):
        return ["O", "B-DATE", "I-DATE", "B-TIME", "I-TIME", "B-DURATION", "I-DURATION", "B-SET", "I-SET"]

    def _read_file(self, filename):
        f = open(filename)
        data = []
        sentence = []
        label = []
        task = self.get_task_id()

        for i, line in enumerate(f, 1):
            if not line.strip() or len(line) == 0 or line.startswith('-DOCSTART') or line[0] == "\n" or line[0] == '.':
                if len(sentence) > 0:
                    data.append((sentence, label, task))
                    sentence = []
                    label = []
                continue

            splits = line.split()
            assert len(splits) >= 2, "error on line {}. Found {} splits".format(i, len(splits))
            word, tag = splits[0], splits[-1]
            assert tag in self.get_labels(), "unknown tag {} in line {}".format(tag, i)
            sentence.append(word.strip())
            label.append(tag.strip())

        if len(sentence) > 0:
            data.append((sentence, label, task))
            sentence = []
            label = []
        return data

    # set_type: string. 'train', 'valid', or 'test'.
    def _create_examples(self, data, set_type):
        examples = []
        #print(data)
        for i, (sentence, label, task) in enumerate(data):
            guid = "%s-%s" % (set_type, i)
            text_a = ' '.join(sentence)
            text_b = None
            label = label
            task = task
            examples.append(InputExample(
                guid=guid, text_a=text_a, text_b=text_b, label=label, task=task))
        return examples

    @staticmethod
    def get_task_id():
        return 0


    # set_type: string. 'train', 'valid', or 'test'.
    def _create_examples(self, data, set_type):
        examples = []

        for i, (sentence, label, task) in enumerate(data):
            guid = "%s-%s" % (set_type, i)
            text_a = sentence
            text_b = None
            label = label
            task = task
            examples.append(InputExample(
                guid=guid, text_a=text_a, text_b=text_b, label=label, task=task))
        return examples

    @staticmethod
    def get_task_id():
        return 2


# Encode the dataset to find out the length of the longest sequence
def findout_max_seq_length(examples, encode_method):
    max_seq_length = 0
    for _, example in enumerate(examples):
        textlist = example.text_a.split(' ')
        token_ids = [] # contain all the token ids in this sentence
        for _, word in enumerate(textlist):  # iterate through all the words in this sentence
            tokens = encode_method(word.strip())  # word token ids  
            token_ids.extend(tokens)  
        if len(token_ids) >= max_seq_length:
            max_seq_length = len(token_ids)
    return max_seq_length+2  # +2 to provide room for bos_token_id and eos_token_id

def convert_examples_to_features(examples, max_seq_length, encode_method, special_token_ids, label_list = None):
    """
    Converts a set of examples into language model compatible format.
    Labels are only assigned to the positions correspoinding to the first BPE token of each word.
    Other positions are labeled with 0 ("IGNORE").
    """
    task = examples[0].task

    # for sequnece labeling
    if task == 0:
        ignored_label = "IGNORE"
        label_map = {label: i for i, label in enumerate(label_list, 1)}
        label_map[ignored_label] = 0  # 0 label is to be ignored

        features = []
        for (ex_index, example) in enumerate(examples):

            textlist = example.text_a.split(' ')
            labellist = example.label
            labels = []
            label_mask = []
            token_ids = []
        
            for i, word in enumerate(textlist):  
                #print(word)
                tokens = encode_method(word.strip())  # word token ids  
                #print(tokens)
                token_ids.extend(tokens)  # all sentence token ids
                label_1 = labellist[i]
                for m in range(len(tokens)):
                    if m == 0:  # only label the first BPE token of each work
                        labels.append(label_1)
                        label_mask.append(1)
                    else:
                        labels.append(ignored_label)  # unlabeled BPE token
                        label_mask.append(0)

            logging.debug("token ids = ")
            logging.debug(token_ids)
            logging.debug("labels = ")
            logging.debug(labels)
            logging.debug("label_mask = ")
            logging.debug(label_mask)

            if len(token_ids) >= max_seq_length - 1:  # trim extra tokens
                token_ids = token_ids[0:(max_seq_length-2)]
                labels = labels[0:(max_seq_length-2)]
                label_mask = label_mask[0:(max_seq_length-2)]

            if special_token_ids["bos_token_id"] is not None:
                # adding bos_token
                token_ids.insert(0, special_token_ids["bos_token_id"])
                labels.insert(0, ignored_label)
                label_mask.insert(0, 0)

            # adding eos_token
            token_ids.append(special_token_ids["eos_token_id"])
            labels.append(ignored_label)
            label_mask.append(0)

            assert len(token_ids) == len(labels)
            assert len(label_mask) == len(labels)

            label_ids = []
            for i, _ in enumerate(token_ids):
                label_ids.append(label_map[labels[i]])

            assert len(token_ids) == len(label_ids)
            assert len(label_mask) == len(label_ids)


            while len(token_ids) < max_seq_length:
                token_ids.append(special_token_ids["pad_token_id"])  # adding pad_token
                label_ids.append(label_map[ignored_label])  # label ignore idx
                label_mask.append(0)

            assert len(token_ids) == max_seq_length
            assert len(label_ids) == max_seq_length
            assert len(label_mask) == max_seq_length

            if ex_index < 2:
                logging.info("*** Example ***")
                logging.info("guid: %s" % (example.guid))
                logging.info("input_ids: %s" %
                            " ".join([str(x) for x in token_ids]))
                logging.info("label: %s (id = %s)" % (example.label, " ".join(map(str, label_ids))))
                logging.info("label_mask: %s" %
                            " ".join([str(x) for x in label_mask]))

            features.append(
                InputFeatures(input_ids=token_ids,
                            label_id=label_ids,
                            label_mask=label_mask,
                            task=task))

    # for binary classification
    elif task == 1:
        features = []
        for (ex_index, example) in enumerate(examples):

            textlist = example.text_a.split(' ')
            label = [example.label]

            token_ids = []
            for i, word in enumerate(textlist):  
                #print(word)
                tokens = encode_method(word.strip())  # word token ids  
                #print(tokens)
                token_ids.extend(tokens)  # all sentence token ids

            logging.debug("token ids = ")
            logging.debug(token_ids)
            logging.debug("label = ")
            logging.debug(label)

            if len(token_ids) >= max_seq_length - 1:  # trim extra tokens
                token_ids = token_ids[0:(max_seq_length-2)]

            if special_token_ids["bos_token_id"] is not None:
                # adding bos_token
                token_ids.insert(0, special_token_ids["bos_token_id"])

            # adding eos_token
            token_ids.append(special_token_ids["eos_token_id"])


            while len(token_ids) < max_seq_length:
                token_ids.append(special_token_ids["pad_token_id"])  # adding pad_token

            assert len(token_ids) == max_seq_length

            if ex_index < 2:
                logging.info("*** Example ***")
                logging.info("guid: %s" % (example.guid))
                logging.info("input_ids: %s" %
                            " ".join([str(x) for x in token_ids]))
                logging.info("label: %s" % (str(example.label)))

            features.append(
                InputFeatures(input_ids=token_ids,
                            label_id=label,
                            #label_mask=None,
                            label_mask=0,
                            task=task))

    else:
         print('Invalid task. task has to be 0 or 1')

    return features

def create_dataset(features):
    """
    pack data features into TensorDataset.
    """
    all_input_ids = torch.tensor(
        [f.input_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor(
        [f.label_id for f in features], dtype=torch.long)
    all_lmask_ids = torch.tensor(
        [f.label_mask for f in features], dtype=torch.long)
    all_task = torch.tensor(
        [f.task for f in features], dtype=torch.long)

    return TensorDataset( # Each sample will be retrieved by indexing tensors along the first dimension.
        all_input_ids, all_label_ids, all_lmask_ids, all_task)

class MultiTaskDataset(Dataset):
    """
    Marge the sequence labeling dataset and the binary classification dataset into one.
    """
    def __init__(self, datasets):
        self._datasets = datasets
        task_2_dataset_dic = {}
        for dataset in datasets:
            task = dataset[0][3].item()
            task_2_dataset_dic[task] = dataset
        self._task_2_dataset_dic = task_2_dataset_dic

    def __len__(self):
        return sum(len(dataset) for dataset in self._datasets)
    
    def __getitem__(self, idx):
        task, sample_id = idx
        #print(task)
        #print(sample_id)
        return self._task_2_dataset_dic[task][sample_id]

class MultiTaskBatchSampler(BatchSampler):
    """
    Split the data of both tasks into mini-batches, randomly yeild one mini-batch at a time.
    """
    def __init__(self, datasets, batch_size):
        self._datasets = datasets
        self._batch_size = batch_size
        train_data_list = []
        for dataset in datasets:
            train_data_list.append(self._get_shuffled_index_batches(len(dataset), batch_size))
        self._train_data_list = train_data_list
    
    @staticmethod
    def _get_shuffled_index_batches(dataset_len, batch_size):
        index_batches = [list(range(i, min(i+batch_size, dataset_len))) for i in range(0, dataset_len, batch_size)]
        random.shuffle(index_batches)
        return index_batches

    def __len__(self):
        return sum(len(train_data) for train_data in self._train_data_list)
    
    def __iter__(self):
        all_iters = [iter(item) for item in self._train_data_list]
        all_indices = self._gen_task_indices(self._train_data_list)
        for local_task_idx in all_indices:
            #task = self._datasets[local_task_idx].get_task_id()
            task = self._datasets[local_task_idx][0][3].item()
            batch = next(all_iters[local_task_idx])
            yield [(task, sample_id) for sample_id in batch]
    
    @staticmethod
    def _gen_task_indices(train_data_list):
        all_indices = []
        for i in range(0, len(train_data_list)):
            all_indices += [i] * len(train_data_list[i])
        random.shuffle(all_indices)
        return all_indices