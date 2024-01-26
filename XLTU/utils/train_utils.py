from torch.utils.data import SequentialSampler, DataLoader
from tqdm import tqdm
import seqeval.metrics
import sklearn.metrics
import torch
import torch.nn.functional as F
import numpy as np


def add_args(parser):
    """
    Adds arguments to the passed parser.
    """
    parser.add_argument("--data_dir_sl",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir for the sequence labeling (primary) task. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--data_dir_bc",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir for the binary classification (secondary) task. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument('--backbone', default='XLMR', type=str, required=True,
                        help="XLMR or mBERT")
    parser.add_argument("--model_size", default='base', type=str, required=True,
                        help="base or large. Note that XLMR has base and large versions, while mBERT only has base version.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument('--dropout', 
                        type=float, default=0.3,
                        help = "training dropout probability")
    parser.add_argument("--max_seq_length",
                        default=0,
                        type=int,
                        help="The maximum total input sequence length after tokenization. \n"
                            "Sequences longer than this will be truncated, and sequences shorter \n"
                            "than this will be padded. If set max_seq_length to 0, then the length \n"
                            "of the longest sequence after tokenization in the dataset will be used.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval or not.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                            "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                            "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--freeze_model', 
                        action='store_true', default=False,
                        help = "whether to freeze the XLM-R base model and train only the classification heads")
                                
    return parser


def evaluate_model(model, eval_dataset, label_list, batch_size, device):
    """
    Validate or evaluate the model using the eval_dataset provided.
    Returns:
        F1_score: Macro-average f1_score on the evaluation dataset.
        Report: detailed classification report 
    """
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=batch_size)

    model.eval() # turn off dropout

    y_true = []
    y_pred = []
    f1 = None
    report = None

    task = eval_dataset[0][3]

    if task == 0: # the sequence labeling task
          
        label_map = {i: label for i, label in enumerate(label_list, 1)}
        label_map[0] = "IGNORE"

        for input_ids, label_ids, l_mask, tasks in eval_dataloader:

            input_ids = input_ids.to(device)
            label_ids = label_ids.to(device)
            l_mask = l_mask.to(device)

            with torch.no_grad():
                logits = model(input_ids, labels=None, labels_mask=l_mask,
                                task=tasks[0])
            #print(input_ids)
            #print(logits)
            logits = torch.argmax(logits, dim=2)
            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.cpu().numpy()
            #print(logits)
            #print()
            for i, cur_label in enumerate(label_ids):
                temp_1 = []
                temp_2 = []

                for j, m in enumerate(cur_label):
                    if l_mask[i][j]:  # if it's a valid label
                        temp_1.append(label_map[m])
                        temp_2.append(label_map[logits[i][j]])

                assert len(temp_1) == len(temp_2)
                y_true.append(temp_1)
                y_pred.append(temp_2)

        #print(y_pred)
        report = seqeval.metrics.classification_report(y_true, y_pred, digits=4) # y_true is a list of lists, y_pred also is a list of lists
        f1 = seqeval.metrics.f1_score(y_true, y_pred, average='macro') # 'macro': calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account. 'micro': calculate metrics globally by counting the total true positives, false negatives and false positives.

    elif task == 1: # the binary classification task
        for input_ids, label_ids, l_mask, tasks in eval_dataloader:

            input_ids = input_ids.to(device)
            label_ids = label_ids.to(device)

            with torch.no_grad():
                logits = model(input_ids, labels=None, labels_mask=None,
                                task=tasks[0])

            #print(logits)
            #print(logits.size())
            logits = torch.argmax(logits, dim=-1)
            #print(logits)
            #print(logits.size())
            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.cpu().numpy()

            logits = np.squeeze(logits)
            label_ids = np.squeeze(label_ids)               

            y_true.extend(label_ids)
            y_pred.extend(logits)
          
        report = sklearn.metrics.classification_report(y_true, y_pred, digits=4)

        # 'macro': calculate metrics for each label, and find their unweighted mean.
        # This does not take label imbalance into account. 
        # 'micro': calculate metrics globally by counting the total true positives, 
        # false negatives and false positives.
        f1 = sklearn.metrics.f1_score(y_true, y_pred, average='macro') 

    return f1, report, y_true, y_pred
