"""
map_results.py maps the 'w/ type' TEE results (presented in the upper part of Table 8
of our paper) resulted from running main.py to the 'w/o type' TEE results (presented 
in Table 3 and the lower part of Table 8).
"""
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score
import os
import pickle
import math
import argparse

"""
Evaluate the prediction. If ignore_type equals to True, evaluate w/o tyoe,
otherwise, evaluate w/ tyoe.
"""
def evaluate(list_eval_true, list_eval_pred, ignore_type = False):
    if ignore_type:
        for i in range(len(list_eval_true)):
            for j in range(len(list_eval_true[i])):
                if list_eval_true[i][j] in ['B-DATE', 'B-TIME', 'B-DURATION', 'B-SET']:
                    list_eval_true[i][j] = 'B-TYPE'
                elif list_eval_true[i][j] in ['I-DATE', 'I-TIME', 'I-DURATION', 'I-SET']:
                    list_eval_true[i][j] = 'I-TYPE'

                if list_eval_pred[i][j] in ['B-DATE', 'B-TIME', 'B-DURATION', 'B-SET']:
                    list_eval_pred[i][j] = 'B-TYPE'
                elif list_eval_pred[i][j] in ['I-DATE', 'I-TIME', 'I-DURATION', 'I-SET']:
                    list_eval_pred[i][j] = 'I-TYPE'

    report = classification_report(list_eval_true, list_eval_pred, digits=4)
    f1 = f1_score(list_eval_true, list_eval_pred, average='macro')
    #print(f1)
    #print(report)
    return report

"""
Load and return the ground truth labels and the predicted labels.
list_eval_true: a list of the ground truth labels.
list_eval_pred: a list of the predicted labels.
"""
def load_results(data_path):
    with open(data_path + 'eval_y_true', 'rb') as f:
        list_eval_true = pickle.load(f)
    with open(data_path + 'eval_y_pred', 'rb') as f:
        list_eval_pred = pickle.load(f)
    return list_eval_true, list_eval_pred

"""
Iterate through all the result folders in d, for each result folder, evaluate both
w/ and w/o type.
d: a directory that contains result folders
"""
def eval_w_wo_type_dir(d):
    dir_paths = [os.path.join(d, o) + '/' for o in os.listdir(d) if os.path.isdir(os.path.join(d,o))]
    for each in dir_paths:
        print(each)
        list_eval_true, list_eval_pred = load_results(each)
        eval_w = evaluate(list_eval_true, list_eval_pred, ignore_type = False)
        eval_wo = evaluate(list_eval_true, list_eval_pred, ignore_type = True)
        with open(each + 'eval_results_w_wo_type.txt', 'w') as sf:
            sf.write('with type: \n')
            sf.write(eval_w)
            sf.write('\n')
            sf.write('without type: \n')
            sf.write(eval_wo)
    return

"""
Evaluate the model prediction both w/ and w/o type.
data_path: path to the result folder.
"""
def eval_w_wo_type_one(data_path):
    print(data_path)
    list_eval_true, list_eval_pred = load_results(data_path)
    eval_w = evaluate(list_eval_true, list_eval_pred, ignore_type = False)
    eval_wo = evaluate(list_eval_true, list_eval_pred, ignore_type = True)
    with open(data_path + 'eval_results_w_wo_type.txt', 'w') as sf:
        sf.write('with type: \n')
        sf.write(eval_w)
        sf.write('\n')
        sf.write('without type: \n')
        sf.write(eval_wo)
    return

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type = str, default = None, help = 'path to the result folder')
    parser.add_argument('--dir_path', type = str, default = None, help = 'a directory that contains result folders')
    return parser.parse_args()


def main():
    args = parse_args()
    #data_path = './XLTime-mBERT_EN_2_FR_results/'
    if args.data_path is not None:
        eval_w_wo_type_one(args.data_path)
    if args.dir_path is not None:
        eval_w_wo_type_dir(args.dir_path)

if __name__ == "__main__":
    main()
