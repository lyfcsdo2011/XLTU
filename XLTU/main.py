"""
main.py applies XLTime framework on mBERT, XLMRbase, or XLMRlarge backbone and 
performs temporal expression extraction (TEE) on the target language.

Running main.py gives 'w/ type' TEE results (as presented in the upper part of Table 8
of our paper). To get the 'w/o type' TEE results (presented in Table 3 and the lower 
part of Table 8), one needs to run map_results.py
"""
from __future__ import absolute_import, division, print_function
import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
#from pytorch_transformers import AdamW, WarmupLinearSchedule
from transformers import AdamW, get_linear_schedule_with_warmup
from torch import nn
from torch.utils.data import DataLoader
from model.XLTU import XLTU
from utils.train_utils import add_args, evaluate_model
from utils.data_utils import SLProcessor, BCProcessor, MultiTaskDataset, MultiTaskBatchSampler, create_dataset, convert_examples_to_features, findout_max_seq_length
from tqdm import tqdm
from tqdm import trange
import pickle

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser = add_args(parser)

    args = parser.parse_args()

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps


    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError(
            "At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError(
            "Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    

    # for the sequence labeling (primary) task
    processor_sl = SLProcessor()
    label_list = processor_sl.get_labels()
    num_labels = len(label_list) + 1  # add one for the IGNORE label
    # getting training samples
    train_examples_sl = processor_sl.get_train_examples(args.data_dir_sl)
    # getting validation samples
    val_examples_sl = processor_sl.get_dev_examples(args.data_dir_sl)
    # getting test samples
    test_examples_sl = processor_sl.get_test_examples(args.data_dir_sl)

    # for the binary classification (secondary) task
    processor_bc = BCProcessor()
    # getting training samples
    train_examples_bc = processor_bc.get_train_examples(args.data_dir_bc)

    num_train_optimization_steps = int(
        len(train_examples_sl) + len(train_examples_bc) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs


    # preparing model configs
    hidden_size = 768 if args.model_size == 'base' else 1024

    device = 'cuda' if (torch.cuda.is_available() and not args.no_cuda) else 'cpu'
    logger.info('   Device: %s', device)

    # creating model
    model_type = 'xlm-roberta-base' if args.model_size == 'base' else 'xlm-roberta-large'
    model = XLTU(model_type=model_type, n_labels=num_labels, hidden_size=hidden_size,
                      dropout_p=args.dropout, device=device)

    
    # if args.max_seq_length is unspecified, then use the length of the longest sequence after encoding in the dataset
    if args.max_seq_length == 0:
        # find out the length of the longest sequence after encoding the datasets
        max_seq_length_sl = findout_max_seq_length(train_examples_sl + val_examples_sl + test_examples_sl, model.encode_word)
        max_seq_length_bc = findout_max_seq_length(train_examples_bc, model.encode_word)
        max_seq_length = max(max_seq_length_sl, max_seq_length_bc)
        logger.info("  Max seq length after tokenizing the dataset is %d", max_seq_length)
    else:
        max_seq_length = args.max_seq_length
        logger.info("  Using the specified max_seq_length %d", max_seq_length)


    special_token_ids = model.get_special_token_ids()
    model.to(device)
    no_decay = ['bias', 'final_layer_norm.weight']
    
    params = list(model.named_parameters())

    optimizer_grouped_parameters = [
        {'params': [p for n, p in params if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in params if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    warmup_steps = int(args.warmup_proportion * num_train_optimization_steps)
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate, eps=args.adam_epsilon)
    #scheduler = WarmupLinearSchedule(
    #    optimizer, warmup_steps=warmup_steps, t_total=num_train_optimization_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_train_optimization_steps)

    # freeze model if necessary
    if args.freeze_model:
        logger.info("Freezing XLM-R model...")
        for n, p in model.named_parameters():
            if 'xlmr' in n and p.requires_grad:
                p.requires_grad = False

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.fp16_opt_level)

    global_step = 0
    tr_loss = 0

    if args.do_train:

        output_log_file = os.path.join(args.output_dir, "log.txt")

        # create dataset, sampler, and dataloader for multitask training 
        # (train on primary task, i.e., source languages sequence labeling and 
        # secondary task, i.e., machine translated sequence binary classificaion)
        train_features_sl = convert_examples_to_features(
            train_examples_sl, max_seq_length, model.encode_word, special_token_ids, label_list)
        train_data_sl = create_dataset(train_features_sl)
        train_features_bc = convert_examples_to_features(
            train_examples_bc, max_seq_length, model.encode_word, special_token_ids, None)
        train_data_bc = create_dataset(train_features_bc)
        multi_task_train_dataset = MultiTaskDataset([train_data_sl, train_data_bc])
        multi_task_batch_sampler = MultiTaskBatchSampler([train_data_sl, train_data_bc], args.train_batch_size)
        # for baselines
        # multi_task_train_dataset = MultiTaskDataset([train_data_sl])
        # multi_task_batch_sampler = MultiTaskBatchSampler([train_data_sl], args.train_batch_size)
        multi_task_train_data = DataLoader(multi_task_train_dataset, batch_sampler=multi_task_batch_sampler)

        
        # create dataset for validation (validate on target language sequence labeling)
        val_features_sl = convert_examples_to_features(
            val_examples_sl, max_seq_length, model.encode_word, special_token_ids, label_list)
        val_data_sl = create_dataset(val_features_sl)


        # create dataset for evaluation (evaluate on target language sequence labeling)
        eval_examples = test_examples_sl
        eval_features = convert_examples_to_features(
            eval_examples, max_seq_length, model.encode_word, special_token_ids, label_list)
        eval_data = create_dataset(eval_features)
        
        logger.info("***** Running training *****")
        logger.info("  Num examples for the sequence labeling task = %d", len(train_examples_sl))
        logger.info("  Num examples for the binary classification task = %d", len(train_examples_bc))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        
        best_val_f1 = 0.0 # tracks the average of the vlidation f1 scores of the two tasks

        for e in tqdm(range(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0

            tbar = tqdm(multi_task_train_data, desc="Iteration")
            
            model.train()
            for step, batch in enumerate(tbar):
                batch = tuple(t.to(device) for t in batch)
                input_ids, label_ids, l_mask, tasks, = batch
                # print("--------------------------")
                # print(type(tasks[0].item()))
                # break
                
                loss, logits, pred, ground_truth = model(input_ids, label_ids, l_mask, tasks[0].item())
                """
                if e < 5 and (step%200 == 0):
                    print('step: ', str(step))
                    #print(loss)
                    print(logits)
                    print(pred)
                    print(ground_truth)
                    #print()
                """
                
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                
                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        amp.master_params(optimizer), args.max_grad_norm)
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm)

                tr_loss += loss.item()
                tbar.set_description('Loss = %.4f' %(tr_loss / (step+1)))
                '''
                if step%50 == 0:
                    print('epoch: ', str(e))
                    print('step: ', str(step))
                    print("loss: ", tr_loss / (step+1))
                    print()
                    with open(output_log_file, "a") as writer:
                        writer.write('epoch: ' + str(e) + '\n')
                        writer.write('step: ' + str(step) + '\n')
                        writer.write("loss: " + str(tr_loss / (step+1)) + '\n\n')
                '''
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1
            
            logger.info("\nTesting on validation datasets...")
            f1_sl, report_sl, y_true_sl, y_pred_sl = evaluate_model(model, val_data_sl, label_list, args.eval_batch_size, device)

            if f1_sl > best_val_f1:
                best_val_f1 = f1_sl
                logger.info("\nFound better f1=%.4f on the validation dataset. Saving model\n" %(f1_sl))
                logger.info("Validate on the sequence labeling task:\n")
                logger.info("\n%s\n" %(report_sl))

                with open(output_log_file, "a") as writer:
                    writer.write("\nFound better f1=%.4f on the validation datasets. Saving model\n" %(f1_sl))
                    writer.write("Validate on the sequence labeling task:\n")
                    writer.write("%s\n" %(report_sl))
                
                torch.save(model.state_dict(), open(os.path.join(args.output_dir, 'model.pt'), 'wb'))
                # save report
                output_valid_file = os.path.join(args.output_dir, "valid_results.txt")
                with open(output_valid_file, "a") as writer:
                    logger.info("***** Writing valid_results to file *****")
                    writer.write("-------------Found better f1 after epoch %d--------------\n" %(e))
                    writer.write("Validate on the sequence labeling task:\n")
                    writer.write("%s\n" %(report_sl))
                    logger.info("Done.")
                # save ground truth of the sequence labeling task
                output_sl_true_file = os.path.join(args.output_dir, "valid_y_true_sl")
                with open(output_sl_true_file, "wb") as fp:   #Pickling
                    pickle.dump(y_true_sl, fp)
                # save prediction of the sequence labeling task
                output_sl_pred_file = os.path.join(args.output_dir, "valid_y_pred_sl")
                with open(output_sl_pred_file, "wb") as fp:   #Pickling
                    pickle.dump(y_pred_sl, fp)

            else :
                logger.info("\nNo better F1 score: %.4f\n" %(f1_sl))
                logger.info("Validate on the sequence labeling task:\n")
                logger.info("\n%s\n" %(report_sl))
                with open(output_log_file, "a") as writer:
                    writer.write("\nNo better avg F1 score: %.4f\n" %(f1_sl))
                    writer.write("Validate on the sequence labeling task:\n")
                    writer.write("%s\n" %(report_sl))
            
            # run evaluation after each epoch
            if args.do_eval:
                    
                logger.info("***** Running evaluation *****")
                logger.info("  Num examples = %d", len(eval_examples))
                logger.info("  Batch size = %d", args.eval_batch_size)
                
                f1_score, report, y_true, y_pred = evaluate_model(model, eval_data, label_list, args.eval_batch_size, device)
                
                logger.info("\n%s", report)
                # save report
                with open(output_log_file, "a") as writer:
                    writer.write("\nEvaluation:\n")
                    writer.write("%s\n" %(report))
            
                    
    # load a saved model
    state_dict = torch.load(open(os.path.join(args.output_dir, 'model.pt'), 'rb'))
    model.load_state_dict(state_dict)
    logger.info("Loaded saved model")

    model.to(device)

    if args.do_eval:
        
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        #print(eval_examples)
       
        f1_score, report, y_true, y_pred = evaluate_model(model, eval_data, label_list, args.eval_batch_size, device)
       
        logger.info("\n%s", report)
        # save report
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Writing eval_results to file *****")
            writer.write(report)
            logger.info("Done.")
        # save ground truth
        output_true_file = os.path.join(args.output_dir, "eval_y_true")
        with open(output_true_file, "wb") as fp:   #Pickling
            pickle.dump(y_true, fp)
        # save prediction
        output_pred_file = os.path.join(args.output_dir, "eval_y_pred")
        with open(output_pred_file, "wb") as fp:   #Pickling
            pickle.dump(y_pred, fp)
        # delete model
        #os.remove(os.path.join(args.output_dir, 'model.pt'))

if __name__ == "__main__":
    main()
