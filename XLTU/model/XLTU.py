from transformers import XLMRobertaModel, XLMRobertaTokenizer, XLMRobertaForMaskedLM
import torch
import torch.nn as nn
import torch.nn.functional as F

from TorchCRF import CRF

class XLTU(nn.Module):
    """
    Apply the proposed XLTime framework on the XLMR backbone.
    """
    def __init__(self, model_type, n_labels, hidden_size, dropout_p, label_ignore_idx=0,
                head_init_range=0.04, device='cuda'):
        super().__init__()
        
        self.hidden_size = hidden_size
        # for the sequence labeling task (task = 0)
        self.n_labels = n_labels
        self.linear_1 = nn.Linear(hidden_size, hidden_size)
        self.linear_encoder_out = nn.Linear(hidden_size, hidden_size)
        self.classification_head_1 = nn.Linear(hidden_size, n_labels)
        self.label_ignore_idx = label_ignore_idx
        self.crf = CRF(num_labels=n_labels)

        # self.lstm = nn.LSTM(
        #     hidden_size,
        #     self.hidden_size // 2,
        #     num_layers=1,
        #     bidirectional=True,
        #     batch_first=True,
        # )
        
        # for the binary classification task (task = 1)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        self.classification_head_2 = nn.Linear(hidden_size, 2)

        # load the pretrained model
        if model_type == 'xlm-roberta-base':
            path = "/home/lyf/LLM/huggingface_model/xlm-roberta-base"
        else:
            path = "/home/lyf/LLM/huggingface_model/xlm-roberta-large"
        self.model = XLMRobertaModel.from_pretrained(path)
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(path)
        self.mlm = XLMRobertaForMaskedLM.from_pretrained(path).get_output_embeddings()
        
        # CNN
        self.filter_sizes = (2,3,4)
        self.num_filters = 256
        self.convs = nn.ModuleList(
            [nn.Conv2d(in_channels=1, out_channels=self.num_filters, 
                       kernel_size=(k, 768 if model_type == 'xlm-roberta-base' else 1024)) for k in self.filter_sizes]
        )
        self.fc = nn.Linear(self.num_filters * len(self.filter_sizes), 2)

        self.dropout = nn.Dropout(dropout_p)
        self.device=device
        
        self.bc_fc = nn.Linear(4, 2)

        # initializing classification head
        self.classification_head_1.weight.data.normal_(mean=0.0, std=head_init_range)
        self.classification_head_2.weight.data.normal_(mean=0.0, std=head_init_range)
        self.fc.weight.data.normal_(mean=0.0, std=head_init_range)
        self.bc_fc.weight.data.normal_(mean=0.0, std=head_init_range)
    
    def conv_and_pool(self, x, conv):
        x = conv(x)
        x = F.relu(x)
        x = x.squeeze(3)
        size = x.size(2)
        x = F.max_pool1d(x, size)
        x = x.squeeze(2)
        return x

    def forward(self, inputs_ids, labels, labels_mask, task = 0):
        """
        Computes a forward pass through the model.
        Args:
            inputs_ids: tensor of size (bsz, max_seq_len).
            labels: tensor of size (bsz, max_seq_len).
            labels_mask: indicate where loss gradients should be propagated and where labels should be ignored.
            task: 0 indicates the sequence labeling task, while 1 indicates the binary classification task. 
        """
        if task == 0:
            
            # print(inputs_ids.size())
            # last_hidden_state (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size)) 
            # – Sequence of hidden-states at the output of the last layer of the model.
            transformer_out = self.model(inputs_ids)[0]
            # linear_out = F.relu(self.linear_encoder_out(transformer_out))
            # linear_out = self.dropout(linear_out)
            # encoder_out = self.model_encoder(linear_out)
            #print(transformer_out)
            out_1 = F.relu(self.linear_1(transformer_out))
            out_1 = self.dropout(out_1)
            logits = self.classification_head_1(out_1)  # [batch_size, seq_len, n_labels]
            
            

        
            if labels is not None: # for training
                # nn.CrossEntropyLoss(input, target). ignore_index: target value to be ignored. 
                # This keeps the active parts of the loss
                loss_fct = nn.CrossEntropyLoss(ignore_index=self.label_ignore_idx) 
                loss1 = -self.crf(logits, labels, labels_mask == 1).sum(dim=-1)
                # Only keep active parts of the loss
                if labels_mask is not None:
                    active_loss = labels_mask.view(-1) == 1
                    active_logits = logits.view(-1, self.n_labels)[active_loss]
                    active_labels = labels.view(-1)[active_loss]
                    loss2 = loss_fct(active_logits, active_labels)
                    #print("Preds = ", active_logits.argmax(dim=-1))
                    #print("Labels = ", active_labels)
                else:
                    loss2 = loss_fct(
                        logits.view(-1, self.n_labels), labels.view(-1))
                a = 0.2
                loss = a * loss1 + (1 - a) * loss2
                # loss = loss2
                return loss, logits, active_logits.argmax(dim=-1), active_labels # loss, logits, predictions, labels
            else: # for evaluation
                return logits

        elif task == 1:
            # pooler_output (torch.FloatTensor of shape (batch_size, hidden_size)) 
            
            transformer_out = self.model(inputs_ids)
            out_2 = F.relu(self.linear_2(transformer_out[1]))
            out_2 = self.dropout(out_2)
            logits_1 = self.classification_head_2(out_2)  # [batch_size, n_labels]
            
            # print(logits.size())
            
            encoder_out = transformer_out[0]
            out = encoder_out.unsqueeze(1)
            out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
            out = self.dropout(out)
            logits_2 = self.fc(out) # [batch_size, n_labels]
            
            # # concat 2 results
            # # logits = torch.cat((logits_1, logits_2), 1)
            # # relu and dropout concat
            # logits = self.dropout(torch.cat((F.relu(logits_1), F.relu(logits_2)), 1))
            # logits = self.bc_fc(logits)
            
            # only use CNN results
            logits = logits_2
            # logits = logits_1
            

            if labels is not None: # for training
                loss_fct = nn.CrossEntropyLoss()
                logits = logits.view(-1, 2)
                loss = loss_fct(logits, labels.view(-1))
                return loss, logits, logits.argmax(dim=-1), labels.view(-1) # loss, logits, predictions, labels
            else: # for evaluation
                return logits

        # elif task == 2:
        #     transformer_out = self.model(inputs_ids)[0]
        #     prediction_scores = self.mlm(transformer_out)
        #     loss_fct = nn.CrossEntropyLoss()
            
        #     masked_lm_loss = loss_fct(prediction_scores.view(-1, self.tokenizer.vocab_size), inputs_ids.view(-1))
        #     return masked_lm_loss, prediction_scores, prediction_scores.argmax(dim=-1), inputs_ids.view(-1)
        
        else:
            print('Invalid task. task has to be 0 or 1')

    def encode_word(self, s):
        """
        takes a string and returns a list of token ids
        """
        tensor_ids = self.tokenizer.encode(s)
        # remove <s> and </s> ids
        return tensor_ids[1:-1]
    
    
    
    # XLM-R:bos_token='<s>':0, eos_token='</s>':2, sep_token='</s>', cls_token='<s>', unk_token='<unk>', pad_token='<pad>:1 
    def get_special_token_ids(self):
        return {"bos_token_id":0, "eos_token_id":2, "pad_token_id":1}
    
