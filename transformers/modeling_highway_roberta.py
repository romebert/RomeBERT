from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss

from .modeling_roberta import RobertaEmbeddings
from .modeling_highway_bert import BertModel, BertPreTrainedModel, BertEncoder, entropy, HighwayException, KDLoss, TAKDLoss, GroupKDLoss, BidirectionLoss
from .configuration_roberta import RobertaConfig

ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'roberta-base': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-pytorch_model.bin",
    'roberta-large': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-pytorch_model.bin",
    'roberta-large-mnli': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-mnli-pytorch_model.bin",
    'distilroberta-base': "https://s3.amazonaws.com/models.huggingface.co/bert/distilroberta-base-pytorch_model.bin",
    'roberta-base-openai-detector': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-openai-detector-pytorch_model.bin",
    'roberta-large-openai-detector': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-openai-detector-pytorch_model.bin",
}


class RobertaModel(BertModel):
    r"""
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **last_hidden_state**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, hidden_size)``
            Sequence of hidden-states at the output of the last layer of the model.
        **pooler_output**: ``torch.FloatTensor`` of shape ``(batch_size, hidden_size)``
            Last layer hidden-state of the first token of the sequence (classification token)
            further processed by a Linear layer and a Tanh activation function. The Linear
            layer weights are trained from the next sentence prediction (classification)
            objective during Bert pretraining. This output is usually *not* a good summary
            of the semantic content of the input, you're often better with averaging or pooling
            the sequence of hidden-states for the whole input sequence.
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaModel.from_pretrained('roberta-base')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

    """
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config, slow=False):
        super(RobertaModel, self).__init__(config)

        self.embeddings = RobertaEmbeddings(config)
        self.encoder = BertEncoder(config, slow=slow)
        
        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value


class RobertaForSequenceClassification(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForSequenceClassification.from_pretrained('roberta-base')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]

    """
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(RobertaForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.num_layers = config.num_hidden_layers
        self.config = config
        self.m = None
        self.moco = False

        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        
        self.init_weights()
        
    def use_slow_teacher(self):
        if not self.moco:
            self.moco = True
            self.slow_roberta = RobertaModel(self.config, slow=True)
            self.slow_classifier = nn.Linear(self.config.hidden_size, self.config.num_labels)
            self._slow_teacher_initialization()
            print('Use slow teacher to guide self-distillation.')
    
    def set_momentum_value(self, value=0.999):
        if self.m is None:
            self.m = value
            print('Setting the value of MoCo momentum to {}.'.format(self.m))
            
    def _slow_teacher_initialization(self):
        for param_q, param_k in zip(self.roberta.parameters(), self.slow_roberta.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
        
        for param_q, param_k in zip(self.classifier.parameters(), self.slow_classifier.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient  
    
    @torch.no_grad()
    def _momentum_update_slow_teacher(self):
        """
        Momentum update of the slow teacher consists of slow bert model & slow classifier
        """
        for param_q, param_k in zip(self.roberta.parameters(), self.slow_roberta.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
            
        for param_q, param_k in zip(self.classifier.parameters(), self.slow_classifier.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, labels=None, output_layer=-1, kd_loss_type='raw', gamma=0.9, beta=None, irange=None, temper=3.0):

        exit_layer = self.num_layers
        try:
            outputs = self.roberta(input_ids,
                                   attention_mask=attention_mask,
                                   token_type_ids=token_type_ids,
                                   position_ids=position_ids,
                                   head_mask=head_mask,
                                   inputs_embeds=inputs_embeds)

            pooled_output = outputs[1]

            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
            outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        except HighwayException as e:
            outputs = e.message
            exit_layer = e.exit_layer
            logits = outputs[0]
            
        if self.moco:
            # compute key features
            with torch.no_grad():  # no gradient to keys
                self._momentum_update_slow_teacher()  # update the slow teacher
                slow_outputs = self.slow_roberta(input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids,
                                position_ids=position_ids,
                                head_mask=head_mask,
                                inputs_embeds=inputs_embeds)
                # sequence_output, pooled_output, (hidden_states), (attentions)
                slow_pooled_output = slow_outputs[1]
                slow_pooled_output = self.dropout(slow_pooled_output)
                slow_logits = self.slow_classifier(slow_pooled_output)

        highway_entropy = []
        highway_logits_all = []
        if not self.training:
            original_entropy = entropy(logits)
            
        if labels is not None:
            # work with highway exits
            highway_losses = []
            for highway_exit in outputs[-1]:
                highway_logits = highway_exit[0]
                highway_logits_all.append(highway_logits)
                highway_entropy.append(highway_exit[2])

            if kd_loss_type == 'raw':
                loss_fct = KDLoss(len(outputs[-1]), gamma, temper, self.num_labels)
                if self.moco:
                    soft_labels = slow_logits.detach()
                else:
                    soft_labels = logits.detach()
                loss_kd = loss_fct(logits, highway_logits_all, labels, soft_labels)
                outputs = (loss_kd,) + outputs
            elif kd_loss_type == 'group':
                loss_fct = GroupKDLoss(len(outputs[-1]), gamma, beta, irange, temper, self.num_labels)
                if self.moco:
                    soft_labels = slow_logits.detach()
                else:
                    soft_labels = logits.detach()
                loss_kd = loss_fct(logits, highway_logits_all, labels, soft_labels)
                outputs = (loss_kd,) + outputs
            elif kd_loss_type == 'ta':
                loss_fct = TAKDLoss(len(outputs[-1]), gamma, temper, self.num_labels)
                loss_kd = loss_fct(logits, highway_logits_all, labels)
                outputs = (loss_kd,) + outputs
            elif kd_loss_type == 'bidir':
                loss_fct = BidirectionLoss(len(outputs[-1]), gamma, self.num_labels)
                if self.moco:
                    soft_labels = slow_logits.detach()
                else:
                    soft_labels = logits.detach()
                loss_kd = loss_fct(logits, highway_logits_all, labels, soft_labels)
                outputs = (loss_kd,) + outputs
            else:
                if self.num_labels == 1:
                    #  We are doing regression
                    loss_fct = MSELoss()
                    loss = loss_fct(logits.view(-1), labels.view(-1))
                else:
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                outputs = (loss,) + outputs
        
        if not self.training:
            outputs = outputs + ((original_entropy, highway_entropy), exit_layer)
            if output_layer >= 0:
                outputs = (outputs[0],) + \
                          (highway_logits_all[output_layer],) + \
                          outputs[2:]  ## use the highway of the last layer

        return outputs  # (loss), logits, (hidden_states), (attentions), (entropies), (exit_layer)
