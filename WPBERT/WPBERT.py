from __future__ import absolute_import, division, print_function, unicode_literals
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from CharBERT.modeling.modeling_bert import BertLayerNorm, CharBertEncoder, BertEncoder, BertPooler, BertPreTrainedModel, BertPredictionHeadTransform, BertOnlyMLMHead

class CharBertEmbeddings(nn.Module):
    def __init__(self, config, is_roberta=False):

        char_embedding_size = int(config.hidden_size/2)
        print(char_embedding_size)
        char_vocab_size = 1025
        padding_idx = 1024

        super(CharBertEmbeddings, self).__init__()
        self.config = config
        self.char_embeddings = nn.Embedding(char_vocab_size, char_embedding_size, padding_idx=padding_idx)
                
        self.rnn_layer = nn.GRU(input_size= int(config.hidden_size/2),\
            hidden_size=int(config.hidden_size/4), batch_first=True, bidirectional=True)

    def forward(self, char_input_ids, start_ids, end_ids):
        input_shape = char_input_ids.size()
        #print(f"shape of char_input_ids in CharBertEmbeddings: {list(input_shape)}")
        assert len(input_shape) == 2
        
        batch_size, char_maxlen = input_shape[0], input_shape[1]
        # print(f"batch_size: {batch_size} char_maxlen: {char_maxlen}")
        char_input_ids_reshape = torch.reshape(char_input_ids, (batch_size, char_maxlen))
        char_embeddings = self.char_embeddings(char_input_ids_reshape)

        #print(char_embeddings[-1])

        #print(f"char_embeddings shape: {list(char_embeddings.size())}")
        self.rnn_layer.flatten_parameters()
        all_hiddens, last_hidden = self.rnn_layer(char_embeddings.float())
        #print(f"all_hiddens shape: {list(all_hiddens.size())}")
        #char_rnn_repr = torch.transpose(all_hiddens, 0, 1)
        #print(f"char_rnn_repr shape: {list(char_rnn_repr.size())}")
         
        start_one_hot = nn.functional.one_hot(start_ids, num_classes=char_maxlen)
        #print(f"start_ont_hot shape: {list(start_one_hot.size())}")
        end_one_hot   = nn.functional.one_hot(end_ids, num_classes=char_maxlen)
        #print(f"end_ont_hot shape: {list(end_one_hot.size())}")
        start_hidden  = torch.matmul(start_one_hot.float(), all_hiddens)
        #print(f"start_hidden shape: {list(start_hidden.size())}")
        end_hidden    = torch.matmul(end_one_hot.float(), all_hiddens)
        #print(f"end_hidden shape: {list(end_hidden.size())}")
        char_embeddings_repr = torch.cat([start_hidden, end_hidden], dim=-1)
        #print(f"char_embeddings_repr shape: {list(char_embeddings_repr.size())}")
        return char_embeddings_repr

class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=1027)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        #self.char_embeddings = CharBertEmbeddings(config)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, token_type_ids=None, position_ids=None, inputs_embeds=None):
        #print(f'shape info in BertEmbeddings layer: input_ids {input_ids.size()}')
      
        
        input_shape = inputs_embeds.size()
        seq_length = input_shape[1]

        #print(f'input_shape {input_shape} seq_length {seq_length}')

        device = inputs_embeds.device
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        inputs_embeds = self.word_embeddings(inputs_embeds)

        #print(inputs_embeds[-1])

        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings # + char_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class WPBertModel(BertPreTrainedModel):
    def __init__(self, config, is_roberta=False):
        super(WPBertModel, self).__init__(config)
        self.config = config        
        
        self.embeddings = BertEmbeddings(config)
        #self.encoder = BertEncoder(config)

        self.char_embeddings = CharBertEmbeddings(config, is_roberta=is_roberta)
        #self.char_encoder = CharBertEncoder(config)

        self.encoder = CharBertEncoder(config, is_roberta=is_roberta)

        self.pooler = BertPooler(config)

        self.init_weights()
        

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self, char_input_ids = None, start_ids = None, end_ids = None, attention_mask=None, \
        token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, encoder_hidden_states=None,\
        encoder_attention_mask=None, masked_lm = False):
        """ Forward pass on the Model.

        The model can behave as an encoder (with only self-attention) as well
        as a decoder, in which case a layer of cross-attention is added between
        the self-attention layers, following the architecture described in `Attention is all you need`_ by Ashish Vaswani,
        Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

        To behave as an decoder the model needs to be initialized with the
        `is_decoder` argument of the configuration set to `True`; an
        `encoder_hidden_states` is expected as an input to the forward pass.

        .. _`Attention is all you need`:
            https://arxiv.org/abs/1706.03762

        """
        #print(f'shape info in CharBertModel: input_ids {input_ids.size()}')
        #print(f'shape info in CharBertModel: char_input_ids {char_input_ids.size()}')


        #input_shape = inputs_embeds.size()[:-1]
        input_shape = inputs_embeds.size()

        device = inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if self.config.is_decoder:
                batch_size, seq_length = input_shape
                seq_ids = torch.arange(seq_length, device=device)
                causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
                causal_mask = causal_mask.to(torch.long)  # not converting to long will cause errors with pytorch version < 1.3
                extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError("Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(input_shape, attention_mask.shape))

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=inputs_embeds.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)

            if encoder_attention_mask.dim() == 3:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
            elif encoder_attention_mask.dim() == 2:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
            else:
                raise ValueError("Wrong shape for encoder_hidden_shape (shape {}) or encoder_attention_mask (shape {})".format(encoder_hidden_shape,
                                                                                                                               encoder_attention_mask.shape))

            encoder_extended_attention_mask = encoder_extended_attention_mask.to(dtype=inputs_embeds.dtype)  # fp16 compatibility
            encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -10000.0
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(dtype=inputs_embeds.dtype)  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        
       

        embedding_output = self.embeddings(position_ids=position_ids,\
            token_type_ids=token_type_ids, inputs_embeds=inputs_embeds)
        #print(f'input shape for bert_encoder: embedding_output {embedding_output.size()}')
        #print(f'extended_attention_mask: {extended_attention_mask.size()}') 
        #print(f'head_mask: {head_mask.size()} encoder_hidden_states: {encoder_hidden_states.size()}')
        #print(f'encoder_attention_mask: {encoder_extended_attention_mask.size()}')
        #encoder_outputs = self.encoder(embedding_output,
        #                               attention_mask=extended_attention_mask,
        #                               head_mask=head_mask,
        #                               encoder_hidden_states=encoder_hidden_states,
        #                               encoder_attention_mask=encoder_extended_attention_mask)



        
        char_embeddings = self.char_embeddings(char_input_ids, start_ids, end_ids)
        char_encoder_outputs = self.encoder(char_embeddings,
                                       embedding_output,
                                       attention_mask=extended_attention_mask,
                                       head_mask=head_mask,
                                       encoder_hidden_states=encoder_hidden_states,
                                       encoder_attention_mask=encoder_extended_attention_mask)
        
        sequence_output, char_sequence_output = char_encoder_outputs[0], char_encoder_outputs[1]
        pooled_output = self.pooler(sequence_output)
        char_pooled_output = self.pooler(char_sequence_output)

        #print(char_encoder_outputs[2])
        
        
        
        outputs = (sequence_output, pooled_output, char_sequence_output, char_pooled_output) + char_encoder_outputs[2:]  # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, char_sequence_output, char_pooled_output + (hidden_states), (attentions)




class WPBertForMaskedLM(BertPreTrainedModel):
    r"""
        **masked_lm_labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Labels for computing the masked language modeling loss.
            Indices should be in ``[-1, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-1`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``
        **lm_labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Labels for computing the left-to-right language modeling loss (next word prediction).
            Indices should be in ``[-1, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-1`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **masked_lm_loss**: (`optional`, returned when ``masked_lm_labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Masked language modeling loss.
        **ltr_lm_loss**: (`optional`, returned when ``lm_labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Next token prediction loss.
        **prediction_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, config.vocab_size)``
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForMaskedLM.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, masked_lm_labels=input_ids)
        loss, prediction_scores = outputs[:2]

    """
    def __init__(self, config):
        super(WPBertForMaskedLM, self).__init__(config)

        self.bert = WPBertModel(config)
        self.cls = BertOnlyMLMHead(config)
        self.init_weights()
        
        #self.mlm_outputs = nn.Linear(config.hidden_size * 2, config.hidden_size)

       

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def forward(self, inputs_embeds = None, char_input_ids = None, start_ids = None, end_ids = None, masked_lm_labels = None, attention_mask=None, token_type_ids=None,\
               position_ids=None, head_mask=None, masked_lm = True, \
               encoder_hidden_states=None, encoder_attention_mask=None):
        
        outputs = self.bert(inputs_embeds = inputs_embeds, char_input_ids = char_input_ids, start_ids = start_ids, end_ids = end_ids, 
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            encoder_hidden_states=encoder_hidden_states,
                            encoder_attention_mask=encoder_attention_mask,
                            masked_lm = masked_lm)
           
        
        sequence_repr = outputs[0]
        char_repr = outputs[2]


        prediction_input = torch.cat([sequence_repr, char_repr], dim=-1)
     
        #print(sequence_repr)

        #print(f'shape info in CharBertForMaskLM: input_ids {input_ids.size()}')
        #print(f'shape info in CharBertForMaskLM: char_input_ids {char_input_ids.size()}')
        #print(f'shape info in CharBertForMaskLM: sequence_repr {sequence_repr.size()}')
        #print(f'shape info in CharBertForMaskLM: char_sequence_repr {char_sequence_repr.size()}')
        #sequence_concat = torch.cat((sequence_repr, char_sequence_repr), dim=-1)
        #sequence_output = self.mlm_outputs(sequence_concat)
        #sequence_output = sequence_repr + char_sequence_repr

        #sequence_output = outputs[0]
        prediction_scores = self.cls(prediction_input)
        
        #print(prediction_scores)
        
        outputs = (prediction_scores,) + outputs[0:]  # Add hidden states and attention if they are here

        # Although this may seem awkward, BertForMaskedLM supports two scenarios:
        # 1. If a tensor that contains the indices of masked labels is provided,
        #    the cross-entropy is the MLM cross-entropy that measures the likelihood
        #    of predictions for masked words.
        # 2. If `lm_labels` is provided we are in a causal scenario where we
        #    try to predict the next token for each input in the decoder.
       
        if masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)  # -1 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            outputs = (masked_lm_loss,) + outputs

        #print(outputs[6].shape)
        

        return outputs  # (masked_lm_loss), prediction_scores, sequence_output, pooled_output, char_sequence_output, char_poolehd_output + (hidden_states), (attentions)
