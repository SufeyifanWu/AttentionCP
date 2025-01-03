import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import (
    BertModel, 
    BertEncoder,
    BertLayer,
    BertAttention,
    BertSelfAttention,
    BertIntermediate,
    BertOutput,
    BertConfig,
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions
)
from transformers import BertTokenizer
from typing import List, Optional, Tuple, Union
import math

class CustomBertSelfAttention(BertSelfAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
        modified_attention_probs: Optional[torch.FloatTensor] = None, # new parameter
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        use_cache = past_key_value is not None
        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            if use_cache:
                position_ids_l = torch.tensor(key_length - 1, dtype=torch.long, device=hidden_states.device).view(
                    -1, 1
                )
            else:
                position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        if modified_attention_probs is not None:
            attention_probs = modified_attention_probs
            
        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs
    
class CustomBertAttention(BertAttention):
    def __init__(self, config):
        super().__init__(config)
        
        self.self = CustomBertSelfAttention(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
        modified_attention_probs: Optional[torch.FloatTensor] = None  # new parameter
    ):

        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
            modified_attention_probs=modified_attention_probs
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class CustomBertLayer(BertLayer):
    def __init__(self, config):
        super().__init__(config)
        # use custom attention
        self.attention = CustomBertAttention(config)
        # keep the rest the same
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
        modified_attention_probs: Optional[torch.FloatTensor] = None  # new parameter
    ):

        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions=output_attentions,
            modified_attention_probs=modified_attention_probs  # pass it to attention layer
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # continue output tuple

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs

        return outputs


class CustomBertEncoder(BertEncoder):
    def __init__(self, config):
        super().__init__(config)
        # use custom layer
        self.layer = nn.ModuleList([CustomBertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        modified_attention_list: Optional[Union[List[Optional[torch.FloatTensor]], Tuple[Optional[torch.FloatTensor]]]] = None  # new parameter
    ):
        """
        modified_attention_list:(in English)
        A list of modified attention probabilities for each layer.
        If None, the original attention probabilities are used.
        If not None, the length of the list should be equal to the number of layers.
        """
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        # if self.gradient_checkpointing and self.training:
        #     if use_cache:
        #         logger.warning_once(
        #             "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
        #         )
        #         use_cache = False


        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None
            # add modified_attention_probs to the layer
            modified_attention_for_this_layer = None
            if modified_attention_list is not None and i < len(modified_attention_list):
                modified_attention_for_this_layer = modified_attention_list[i]

            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                head_mask[i] if head_mask is not None else None,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_values[i] if past_key_values is not None else None,
                output_attentions,
                modified_attention_probs=modified_attention_for_this_layer
            )
            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class CustomBertModel(BertModel):
    def __init__(self, config):
        super().__init__(config)
        # use custom encoder
        self.encoder = CustomBertEncoder(config)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        modified_attention_list: Optional[Union[List[Optional[torch.FloatTensor]], Tuple[Optional[torch.FloatTensor]]]] = None  # new parameter
    ):
        # 参考 HuggingFace 的 BertModel.forward() 实现
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Handle inputs_embeds vs input_ids
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # Handle token_type_ids
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(batch_size, seq_length)
        elif self.embeddings.token_type_embeddings.weight is not None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # Handle attention_mask
        extended_attention_mask: Optional[torch.Tensor] = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # Handle head_mask
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        # Get embeddings
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=0,  # 假设没有缓存
        )

        # Pass through encoder with modified_attention_list
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            modified_attention_list=modified_attention_list,  # 传递自定义注意力列表
        )

        sequence_output = encoder_outputs.last_hidden_state
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            output = (sequence_output, pooled_output) + encoder_outputs[1:]
            return tuple(v for v in output if v is not None)

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=None,
        )

# def load_custom_bert_model(model_name_or_path):
#     config = BertConfig.from_pretrained(model_name_or_path)
#     model = CustomBertModel(config)
#     state_dict = torch.load(f"{model_name_or_path}/pytorch_model.bin", map_location="cpu")
#     model.load_state_dict(state_dict, strict=False)
#     return model


class ModifiedBert(nn.Module):
    def __init__(self, model_name_or_path,num_labels = 5):
        super(ModifiedBert, self).__init__()
        config = BertConfig.from_pretrained(model_name_or_path)
        self.model = CustomBertModel(config)  # 用自定义BertModel
        self.classfier = nn.Linear(config.hidden_size, num_labels)
        self.tokenizer = BertTokenizer.from_pretrained(model_name_or_path)
        
    def f(self, 
        input_ids,
        attention_mask = None,
        token_type_ids = None,
    ):
        # get [cls] attention weight
        outputs = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,output_attentions=True,output_hidden_states=True)
        attentions = outputs.attentions
        last_attention = attentions[-1]        # Shape: [batch_size, num_heads, seq_len, seq_len]
        # 带有head的[CLS] attention
        head_cls_attention = last_attention[:, :, 0, :] # Shape: [batch_size, num_heads, seq_len]
        return head_cls_attention,last_attention
    
    # def g(self, input_ids, attention_mask, token_type_ids, modified_attention):
    #     """
    #     这里严格走原生 forward，但最后一层注意力用 v。
    #     假设 BERT 有12层，只替换最后一层 => modified_attention_list = [None]*11 + [v_4D]
    #     注意v必须是[batch, num_heads, seq_len, seq_len]
    #     """
    #     # 这里只是演示，把 v 传给最后一层
    #     # 你需要先把 v reshape/expand 成 [batch_size, num_heads, seq_len, seq_len]
    #     # 如果你只想改 `[CLS]` 行，可以把 attention_probs.clone() 后只替换第0行，然后传进来
    #     num_layers = self.model.config.num_hidden_layers
    #     modified_attention_list = [None] * (num_layers - 1) + [modified_attention]

    #     logits = self.forward(
    #         input_ids=input_ids,
    #         attention_mask=attention_mask,
    #         token_type_ids=token_type_ids,
    #         modified_attention_list=modified_attention_list
    #     )
    #     return logits
    def g(self, input_ids, attention_mask, token_type_ids, v):
        """
        严格按照原始 forward 机制，但替换最后一层的 [CLS] 注意力权重。
        假设有 N 层，这里只替换最后一层的注意力。
        
        :param v: [CLS] 注意力权重，形状为 [batch_size, num_heads, seq_len]
                  需要扩展为 [batch_size, num_heads, 1, seq_len]，因为只替换第0行。
        :return: logits
        """
        batch_size, num_heads, seq_len = v.size()
        # 将 v 扩展为 [batch_size, num_heads, 1, seq_len]
        v_expanded = v.unsqueeze(2)  # [batch_size, num_heads, 1, seq_len]
        
        # 构造一个与注意力矩阵相同形状的 tensor，其中只替换第0行
        # 首先，获取一个全1的 tensor
        mask = torch.ones((batch_size, num_heads, seq_len, seq_len), device=v.device)
        # 将第0行替换为 v_expanded
        mask[:, :, 0, :] = v_expanded.squeeze(2)
        # 其余部分保持原有注意力权重（通过 masking + v 的方式）
        # 这里假设你想完全替换第0行的注意力权重
        # 需要确保 v 是一个概率分布 (softmax后的)

        modified_attention_list = [None]*(self.model.encoder.config.num_hidden_layers -1) + [mask]

        logits = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            modified_attention_list=modified_attention_list
        )
        return logits
    
    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        modified_attention_list=None
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=True,
            modified_attention_list=modified_attention_list  # 关键
        )
        # 取 pooler_output 或 [CLS] hidden state
        pooled_output = outputs["pooler_output"]  # [batch, hidden_size]
        logits = self.classfier(pooled_output)
        return logits


    def get_model(self):
        return self.model

    def get_tokenizer(self):
        return self.tokenizer
    
    
def load_custom_model(checkpoint_path, model_path="bert-base-uncased", device='cpu'):
    # 初始化自定义分类模型
    model = ModifiedBert(model_path)
    model.to(device)
    
    # 加载保存的 state_dict
    state_dict = torch.load(checkpoint_path, map_location=device)
    
    # 处理 state_dict：移除 'model.' 前缀，并修正拼写错误
    new_state_dict = {}
    for k, v in state_dict.items():
        # 移除 'model.' 前缀
        print(k)
        if k.startswith('model.'):
            new_key = k[len('model.'):]
        else:
            new_key = "bert." + k
        
        # 修正 'classfier' 为 'classifier'
        if 'classfier' in new_key:
            new_key = new_key.replace('classfier', 'classifier')
        
        new_state_dict[new_key] = v
    
    # 加载 state_dict
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    
    # 打印缺失和意外的键，便于调试
    if missing_keys:
        print("Missing keys:", missing_keys)
    if unexpected_keys:
        print("Unexpected keys:", unexpected_keys)
    
    if not missing_keys and not unexpected_keys:
        print("All weights loaded successfully.")
    else:
        print("Some weights were not loaded. Please check the keys.")
    
    return model




# # Load fine-tuned model and tokenizer
# model_path = ".\\model_hub\\bert-base"
# checkpoint_path = ".\\trained_model\\model_step_43000.pth"  
# config = BertConfig.from_pretrained(model_path)
# model = CustomBertModel(config)
# state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
# model.load_state_dict(state_dict, strict=False)

# model.eval()
# from transformers import BertTokenizer

# tokenizer = BertTokenizer.from_pretrained(model_path)
# input_text = "This is a sample input for testing."
# inputs = tokenizer(input_text, return_tensors="pt")

# modified_attention_list = None 

# with torch.no_grad():
#     outputs = model(
#         input_ids=inputs['input_ids'],
#         attention_mask=inputs['attention_mask'],
#         token_type_ids=inputs.get('token_type_ids'),
#         modified_attention_list=modified_attention_list
#     )


# last_hidden_state = outputs.last_hidden_state
# print(last_hidden_state.shape)  


# # test when modified_attention_list is not None
# modified_attention_list = [None] * config.num_hidden_layers  # create a list of None
# # modify the attention probabilities of the 5th layer using the same shape as the original attention probabilities
# modified_attention_list[4] = torch.ones(1, 12, 12) / 12  # set all attention probabilities to 1/12
# # Expected size for first two dimensions of batch2 tensor to be: [12, 12] but got: [12, 10].
# # This is because the attention mask is not square, so it is not compatible with the modified attention probabilities.
# # To fix this, we need to make the attention mask square by adding more padding tokens to the input sequence.
# # We can use the tokenizer to do this automatically.
# input_text = "This is a sample input for testing."
# inputs = tokenizer(input_text, return_tensors="pt", padding="max_length", max_length=12, truncation=True)
# with torch.no_grad():
#     outputs = model(
#         input_ids=inputs['input_ids'],
#         attention_mask=inputs['attention_mask'],
#         token_type_ids=inputs.get('token_type_ids'),
#         modified_attention_list=modified_attention_list
#     )




