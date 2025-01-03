# import re
# import torch

# def clean_text(tokens):
#     cleaned_tokens = []
#     for token in tokens:
#         if re.match(r'^[a-zA-Z]+$', token):
#             cleaned_tokens.append(token)
#     return cleaned_tokens

# def norm_analysis_v1(model, input_ids, attention_mask, token_type_ids):
#     # 自定义钩子函数
#     pre_norm_inputs = []
#     def hook_fn(module, input, output):
#         # 保存 LayerNorm 输入
#         pre_norm_inputs.append(input[0])  # input[0] 是输入
#     # 注册钩子
#     model.model.encoder.layer[-1].output.LayerNorm.register_forward_hook(hook_fn)
    
#     # Forward pass through the model
#     with torch.no_grad():
#         outputs,_ = model(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)


#     # import pdb; pdb.set_trace()
    
#     # Get hidden states and attentions
#     hidden_states = outputs.hidden_states  # List of hidden states per layer
#     # print(len(hidden_states))
#     attentions = outputs.attentions  # List of attention weights per layer
    
#     # Access the last layer's attention and hidden states
#     last_attention = attentions[-1]  # Shape: [batch_size, num_heads, seq_len, seq_len]
#     last_hidden_state = hidden_states[-1]  # Shape: [batch_size, seq_len, hidden_size]

#     # Average attention weights over all heads
#     avg_attention = last_attention.mean(dim=1)  # Shape: [batch_size, seq_len, seq_len]
    
#     # [CLS] token attention (first row)
#     cls_attention = avg_attention[:, 0, :] 
    
    
#     # cls_attention乘以 last_hidden_state[0] a_1 * h_1 + a_2 * h_2 + ... + a_n * h_n 得到 norm_attention
#     norm_attention = cls_attention.unsqueeze(-1) * last_hidden_state  # Shape: [seq_len, hidden_size] 
#     norm_attention = norm_attention.squeeze(0) # Remove batch dimension
    
#     # Normalize ||norm_attention||  (L2 norm)
#     norm_alpha_v = torch.norm(norm_attention, dim = 1, p = 2)  # Shape: [seq_len]
#     return norm_alpha_v.cpu().detach().numpy()



# def norm_analysis_v2(model, input_ids, attention_mask, token_type_ids=None): 

#     model.eval()
#     with torch.no_grad():
#         outputs,_ = model(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids
#         )

#     hidden_states = outputs.hidden_states  # List of hidden states per layer
#     attentions = outputs.attentions        # List of attention weights per layer
    
#     encoder_layers = model.model.encoder.layer

#     last_attention = attentions[-1]        # Shape: [batch_size, num_heads, seq_len, seq_len]

#     avg_attention = last_attention.mean(dim=1)  # Shape: [batch_size, seq_len, seq_len]
    
#     cls_attention = avg_attention[:, 0, :]       # Shape: [batch_size, seq_len]
    
#     last_hidden_state = hidden_states[-2]        # Shape: [batch_size, seq_len, hidden_size]
    

#     last_layer = encoder_layers[-1]
#     self_attention = last_layer.attention.self
#     value_linear = self_attention.value            # Linear
    

#     value_vectors = value_linear(last_hidden_state)  # Shape: [batch_size, seq_len, all_head_size]
    
#     norm_attention = cls_attention.unsqueeze(-1) * value_vectors # Shape: [batch_size, seq_len, all_head_size]
#     norm_attention = norm_attention.squeeze(0)  # Remove batch dimension
    
#     norm_alpha_v = torch.norm(norm_attention, dim=1, p=2)  # Shape: [seq_len]
#     return norm_alpha_v.cpu().detach().numpy()
    
# def norm_analysis_v2(model, input_ids, attention_mask, token_type_ids=None): 

#     model.eval()
#     with torch.no_grad():
#         outputs,_ = model(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids
#         )

#     hidden_states = outputs.hidden_states  # List of hidden states per layer
#     attentions = outputs.attentions        # List of attention weights per layer
    
#     encoder_layers = model.model.encoder.layer

#     last_attention = attentions[4]        # Shape: [batch_size, num_heads, seq_len, seq_len]

#     avg_attention = last_attention.mean(dim=1)  # Shape: [batch_size, seq_len, seq_len]
    
#     cls_attention = avg_attention[:, 0, :]       # Shape: [batch_size, seq_len]
    
#     last_hidden_state = hidden_states[4]        # Shape: [batch_size, seq_len, hidden_size]
    

#     last_layer = encoder_layers[4]
#     self_attention = last_layer.attention.self
#     value_linear = self_attention.value            # Linear
    

#     value_vectors = value_linear(last_hidden_state)  # Shape: [batch_size, seq_len, all_head_size]
    
#     norm_attention = cls_attention.unsqueeze(-1) * value_vectors # Shape: [batch_size, seq_len, all_head_size]
#     norm_attention = norm_attention.squeeze(0)  # Remove batch dimension
    
#     norm_alpha_v = torch.norm(norm_attention, dim=1, p=2)  # Shape: [seq_len]
#     return norm_alpha_v.cpu().detach().numpy()
    

# def weighted_analysis(model, input_ids, attention_mask, token_type_ids):
#     # Forward pass through the model
#     with torch.no_grad():
#         outputs,_ = model(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids
#         )

#     # Get attentions 
#     attentions = outputs.attentions  # List of attention weights per layer

#     # Access the last layer's attention and hidden states
#     last_attention = attentions[-1]  # Shape: [batch_size, num_heads, seq_len, seq_len]
#     last_layer_attention = last_attention[0]  # Take batch's first sample
#     # Average attention weights over all heads
#     avg_attention = last_attention.mean(dim=1)  # Shape: [batch_size, seq_len, seq_len]
    
#     # [CLS] token attention (first row)
#     cls_attention = avg_attention[:, 0, :].squeeze(0)  # Remove batch dimension
#     return cls_attention.cpu().detach().numpy()
    