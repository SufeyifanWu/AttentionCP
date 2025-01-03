# get attention weights to visualize key words
from transformers import BertTokenizer, BertModel
from train import train
from data_process import AmazonReviewsDatasetSplit
from utils import clean_text, norm_analysis_v1, weighted_analysis, norm_analysis_v2
from model import Bert
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import sys
import string
import re
import textwrap

# train model
model_path = ".\\model_hub\\bert-base"
csv_file = ".\\data\\Reviews.csv"
max_len = 512
batch_size = 16
learning_rate = 2e-5
num_epochs = 2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device: ", device)
train(model_path, csv_file, max_len, batch_size, num_epochs, learning_rate, device)



# Attention weights
# Load fine-tuned model and tokenizer
# model_path = ".\\model_hub\\bert-base"
# max_len = 512
# model = Bert(model_path)
# tokenizer = model.get_tokenizer()
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # Load fine-tuned model
# state_dict = torch.load(".\\trained_model\\model_step_43000.pth", weights_only=True)
# model.load_state_dict(state_dict)
# model.eval()
# model.to(device)

# # Get attention weights from test set
# data = AmazonReviewsDatasetSplit(".\\data\\Reviews.csv", tokenizer, max_len)
# _, _, test_dataset = data.get_data()
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)

# for i, data in enumerate(test_loader):
#     if i < 1000:
        
#         input_ids = data['input_ids']
#         attention_mask = data['attention_mask']
#         token_type_ids = data['token_type_ids']
#         labels = data['label']
        
#         if labels != 0:
#             continue
        
#         input_ids = input_ids.to(device)
#         attention_mask = attention_mask.to(device)
#         token_type_ids = token_type_ids.to(device)
#         labels = labels.to(device)
        
#         pred_label = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[1].argmax().item()
#         if pred_label != 0:
#             continue
        
        
#         initial_attention_weights = weighted_analysis(model, input_ids, attention_mask, token_type_ids)
#         last_layer_norm_weights = norm_analysis_v1(model, input_ids, attention_mask, token_type_ids)
#         value_vectors_norm_weights = norm_analysis_v2(model, input_ids, attention_mask, token_type_ids)
        
#         # Convert token IDs back to tokens
#         tokens = tokenizer.convert_ids_to_tokens(input_ids[0], skip_special_tokens=True)
        
#         # Filter out special tokens
#         filtered_tokens = []
#         initial_filtered_weights = []
#         last_layer_norm_filtered_weights = []
#         value_vectors_norm_filtered_weights = []
#         for token, initial_weight, last_layer_norm_weight, value_vectors_norm_weight in zip(tokens, initial_attention_weights, last_layer_norm_weights, value_vectors_norm_weights):
#             if token not in ["[PAD]"]:
#                 filtered_tokens.append(token)
#                 initial_filtered_weights.append(initial_weight)
#                 last_layer_norm_filtered_weights.append(last_layer_norm_weight)
#                 value_vectors_norm_filtered_weights.append(value_vectors_norm_weight)
        
#         text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
#         print(type(last_layer_norm_filtered_weights))
#         print(last_layer_norm_filtered_weights)
#         print("text: ", text)
#         print("true label: ", labels[0].item(), "predicted label: ", model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[1].argmax().item())

#         # Plot the initial attention weights
#         plt.figure(figsize=(10, 5))
#         sns.barplot(x=filtered_tokens, y=initial_filtered_weights, palette='viridis',ci=None)
#         plt.xlabel(f"Initial Attention Weights")
#         plt.ylabel("Attention Weight")
#         plt.title("\n".join(textwrap.wrap(f"Text: {text}", width=100)))
#         plt.xticks(rotation=90)


#         # Plot the last_layer_norm_filtered_weights
#         plt.figure(figsize=(10, 5))
#         sns.barplot(x=filtered_tokens, y=last_layer_norm_filtered_weights, palette='viridis',ci=None)
#         plt.xlabel(f"The last layer norm weights")
#         plt.ylabel("Attention Weight")
#         plt.title("\n".join(textwrap.wrap(f"Text: {text}", width=100)))
#         plt.xticks(rotation=90)
        
#         # Plot the value_vectors_norm_filtered_weights
#         plt.figure(figsize=(10, 5))
#         sns.barplot(x=filtered_tokens, y=value_vectors_norm_filtered_weights, palette='viridis',ci=None)
#         plt.xlabel(f"Value Vectors Norm Weights")
#         plt.ylabel("Attention Weight")
#         plt.title("\n".join(textwrap.wrap(f"Text: {text}", width=100)))
#         plt.xticks(rotation=90)
#         plt.show()
        
        

#         break






# all_attention_dict ={}
# # Get attention weights for a random sample
# for i, data in enumerate(test_loader):
#     if i == 1:
#         input_ids = data['input_ids']
#         attention_mask = data['attention_mask']
#         token_type_ids = data['token_type_ids']
#         labels = data['label']
        
#         input_ids = input_ids.to(device)
#         attention_mask = attention_mask.to(device)
#         token_type_ids = token_type_ids.to(device)
#         labels = labels.to(device)
        
        
#         # Get attention weights from the model
#         with torch.no_grad():
#             output, logits = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
#         attentions = output.attentions
#         print(len(attentions))
#         # Extract last layer's attention: shape [batch_size, num_heads, seq_len, seq_len]
#         print(attentions[-1].shape)
#         last_layer_attention = attentions[-1][0]  # Take batch's first sample
#         print(last_layer_attention.shape)
#         # break
        
        
#         avg_attention = last_layer_attention.mean(dim=0)  # Average over all heads, shape: [seq_len, seq_len]

#         # [CLS] token attention (first row)
#         cls_attention = avg_attention[0].cpu().detach().numpy()
        
#         #
#         hidden_states = output.hidden_states 
#         last_hidden_state = hidden_states[-1]  
            
#         # # Convert token IDs back to tokens
#         # tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
#         # cls_attention = cls_attention[:len(tokens)]  # Align attention weights with tokens

#         # # Filter out special tokens
#         # filtered_tokens = []
#         # filtered_weights = []
#         # for token, weight in zip(tokens, cls_attention):
#         #     if token not in ["[CLS]", "[SEP]", "[PAD]"]:
#         #         filtered_tokens.append(token)
#         #         filtered_weights.append(weight)
                
#         # # 排序tokens和weights,并限制tokens数量,以便于可视化,这里限制为20个tokens,如果超过20个tokens,取前20个
#         # sorted_index = np.argsort(filtered_weights)[::-1]
#         # sorted_tokens = [filtered_tokens[i] for i in sorted_index]
#         # sorted_weights = [filtered_weights[i] for i in sorted_index]
        
#         # if len(sorted_tokens) > 20:
#         #     sorted_tokens = sorted_tokens[:20]
#         #     sorted_weights = sorted_weights[:20]
        
        
#         # # Create directory to save attention weights
#         # if not os.path.exists(".\\attention_weights"):
#         #     os.makedirs(".\\attention_weights")
        
#         # # Save attention weights as a dict, {text:{sorted_tokens:sorted_weights}}
#         # attention_dict = dict(zip(sorted_tokens, sorted_weights))
        
#         # # 合并filtered_tokens list为一个字符串
#         # filtered_tokens = " ".join(filtered_tokens)
#         # all_attention_dict[filtered_tokens] = attention_dict
        
#         # # Plot the [CLS] attention heatmap
#         # plt.figure(figsize=(10, 5))
#         # sns.barplot(x=sorted_tokens, y=sorted_weights,palette='viridis')
#         # plt.xlabel(f"{filtered_tokens}")
#         # plt.ylabel("Attention Weight (CLS)")
#         # plt.title(f"ture lable:{labels[0].item()} vs predicted lable:{logits.argmax().item()}")
#         # plt.xticks(rotation=90)
        
#         # # Save plot
#         # plt.tight_layout()
#         # plt.savefig(f".\\attention_weights\\attention_weights_{i}.png")
        
#         # # # Print top 10 tokens and attention weights
#         # # top_k = 10
#         # # top_tokens = np.argsort(cls_attention)[::-1][:top_k]
#         # # print(f"Text: {filtered_tokens}")
#         # # print("Top 10 tokens and attention weights:")
#         # # for j in top_tokens:
#         # #     print(f"Token: {tokens[j]:<15} Attention weight: {cls_attention[j]:.4f}")
#         # # print("\n")
        
#         # # Save attention weights as a json file
#         # pd.DataFrame(all_attention_dict).to_json(f".\\attention_weights\\all_attention_weights.json")
        
        
        







