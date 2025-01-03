from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
import torch.nn.functional as F
# Load pre-trained model and tokenizer
class Bert(nn.Module):
    def __init__(self, model_path):
        super(Bert, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertModel.from_pretrained(model_path, output_attentions=True)
        self.classfier = nn.Linear(768, 5)
        

    def get_model(self):
        return self.model

    def get_tokenizer(self):
        return self.tokenizer
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        output = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, output_attentions=True, output_hidden_states=True)
        logits = self.classfier(output.pooler_output)     
        return output, logits
    
    # def f(self, input_ids, attention_mask, token_type_ids):
    #     # get [cls] attention weight
    #     outputs,_ = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
    #     attentions = outputs.attentions
    #     last_attention = attentions[-1]        # Shape: [batch_size, num_heads, seq_len, seq_len]
    #     avg_attention = last_attention.mean(dim=1)
    #     cls_attention = avg_attention[:, 0, :] # Shape: [batch_size, seq_len]
    #     return cls_attention 

        
    # def g(self, input_ids, attention_mask, token_type_ids, modified_cls_attention):
    #     """
    #     输入修改后的modified_cls_attention(最后一层attention中avg_attention中的cls_attention)，然后维持不变并进行前向传播
    #     :param input_ids: 输入的 token ids
    #     :param attention_mask: 注意力 mask
    #     :param token_type_ids: token 类型 id
    #     :param modified_weights: 修改的[CLS]注意力权重，形状: [batch_size, seq_len]
    #     :return: logits: 分类结果
    #     """
        
    #     return logits

    

        

# # test
# bertmodel = Bert(".\\model_hub\\bert-base")
# tokenizer = bertmodel.get_tokenizer()
# model = bertmodel.get_model()
# text = "Replace me by any text you'd like."
# encoded_input = tokenizer(text, return_tensors='pt')
# output = bertmodel(**encoded_input)
# print(output[1])
# # print(encoded_input)