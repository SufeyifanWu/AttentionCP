import torch
from torch.utils.data import Dataset
import pandas as pd
import re


class AmazonReviewsDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        
        review = self.data.iloc[idx]['Text']
        review = self.clean_text(review)
        label = self.data.iloc[idx]['Score'] - 1
        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }
        
    @staticmethod
    def clean_text(text):
        text = re.sub(r'<.*?>', '', text)  # 移除HTML标签
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # 移除URLs
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # 移除非字母字符
        text = text.lower().strip()  # 转换为小写并去掉两端的空白字符
        return text
        
        
# Split data into train, val and test sets
class AmazonReviewsDatasetSplit:
    def __init__(self, csv_file, tokenizer, max_len, train_size=0.4, calibarate_size = 0.4, val_size=0.1, test_size=0.1):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.calibarate_size = calibarate_size
        
    def __split_data(self):
        train_data = self.data.sample(frac=self.train_size, random_state=42)
        calibarate_data = self.data.drop(train_data.index).sample(frac=self.calibarate_size/(1-self.train_size), random_state=42)
        val_data = self.data.drop(train_data.index).drop(calibarate_data.index).sample(frac=self.val_size/(1-self.train_size-self.calibarate_size), random_state=42)
        test_data = self.data.drop(train_data.index).drop(calibarate_data.index).drop(val_data.index)
        # save the data into {train, val, test} csv files
        train_data.to_csv("./data/train.csv", index=False)
        calibarate_data.to_csv("./data/calibarate.csv", index=False)
        val_data.to_csv("./data/val.csv", index=False)
        test_data.to_csv("./data/test.csv", index=False)
        return train_data, calibarate_data, val_data, test_data
    
    def get_data(self):
        train_data, calibarate_data, val_data, test_data = self.__split_data()
        train_dataset = AmazonReviewsDataset(train_data, self.tokenizer, self.max_len)
        calibarate_dataset = AmazonReviewsDataset(calibarate_data, self.tokenizer, self.max_len)
        val_dataset = AmazonReviewsDataset(val_data, self.tokenizer, self.max_len)
        test_dataset = AmazonReviewsDataset(test_data, self.tokenizer, self.max_len)
        
        return train_dataset, calibarate_dataset, val_dataset, test_dataset
        
        