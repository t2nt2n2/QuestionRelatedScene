from torchvision import datasets, transforms
from base import BaseDataLoader
import pandas as pd
import torch
from torch.utils.data import Dataset
import math
import random

class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class FriendsBertDataset(Dataset):
    def __init__(self, csv_file, tokenizer,transform=None):
        self.samples = pd.read_csv(csv_file)
        self.transform = transform
        self.tokenizer = tokenizer
    def __len__(self):
        return len(self.samples)
    def negative_sampling(self,idx):
        index = 0
        while(True):
            index = random.randrange(0,len(self.samples))
            if self.samples['shot_id'].values[index]!=self.samples['shot_id'].values[idx]:
                return self.samples['question'].values[index]

    def __getitem__(self, idx):
        index = self.samples['index'].values[idx]
        shot_id = self.samples['shot_id'].values[idx]
        question = self.samples['question'].values[idx]
        subtitles = self.samples['subtitles'].values[idx]
        shot_descs = self.samples['shot_descs'].values[idx]
        #label = self.samples['label'].values[idx+1]
        negative_question = self.negative_sampling(idx)
        print(question)

        try:
            input1 = "[CLS] " + question + " [SEP] " + subtitles
        except:
            subtitles = ""
            input1 = "[CLS] " + question + " [SEP] " + subtitles
        try:
            input3 = "[CLS] " + negative_question + " [SEP] " + subtitles
        except:
            subtitles = ""
            input3 = "[CLS] " + negative_question + " [SEP] " + subtitles

        #positive question with subtitle
        #print(input1)
        input1 = self.tokenizer.tokenize(input1)
        input1_SEP_index = input1.index('[SEP]')
        input1_segment = [0] * (input1_SEP_index+1) + [1] * (len(input1) - input1_SEP_index - 1)
        input1 = self.tokenizer.convert_tokens_to_ids(input1)
        if len(input1)>512:
            input1=input1[:512]
            input1_segment = input1_segment[:512]
        else:
            input1 = input1 + [0] * (512-len(input1))
            input1_segment = input1_segment + [1] * (512-len(input1_segment))

        #positive question with description
        input2 = "[CLS] " + question + " [SEP] " + shot_descs        
        #print(input2)
        input2 = self.tokenizer.tokenize(input2)
        input2_SEP_index = input2.index('[SEP]')
        input2_segment = [0] * (input2_SEP_index+1) + [1] * (len(input2) - input2_SEP_index - 1)
        input2 = self.tokenizer.convert_tokens_to_ids(input2)
        if len(input2)>512:
            input2=input2[:512]
            input2_segment =input2_segment[:512]
        else:
            input2 = input2 + [0] * (512-len(input2))
            input2_segment = input2_segment + [1] * (512-len(input2_segment))

        

        #negative question with subtitle
        #print(input3)
        input3 = self.tokenizer.tokenize(input3)
        input3_SEP_index = input3.index('[SEP]')
        input3_segment = [0] * (input3_SEP_index+1) + [1] * (len(input3) - input3_SEP_index - 1)
        
        input3 = self.tokenizer.convert_tokens_to_ids(input3)
        if len(input3)>512:
            input3=input3[:512]
            input3_segment= input3_segment[:512]
        else:
            input3 = input3 + [0] * (512-len(input3))
            input3_segment = input3_segment + [1] * (512-len(input3_segment))

        #negative question with description
        input4 = "[CLS] " + negative_question + " [SEP] " + shot_descs
        #print(input4)
        input4 = self.tokenizer.tokenize(input4)
        input4_SEP_index = input4.index('[SEP]')
        input4_segment = [0] * (input4_SEP_index+1) + [1] * (len(input4) - input4_SEP_index - 1)
        input4 = self.tokenizer.convert_tokens_to_ids(input4)        
               
        if len(input4)>512:
            input4=input4[:512]
            input4_segment = input4_segment[:512]
        else:
            input4 = input4 + [0] * (512-len(input4))
            input4_segment = input4_segment + [1] * (512-len(input4_segment))


        input1 = torch.tensor(input1)
        input1_segment = torch.tensor(input1_segment)
        input2 = torch.tensor(input2)
        input2_segment = torch.tensor(input2_segment)
        input3 = torch.tensor(input3)
        input3_segment = torch.tensor(input3_segment)
        input4 = torch.tensor(input4)
        input4_segment = torch.tensor(input4_segment)

        #print(input1.shape)

        sample = {'input1':(input1,input1_segment), 'input2':(input2,input2_segment),'input3':(input3,input3_segment),'input4':(input4,input4_segment)}

        if self.transform:
            sample = self.transform(sample)

        return sample


class FriendsBertDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True, tokenizer = None):
        trsfm = transforms.Compose([
            transforms.ToTensor()
        ])
        if tokenizer==None:
            tokenizer = torch.hub.load('huggingface/pytorch-pretrained-BERT', 'bertTokenizer', 'bert-base-cased', do_basic_tokenize=False, do_lower_case =False)
        
        self.dataset = FriendsBertDataset(data_dir,tokenizer)
        # for data in self.dataset:
        #     data
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)