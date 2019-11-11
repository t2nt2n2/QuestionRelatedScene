import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from pytorch_pretrained_bert import BertForQuestionAnswering
from pytorch_pretrained_bert import BertModel

from pytorch_pretrained_bert import BertConfig
import torch
class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class FriendsBertModel(BaseModel):
    def __init__(self, num_classes=2):
        super().__init__()
        config = BertConfig(vocab_size_or_config_json_file=30522)
        config.output_hidden_states=True

        #self.bert1 = torch.hub.load('huggingface/pytorch-pretrained-BERT', 'bertForQuestionAnswering', 'bert-base-cased', output_hidden_states=True)
        #self.bert2 = torch.hub.load('huggingface/pytorch-pretrained-BERT', 'bertForQuestionAnswering', 'bert-base-cased', output_hidden_states=True)
        self.bert1 = torch.hub.load('huggingface/pytorch-pretrained-BERT', 'bertModel', 'bert-base-cased')
        self.bert2 = torch.hub.load('huggingface/pytorch-pretrained-BERT', 'bertModel', 'bert-base-cased')
        #self.bert1 = BertForQuestionAnswering.from_pretrained(config=config)
        #self.bert2 = BertForQuestionAnswering.from_pretrained(config=config)
        #self.bert1 = BertModel.from_pretrained('bert-base-uncased')
        #self.bert2 = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(1536, num_classes)

    def forward(self, input1,input2,input3,input4):
        x1,_ = self.bert1(input1[0],input1[1])
        x2,_ = self.bert2(input2[0],input2[1])
        x3,_ = self.bert1(input3[0],input3[1])
        x4,_ = self.bert2(input4[0],input4[1])
        y1 = torch.cat((x1[:,0],x2[:,0]),1)
        y2 = torch.cat((x3[:,0],x4[:,0]),1)
        y1 = self.fc(y1)
        y2 = self.fc(y2)
        y1 = F.softmax(y1,dim=1)
        y2 = F.softmax(y2,dim=1)
        #print(y1,y2)
        #print(y1[:,0].shape,y2[:,0].shape)
        return (y1,y2)