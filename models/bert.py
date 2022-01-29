from .base import BaseModel
from .bert_modules.bert import BERT
from .config import Config

import json
import torch
import torch.nn as nn


class BERTModel(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.bert = BERT(args)
        self.out = nn.Linear(self.bert.hidden, args.num_items + 1)

    @classmethod
    def code(cls):
        return 'bert'

    def forward(self, x):
        x = self.bert(x)
        return self.out(x)

    def predict_topk(self, x, k=100):
        x = self(x)[:,-1,:]
        return torch.topk(x, k, dim=1)
    
    @classmethod
    def load_trained(self, config_path, pth_path):
        with open("artifacts/config.json") as f:
            config = Config(**json.load(f))
            
        model = BERTModel(config)
        sdict = torch.load(pth_path, map_location=torch.device('cpu'))
        model.load_state_dict(sdict["model_state_dict"])
        return model.eval()