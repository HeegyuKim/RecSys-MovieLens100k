
import json
import torch
import pandas as pd
from models import Tokenizer, Config, BERTModel


with open("artifacts/config.json") as f:
    config = Config(**json.load(f))
    
with open("artifacts/smap.json") as f:
    smap = json.load(f)
    

ids = [2797, 2321, 720, 1270, 527, 2340, 48, 1097, 1721, 1545, 745, 2294, 3186, 1566, 588, 1907, 783, 1836, 1022, 2762, 150, 1, 1961]
ids = [str(x) for x in ids]
tk = Tokenizer(config, smap)

inputs = tk.encode(ids, insert_mask_token_last=True)

print(inputs)

bert = BERTModel.load_trained("artifacts/config.json", "artifacts/best_acc_model.pth")
values, indices = bert.predict_topk(torch.LongTensor([inputs]))
sids = tk.decode([str(x) for x in indices[0].detach().numpy()])
df = pd.DataFrame({"values": values[0].detach().numpy()}, index=sids)
print(df)