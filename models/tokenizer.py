import copy



class Tokenizer:
    
    def __init__(self, args, encoding_map):
        self.max_len = args.bert_max_len
        self.mask_token = len(encoding_map) + 1
        self.padding_token = 0
        self.unknown_token = 0
        self.encoding_map = copy.deepcopy(encoding_map)
        self.decoding_map = {i: s for i, s in encoding_map.items()}
        
    def encode(self, 
               inputs, 
               pad=True,
               insert_mask_token_last=False):
        
        inputs = [self.encoding_map.get(x, self.unknown_token) for x in inputs]
        
        target_len = self.max_len - 1 if insert_mask_token_last else self.max_len
        
        if len(inputs) > target_len:
            inputs = inputs[-target_len:]
            
        outputs = [self.padding_token] * (target_len - len(inputs)) + inputs
        
        if insert_mask_token_last:
            outputs.append(self.mask_token)
            
        return outputs
    
    def decode(self, inputs):
        return [self.decoding_map.get(x) for x in inputs]
