
class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)
        
    def to_dict(self):
        return self.__dict__