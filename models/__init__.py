from .bert import BERTModel
from .config import Config
from .tokenizer import Tokenizer


MODELS = {
    BERTModel.code(): BERTModel
}


def model_factory(args):
    model = MODELS[args.model_code]
    return model(args)
