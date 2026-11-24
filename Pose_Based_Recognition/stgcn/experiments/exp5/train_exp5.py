from experiments.exp5.dataset_exp5 import Exp5Dataset
from experiments.exp5.model_exp5 import build_model
from experiments.common.engine import train

if __name__ == "__main__":
    train(5, Exp5Dataset, build_model, use_mask=True)
