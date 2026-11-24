from experiments.exp6.dataset_exp6 import Exp6Dataset
from experiments.exp6.model_exp6 import build_model
from experiments.common.engine import train

if __name__ == "__main__":
    train(6, Exp6Dataset, build_model, use_mask=True)
