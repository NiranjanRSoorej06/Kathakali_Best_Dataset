from experiments.exp4.dataset_exp4 import Exp4Dataset
from experiments.exp4.model_exp4 import build_model
from experiments.common.engine import train

if __name__ == "__main__":
    train(4, Exp4Dataset, build_model, use_mask=False)
