from experiments.common.dataset import KathakaliDataset

class Exp4Dataset(KathakaliDataset):
    def __init__(self, split):
        super().__init__(split, in_channels=2)
