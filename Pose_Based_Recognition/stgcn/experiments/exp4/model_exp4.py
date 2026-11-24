from shared.graph.model_single_adjacency import STGCN

def build_model():
    return STGCN(num_classes=41, in_channels=2)
