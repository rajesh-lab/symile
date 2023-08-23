from torch.utils.data import Dataset

from templates import template_1, template_2, template_3, template_4

class SymileDataset(Dataset):
    """
    TODO: add comments
    """
    def __init__(self, args):
        t1 = template_1(args)
        breakpoint()
        # t2 = template_2(args)
        # t3 = template_3(args)
        # t4 = template_4(args)