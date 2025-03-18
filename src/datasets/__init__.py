from src.datasets.LAVIB import LAVIBDataset
from src.datasets.DAVIS import DAVISDataset
from src.datasets.DAIN_HD import HDDataset
from src.datasets.SNU_FILM import SNUFILMDataset


def load_dataset(dataset_name, **dataset_args):
    if dataset_name == "LAVIB":
        return LAVIBDataset(**dataset_args)
    elif dataset_name == "DAVIS":
        return DAVISDataset(**dataset_args)
    elif dataset_name == "DAIN_HD":
        return HDDataset(**dataset_args)
    elif dataset_name == "SNU_FILM":
        return SNUFILMDataset(**dataset_args)
    else:
        raise f"No dataset named {dataset_name} in datasets!"
