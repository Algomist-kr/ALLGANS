from data_handler.BaseDataset import BaseDataset
from data_handler.BaseDatasetPack import BaseDatasetPack


class DummyDataset(BaseDataset):
    def __init__(self, set=None):
        super().__init__()

        if set is not None:
            for key in set:
                self.add_data(key, set[key])
