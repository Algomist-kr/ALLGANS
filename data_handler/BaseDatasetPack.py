from util.Logger import Logger


class BaseDatasetPack:
    def __init__(self):
        self.logger = Logger(self.__class__.__name__)
        self.log = self.logger.get_log()

        self.set = {}

    def load(self, path, **kwargs):
        for k in self.set:
            self.set[k].load(path, **kwargs)

    def shuffle(self):
        for key in self.set:
            self.set[key].shuffle()

    def split(self, from_key, a_key, b_key, rate):
        from_set = self.set[from_key]
        self.set.pop(from_key)

        a_set, b_set = from_set.split(rate)
        self.set[a_key] = a_set
        self.set[b_key] = b_set
        return a_set, b_set

    def merge_shuffle(self, a_key, b_key, rate):
        a_set = self.set[a_key]
        b_set = self.set[b_key]

        merge_set = a_set.merge(a_set, b_set)
        merge_set.shuffle()
        a_set, b_set = merge_set.split(rate, shuffle=True)

        self.set[a_key] = a_set
        self.set[b_key] = b_set
        return a_set, b_set

    def merge(self, a_key, b_key, merge_set_key):
        a_set = self.set[a_key]
        b_set = self.set[b_key]
        self.set.pop(a_key)
        self.set.pop(b_key)

        merge_set = a_set.merge(a_set, b_set)
        self.set[merge_set_key] = merge_set

    def sort(self, sort_key=None):
        for key in self.set:
            self.set[key].sort(sort_key)