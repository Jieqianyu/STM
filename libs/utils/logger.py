import numpy as np
import time
import logging

from collections import deque

def set_logging(filename, resume=False):
    root_logger = logging.getLogger()

    ch = logging.StreamHandler()
    fh = logging.FileHandler(filename=filename, mode='a' if resume else 'w')

    root_logger.setLevel(logging.INFO)
    ch.setLevel(logging.INFO)
    fh.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    root_logger.addHandler(ch)
    root_logger.addHandler(fh)

class Logger(object):

    DefaultItemCount = 1

    def __init__(self, fpath, resume=False):

        mode = 'a' if resume else 'w'
        self.file = open(fpath, mode)
        self.items = []
        self.vals = []

    def close(self):

        self.file.close()
        self.items = []
        self.vals = []

    def set_items(self, item_names=None):

        if item_names is None:
            self.items.append('term %d' % self.DefaultItemCount)
            self.DefaultItemCount += 1
        elif isinstance(item_names, list):
            for item_name in item_names:
                self.items.append(item_name)

    def log(self, *terms):

        assert len(terms) == len(self.items), 'mismatch logger information'

        self.file.write('==> log info time: %s' % time.ctime())
        self.file.write('\n')

        log = ''
        for item, val in zip(self.items, terms):
            if isinstance(val, float):
                formats = '%s %.5f '
            else:
                formats = '%s %d '

            log += formats % (item, val)

        self.file.write(log)

        self.file.write('\n')

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self,):
        self.reset() 

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
