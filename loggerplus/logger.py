import atexit
import csv
import logging
import os
import sys

from datetime import datetime

try:
    from torch.utils.tensorboard import SummaryWriter
    TORCH_TENSORBOARD = True
except:
    TORCH_TENSORBOARD = False


class Handler():
    def log(self, tag, step, **metrics):
        raise NotImplementedError()

    def info(self, message):
        raise NotImplementedError()

    def close(self):
        pass


class StreamHandler(Handler):
    def __init__(self, verbose=True):
        self.verbose = verbose

    def log(self, tag, step, **metrics):
        if self.verbose:
            print('{} {} -- step: {}  {}'.format(self._prefix(), tag, step, 
                    self._format_metrics(**metrics)))

    def info(self, message):
        if self.verbose:
            print('{} {}'.format(self._prefix(), message))

    def _prefix(self):
        return '[{}]'.format(
                datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])

    def _format_metric(self, metric):
        if isinstance(metric, float):
            if metric < 0.0001:
                return '{:.4e}'.format(metric)
            return '{:.4f}'.format(metric)
        return str(metric)

    def _format_metrics(self, **metrics):
        return '  '.join(['{}: {}'.format(k, self._format_metric(v)) 
                          for k, v in metrics.items()])
        

class FileHandler(StreamHandler):
    def __init__(self, filename, overwrite=False, verbose=True):
        super(FileHandler, self).__init__(verbose=verbose)
        if not self.verbose:
            return

        if not os.path.isdir(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))

        self.f = open(filename, 'w' if overwrite or 
                not os.path.isfile(filename) else 'a')
 
    def log(self, tag, step, **metrics):
        if self.verbose:
            self.f.write('{} {} -- step: {}  {}\n'.format(
                    self._prefix(), tag, step, self._format_metrics(**metrics)))

    def info(self, message):
        if self.verbose:
            self.f.write('{} {}\n'.format(self._prefix(), message))

    def close(self):
        if not self.verbose:
            return

        self.f.close()


class TorchTensorboardHandler(Handler):
    def __init__(self, filedir, verbose=True):
        if not TORCH_TENSORBOARD:
            raise RuntimeError('Unable to import torch.utils.tensorboard')

        if not os.path.isdir(filedir):
            os.makedirs(filedir)

        self.verbose = verbose
        self.filedir = filedir
        self.initialized = False

    def log(self, tag, step, **metrics):
        if not self.verbose:
            return

        # wait to intialize until first call to log so we can purge any
        # logged events greater than the current step
        if not self.initialized:
            self.tb = SummaryWriter(self.filedir, purge_step=step)
            self.initialized = True
        for key, value in metrics.items():
            self.tb.add_scalar('{}/{}'.format(tag, key), value,
                               global_step=step)

    def info(self, message):
        pass

    def close(self):
        if not self.verbose:
            return

        if self.initialized:
            self.tb.flush()


class CSVHandler(Handler):
    def __init__(self, filename, overwrite=False, verbose=True):
        self.verbose = verbose

        if not self.verbose:
            return

        self.headers = None

        if not os.path.isdir(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))

        if os.path.isfile(filename) and not overwrite:
            with open(filename, 'r') as f:
                reader = csv.DictReader(f)
                self.headers = reader.fieldnames

        self.f = open(filename, 'w' if overwrite 
                or not os.path.isfile(filename) else 'a')
        self.writer = None
        # We can only init the DictWriter once we know the fields so if the
        # file does not exists to read the headers, we delay creating the
        # object to the first call to log()
        if self.headers is not None:
            self.writer = csv.DictWriter(self.f, fieldnames=self.headers)

    def log(self, tag, step, **metrics):
        if not self.verbose:
            return

        metrics = {
            'timestamp': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
            'tag': tag, 
            'step': step, 
            **metrics
        }

        if self.writer is None:
            self.writer = csv.DictWriter(self.f, fieldnames=metrics.keys())
            self.writer.writeheader()
        self.writer.writerow(metrics)

    def info(self, message):
        pass

    def close(self):
        if not self.verbose:
            return

        self.f.close()


class Logger():
    def __init__(self, handlers):
        self.handlers = handlers
        atexit.register(self.close)

    def log(self, tag, step, **metrics):
        for h in self.handlers:
            h.log(tag, step, **metrics)

    def info(self, message):
        for h in self.handlers:
            h.info(message)

    def close(self):
        for h in self.handlers:
            h.close()

