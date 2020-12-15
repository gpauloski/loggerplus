# loggerplus

Custom Logger for Machine Learning.
Inspired by [NVIDIA/dllogger](https://github.com/NVIDIA/dllogger), this logging utility provides seamless logging to stdout, file, CSVs, and TensorBoard.
loggerplus can also append to previous logging files in the event that training is resumed from a checkpoint.

## Install

From GitHub:
```
$ pip install git+https://github.com/gpauloski/loggerplus
```

For local development:
```
$ git clone https://github.com/gpauloski/loggerplus.git
$ cd loggerplus
$ pip install -e .
```

## Usage

```
import loggerplus as logger

logger.init(
    handlers=[
        logger.StreamHandler(),
        logger.FileHandler('logs/output.log', overwrite=True),
        logger.TorchTensorboardHandler('logs/'),
        logger.CSVHandler('logs/metrics.csv', overwrite=False),
    ]
)

metrics = {'LR': 0.001, 'loss': 2.356, 'accuracy': 0.768}
logger.log('TRAIN', step=0, **metrics)

logger.info('Finished Logging')
```

### Handlers

- `StreamHandler`: logs to standard out
- `FileHandler`: logs to a file, can append to existing files
- `TorchTensorboardHandler`: TensorBoard metric logging
- `CSVHandler`: CSV metric logging

All handlers have a verbose flag to disable logging (e.g. when using multiple processes so only the main process writes to files).

### Functions

`loggerplus.log(tag, step, **kwargs)` is the primary function for logging training metrics.

`loggerplus.info(message)` is essentially just a `print` and will log to both `StreamHandler` and `FileHandler`.
`TorchTensorboardHandler` and `CSVHandler` only log metrics and will ignore calls to `info()`.


