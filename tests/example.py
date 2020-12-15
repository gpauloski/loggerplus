import os
import loggerplus as logger

os.makedirs('tests/output', exist_ok=True)

logger.init(
    handlers=[
        logger.StreamHandler(),
        logger.FileHandler('tests/output/test.log', overwrite=True),
        logger.TorchTensorboardHandler('tests/output'),
        logger.CSVHandler('tests/output/test.csv', overwrite=False),
    ]
)

for i in range(10):
    for j in range(5):
        global_step = i*10 + j
        logger.log(tag='train', 
                   step=global_step, 
                   lr=0.001/(global_step+1), 
                   loss=100/(global_step+1))
    logger.info('Completed epoch {}'.format(i))

