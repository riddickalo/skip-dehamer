import logging
import argparse
from importlib import reload

def Params_logs(mode: str, args: argparse.Namespace) -> None:
    if mode == 'test':        
        path = f'logs/{args.model_dir}/test_params.txt'
    else:
        path = f'logs/{args.exp_prefix}/train_params.txt'
    
    with open(path, 'wt') as logs:
        logs.write('---------- Parameters ----------\n')
        for key, value in sorted(vars(args).items()):
            logs.write(f'{key}: {value}\n')
        logs.write('------------- End --------------\n')

def set_logger(mode: str, prefix: str, level: str, tty_stdout=False) -> logging.Logger:
    if level == 'debug':
        LEVEL = logging.DEBUG
    else:
        LEVEL = logging.INFO
    
    logger = logging.getLogger(prefix + mode)
    
    if tty_stdout:
        stream = logging.StreamHandler()
        stream.setLevel(LEVEL)
        streamformat = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
        stream.setFormatter = streamformat
        logger.addHandler(stream)
        
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                        filename=f'logs/{prefix}/{mode}_log.txt',
                        filemode='w',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=LEVEL)
    
    return logger

def close_logger() -> None:
    logging.shutdown()
    reload(logging)