#!/usr/bin/python
# -*- coding:utf-8 -*-

import os
import logging
from types import SimpleNamespace
from datetime import datetime
from utils.logger import setlogger
from utils.train_val_test import train_val_test

def main():
    # minimal args for a quick smoke-test training run
    args = SimpleNamespace(
        model_name='Liconvformer',
        save_dataset=False,
        normalize_type='0-1',
        num_workers=0,
        batch_size=8,
        dataset_name='XJTU_gearbox',
        sigma=0.0,
        lr=0.001,
        patience=5,
        min_lr=1e-5,
        epoch=1,
        operation_num=1,
        only_test=False
    )

    save_dir = os.path.join('./results/{}'.format(args.dataset_name))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    setlogger(os.path.join(save_dir, args.model_name + '.log'))

    logging.info("\n")
    time = datetime.strftime(datetime.now(), '%m-%d %H:%M:%S')
    logging.info('{}'.format(time))
    for k, v in args.__dict__.items():
        logging.info("{}: {}".format(k, v))

    operation = train_val_test(args)
    for i in range(args.operation_num):
        if args.only_test == 0:
            operation.setup(i)
            operation.train_val(i)
        else:
            operation.setup(i)
        acc, j = operation.test(i)
        logging.info('Done op {}: acc {} j {}'.format(i, acc, j))

if __name__ == '__main__':
    main()
