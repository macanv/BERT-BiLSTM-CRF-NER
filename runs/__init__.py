# -*- coding: utf-8 -*-

"""

 @Time    : 2019/1/30 16:47
 @Author  : MaCan (ma_cancan@163.com)
 @File    : __init__.py.py
"""

def start_server():
    from server.server import BertServer
    from server.helper import get_run_args

    args = get_run_args()
    # print(args)
    server = BertServer(args)
    server.start()
    server.join()


def start_client():
    pass


def train_ner():
    import os
    from train.train_helper import get_args_parser
    from train.bert_lstm_ner import main

    args = get_args_parser()
    # print(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_map
    main(args=args)


# if __name__ == '__main__':
#     start_server()