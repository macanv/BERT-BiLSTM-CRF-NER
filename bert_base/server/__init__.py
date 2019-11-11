#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
#@Time    : 2019/1/28 10:51
# @Author  : MaCan (ma_cancan@163.com)
# @File    : server.py

some code copy from <https://hanxiao.github.io>
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing
import os
import random
import sys
import threading
import time
import pickle
from collections import defaultdict
from datetime import datetime
from multiprocessing import Process
from multiprocessing.pool import Pool

import numpy as np
import zmq
import zmq.decorators as zmqd
from termcolor import colored
from zmq.utils import jsonapi

from .helper import *
from .http import BertHTTPProxy
from .zmq_decor import multi_socket
sys.path.append('..')
from bert_base.train.models import convert_id_to_label

__all__ = ['__version__', 'BertServer']
__version__ = '1.7.8'

_tf_ver_ = check_tf_version()


class ServerCommand:
    terminate = b'TERMINATION'
    show_config = b'SHOW_CONFIG'
    new_job = b'REGISTER'

    @staticmethod
    def is_valid(cmd):
        return any(not k.startswith('__') and v == cmd for k, v in vars(ServerCommand).items())


class BertServer(threading.Thread):
    def __init__(self, args):
        super().__init__()
        self.logger = set_logger(colored('VENTILATOR', 'magenta'), args.verbose)

        self.max_seq_len = args.max_seq_len
        self.num_worker = args.num_worker
        self.max_batch_size = args.max_batch_size
        self.num_concurrent_socket = max(8, args.num_worker * 2)  # optimize concurrency for multi-clients
        self.port = args.port
        self.args = args
        self.status_args = {k: (v if k != 'pooling_strategy' else v.value) for k, v in sorted(vars(args).items())}
        self.status_static = {
            'tensorflow_version': _tf_ver_,
            'python_version': sys.version,
            'server_version': __version__,
            'pyzmq_version': zmq.pyzmq_version(),
            'zmq_version': zmq.zmq_version(),
            'server_start_time': str(datetime.now()),
        }
        self.processes = []
        # 如果BERT model path 不是空的，那么就启动bert模型
        if args.mode == 'BERT':
            self.logger.info('freeze, optimize and export graph, could take a while...')
            with Pool(processes=1) as pool:
                # optimize the graph, must be done in another process
                from .graph import optimize_bert_graph
                self.graph_path = pool.apply(optimize_bert_graph, (self.args,))
            # from .graph import optimize_graph
            # self.graph_path = optimize_graph(self.args, self.logger)
            if self.graph_path:
                self.logger.info('optimized graph is stored at: %s' % self.graph_path)
            else:
                raise FileNotFoundError('graph optimization fails and returns empty result')
        elif args.mode == 'NER':
            self.logger.info('lodding ner model, could take a while...')
            with Pool(processes=1) as pool:
                # optimize the graph, must be done in another process
                from .graph import optimize_ner_model
                num_labels, label2id, id2label = init_predict_var(self.args.model_dir)
                self.num_labels = num_labels + 1
                self.id2label = id2label
                self.graph_path = pool.apply(optimize_ner_model, (self.args, self.num_labels))
            if self.graph_path:
                self.logger.info('optimized graph is stored at: %s' % self.graph_path)
            else:
                raise FileNotFoundError('graph optimization fails and returns empty result')
        elif args.mode == 'CLASS':
            self.logger.info('lodding classification predict, could take a while...')
            with Pool(processes=1) as pool:
                # optimize the graph, must be done in another process
                from .graph import optimize_class_model
                num_labels, label2id, id2label = init_predict_var(self.args.model_dir)
                self.num_labels = num_labels
                self.id2label = id2label
                self.logger.info('contain %d labels:%s' %(num_labels, str(id2label.values())))
                self.graph_path = pool.apply(optimize_class_model, (self.args, num_labels))
            if self.graph_path:
                self.logger.info('optimized graph is stored at: %s' % self.graph_path)
            else:
                raise FileNotFoundError('graph optimization fails and returns empty result')
        else:
            raise ValueError('args model not special')

    def close(self):
        self.logger.info('shutting down...')
        self._send_close_signal()
        for p in self.processes:
            p.close()
        self.join()

    @zmqd.context()
    @zmqd.socket(zmq.PUSH)
    def _send_close_signal(self, _, frontend):
        frontend.connect('tcp://localhost:%d' % self.port)
        frontend.send_multipart([b'', ServerCommand.terminate, b'', b''])

    def run(self):
        self._run()

    @zmqd.context()
    @zmqd.socket(zmq.PULL)
    @zmqd.socket(zmq.PAIR)
    @multi_socket(zmq.PUSH, num_socket='num_concurrent_socket')
    def _run(self, _, frontend, sink, *backend_socks):

        def push_new_job(_job_id, _json_msg, _msg_len):
            # backend_socks[0] is always at the highest priority
            _sock = backend_socks[0] if _msg_len <= self.args.priority_batch_size else rand_backend_socket
            _sock.send_multipart([_job_id, _json_msg])

        # bind all sockets
        self.logger.info('bind all sockets')
        frontend.bind('tcp://*:%d' % self.port)
        addr_front2sink = auto_bind(sink)
        addr_backend_list = [auto_bind(b) for b in backend_socks]
        self.logger.info('open %d ventilator-worker sockets, %s'
                         % (len(addr_backend_list), ','.join(addr_backend_list)))

        # start the sink process
        # sink是用来接收上层BertWork的产出，然后发送给client
        self.logger.info('start the sink')
        proc_sink = BertSink(self.args, addr_front2sink)
        self.processes.append(proc_sink)
        proc_sink.start()
        addr_sink = sink.recv().decode('ascii')

        # start the backend processes
        # 这里启动多个进程，加载主模型
        device_map = self._get_device_map()
        for idx, device_id in enumerate(device_map):
            if self.args.mode == 'BERT':
                process = BertWorker(idx, self.args, addr_backend_list, addr_sink, device_id,
                                     self.graph_path, self.args.mode)
                self.processes.append(process)
                process.start()
            elif self.args.mode in ['NER', 'CLASS']:
                process = BertWorker(idx, self.args, addr_backend_list, addr_sink, device_id,
                                     self.graph_path, self.args.mode, self.id2label)
                self.processes.append(process)
                process.start()

        # start the http-service process
        if self.args.http_port:
            self.logger.info('start http proxy')
            proc_proxy = BertHTTPProxy(self.args)
            self.processes.append(proc_proxy)
            proc_proxy.start()

        rand_backend_socket = None
        server_status = ServerStatistic()
        while True:
            try:
                request = frontend.recv_multipart()
                client, msg, req_id, msg_len = request
            except ValueError:
                self.logger.error('received a wrongly-formatted request (expected 4 frames, got %d)' % len(request))
                self.logger.error('\n'.join('field %d: %s' % (idx, k) for idx, k in enumerate(request)), exc_info=True)
            else:
                server_status.update(request)
                if msg == ServerCommand.terminate:
                    break
                elif msg == ServerCommand.show_config:
                    self.logger.info('new config request\treq id: %d\tclient: %s' % (int(req_id), client))
                    status_runtime = {'client': client.decode('ascii'),
                                      'num_process': len(self.processes),
                                      'ventilator -> worker': addr_backend_list,
                                      'worker -> sink': addr_sink,
                                      'ventilator <-> sink': addr_front2sink,
                                      'server_current_time': str(datetime.now()),
                                      'statistic': server_status.value,
                                      'device_map': device_map,
                                      'num_concurrent_socket': self.num_concurrent_socket}

                    sink.send_multipart([client, msg, jsonapi.dumps({**status_runtime,
                                                                     **self.status_args,
                                                                     **self.status_static}), req_id])
                else:
                    self.logger.info('new encode request\treq id: %d\tsize: %d\tclient: %s' %
                                     (int(req_id), int(msg_len), client))
                    # register a new job at sink
                    sink.send_multipart([client, ServerCommand.new_job, msg_len, req_id])

                    # renew the backend socket to prevent large job queueing up
                    # [0] is reserved for high priority job
                    # last used backennd shouldn't be selected either as it may be queued up already
                    rand_backend_socket = random.choice([b for b in backend_socks[1:] if b != rand_backend_socket])

                    # push a new job, note super large job will be pushed to one socket only,
                    # leaving other sockets free
                    job_id = client + b'#' + req_id
                    if int(msg_len) > self.max_batch_size:
                        seqs = jsonapi.loads(msg)
                        job_gen = ((job_id + b'@%d' % i, seqs[i:(i + self.max_batch_size)]) for i in
                                   range(0, int(msg_len), self.max_batch_size))
                        for partial_job_id, job in job_gen:
                            push_new_job(partial_job_id, jsonapi.dumps(job), len(job))
                    else:
                        push_new_job(job_id, msg, int(msg_len))

        self.logger.info('terminated!')

    def _get_device_map(self):
        self.logger.info('get devices')
        run_on_gpu = False
        device_map = [-1] * self.num_worker
        if not self.args.cpu:
            try:
                import GPUtil
                num_all_gpu = len(GPUtil.getGPUs())
                avail_gpu = GPUtil.getAvailable(order='memory', limit=min(num_all_gpu, self.num_worker))
                num_avail_gpu = len(avail_gpu)

                if num_avail_gpu >= self.num_worker:
                    run_on_gpu = True
                elif 0 < num_avail_gpu < self.num_worker:
                    self.logger.warning('only %d out of %d GPU(s) is available/free, but "-num_worker=%d"' %
                                        (num_avail_gpu, num_all_gpu, self.num_worker))
                    if not self.args.device_map:
                        self.logger.warning('multiple workers will be allocated to one GPU, '
                                            'may not scale well and may raise out-of-memory')
                    else:
                        self.logger.warning('workers will be allocated based on "-device_map=%s", '
                                            'may not scale well and may raise out-of-memory' % self.args.device_map)
                    run_on_gpu = True
                else:
                    self.logger.warning('no GPU available, fall back to CPU')

                if run_on_gpu:
                    device_map = ((self.args.device_map or avail_gpu) * self.num_worker)[: self.num_worker]
            except FileNotFoundError:
                self.logger.warning('nvidia-smi is missing, often means no gpu on this machine. '
                                    'fall back to cpu!')
        self.logger.info('device map: \n\t\t%s' % '\n\t\t'.join(
            'worker %2d -> %s' % (w_id, ('gpu %2d' % g_id) if g_id >= 0 else 'cpu') for w_id, g_id in
            enumerate(device_map)))
        return device_map


class BertSink(Process):
    def __init__(self, args, front_sink_addr):
        super().__init__()
        self.port = args.port_out
        self.exit_flag = multiprocessing.Event()
        self.logger = set_logger(colored('SINK', 'green'), args.verbose)
        self.front_sink_addr = front_sink_addr
        self.verbose = args.verbose
        self.args = args

    def close(self):
        self.logger.info('shutting down...')
        self.exit_flag.set()
        self.terminate()
        self.join()
        self.logger.info('terminated!')

    def run(self):
        self._run()

    @zmqd.socket(zmq.PULL)
    @zmqd.socket(zmq.PAIR)
    @zmqd.socket(zmq.PUB)
    def _run(self, receiver, frontend, sender):
        receiver_addr = auto_bind(receiver)
        frontend.connect(self.front_sink_addr)
        sender.bind('tcp://*:%d' % self.port)

        pending_checksum = defaultdict(int)
        pending_result = defaultdict(list)
        job_checksum = defaultdict(int)

        poller = zmq.Poller()
        poller.register(frontend, zmq.POLLIN)
        poller.register(receiver, zmq.POLLIN)

        # send worker receiver address back to frontend
        frontend.send(receiver_addr.encode('ascii'))

        # Windows does not support logger in MP environment, thus get a new logger
        # inside the process for better compability
        logger = set_logger(colored('SINK', 'green'), self.verbose)
        logger.info('ready')

        while not self.exit_flag.is_set():
            socks = dict(poller.poll())
            if socks.get(receiver) == zmq.POLLIN:
                msg = receiver.recv_multipart()
                job_id = msg[0]
                job_info = job_id.split(b'@')
                job_id = job_info[0]
                partial_id = job_info[1] if len(job_info) == 2 else 0

                # parsing the ndarray
                if self.args.mode == 'BERT':
                    arr_info, arr_val = jsonapi.loads(msg[1]), msg[2]
                    # np.frombuffer() 以流的形式将数据转化为ndarray
                    X = np.frombuffer(memoryview(arr_val), dtype=arr_info['dtype'])
                    X = X.reshape(arr_info['shape'])
                    pending_result[job_id].append((X, partial_id))
                    pending_checksum[job_id] += X.shape[0]
                elif self.args.mode == 'NER':
                    arr_info, arr_val = jsonapi.loads(msg[1]), pickle.loads(msg[2])
                    pending_result[job_id].append((arr_val, partial_id))
                    pending_checksum[job_id] += len(arr_val)
                elif self.args.mode == 'CLASS':
                    arr_info, arr_val = jsonapi.loads(msg[1]), pickle.loads(msg[2])
                    pending_result[job_id].append((arr_val, partial_id))
                    pending_checksum[job_id] += len(arr_val['pred_label'])
                # print('type of msg[2]', type(arr_val))
                logger.info('collect job %s (%d/%d)' % (job_id,
                                                        pending_checksum[job_id],
                                                        job_checksum[job_id]))

                # check if there are finished jobs, send it back to workers
                finished = [(k, v) for k, v in pending_result.items() if pending_checksum[k] == job_checksum[k]]
                for job_info, tmp in finished:
                    logger.info('send back\tsize: %d\tjob id:%s\t' % (job_checksum[job_info], job_info))
                    # re-sort to the original order
                    tmp = [x[0] for x in sorted(tmp, key=lambda x: int(x[1]))]
                    client_addr, req_id = job_info.split(b'#')
                    if self.args.mode == 'CLASS': # 因为分类模型带上了分类的概率，所以不能直接返回结果，需要使用json格式的数据进行返回。
                        send_ndarray(sender, client_addr, tmp, req_id)
                    else:
                        send_ndarray(sender, client_addr, np.concatenate(tmp, axis=0), req_id)
                    pending_result.pop(job_info)
                    pending_checksum.pop(job_info)
                    job_checksum.pop(job_info)

            if socks.get(frontend) == zmq.POLLIN:
                client_addr, msg_type, msg_info, req_id = frontend.recv_multipart()
                if msg_type == ServerCommand.new_job:
                    job_info = client_addr + b'#' + req_id
                    job_checksum[job_info] = int(msg_info)
                    logger.info('job register\tsize: %d\tjob id: %s' % (int(msg_info), job_info))
                elif msg_type == ServerCommand.show_config:
                    time.sleep(0.1)  # dirty fix of slow-joiner: sleep so that client receiver can connect.
                    logger.info('send config\tclient %s' % client_addr)
                    sender.send_multipart([client_addr, msg_info, req_id])


class BertWorker(Process):
    def __init__(self, id, args, worker_address_list, sink_address, device_id, graph_path, mode, id2label=None):
        super().__init__()
        self.worker_id = id
        self.device_id = device_id
        self.logger = set_logger(colored('WORKER-%d' % self.worker_id, 'yellow'), args.verbose)
        self.max_seq_len = args.max_seq_len
        self.mask_cls_sep = args.mask_cls_sep
        self.daemon = True
        self.exit_flag = multiprocessing.Event()
        self.worker_address = worker_address_list
        self.num_concurrent_socket = len(self.worker_address)
        self.sink_address = sink_address
        self.prefetch_size = args.prefetch_size if self.device_id > 0 else None  # set to zero for CPU-worker
        self.gpu_memory_fraction = args.gpu_memory_fraction
        self.verbose = args.verbose
        self.graph_path = graph_path
        self.use_fp16 = args.fp16
        self.args = args
        self.mode = mode
        self.id2label = id2label

    def close(self):
        self.logger.info('shutting down...')
        self.exit_flag.set()
        self.terminate()
        self.join()
        self.logger.info('terminated!')

    def get_estimator(self, tf):
        from tensorflow.python.estimator.estimator import Estimator
        from tensorflow.python.estimator.run_config import RunConfig
        from tensorflow.python.estimator.model_fn import EstimatorSpec

        def model_fn(features, labels, mode, params):
            with tf.gfile.GFile(self.graph_path, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())

            input_names = ['input_ids', 'input_mask', 'input_type_ids']

            output = tf.import_graph_def(graph_def,
                                         input_map={k + ':0': features[k] for k in input_names},
                                         return_elements=['final_encodes:0'])

            return EstimatorSpec(mode=mode, predictions={
                'client_id': features['client_id'],
                'encodes': output[0]
            })

        def ner_model_fn(features, labels, mode, params):
            """
            命名实体识别模型的model_fn
            :param features:
            :param labels:
            :param mode:
            :param params:
            :return:
            """
            with tf.gfile.GFile(self.graph_path, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
            input_ids = features["input_ids"]
            input_mask = features["input_mask"]
            input_map = {"input_ids": input_ids, "input_mask": input_mask}
            pred_ids = tf.import_graph_def(graph_def, name='', input_map=input_map, return_elements=['pred_ids:0'])

            return EstimatorSpec(mode=mode, predictions={
                'client_id': features['client_id'],
                'encodes': pred_ids[0]
            })

        def classification_model_fn(features, labels, mode, params):
            """
            文本分类模型的model_fn
            :param features:
            :param labels:
            :param mode:
            :param params:
            :return:
            """
            with tf.gfile.GFile(self.graph_path, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
            input_ids = features["input_ids"]
            input_mask = features["input_mask"]
            #为了兼容多输入，增加segment_id特征，即训练代码中的input_type_ids特征。
            #input_map = {"input_ids": input_ids, "input_mask": input_mask}
            segment_ids=features["input_type_ids"]
            input_map = {"input_ids": input_ids, "input_mask": input_mask,"segment_ids":segment_ids}            
            pred_probs = tf.import_graph_def(graph_def, name='', input_map=input_map, return_elements=['pred_prob:0'])

            return EstimatorSpec(mode=mode, predictions={
                'client_id': features['client_id'],
                'encodes': tf.argmax(pred_probs[0], axis=-1),
                'score': tf.reduce_max(pred_probs[0], axis=-1)
            })

        # 0 表示只使用CPU 1 表示使用GPU
        config = tf.ConfigProto(device_count={'GPU': 0 if self.device_id < 0 else 1})
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = self.gpu_memory_fraction
        config.log_device_placement = False
        # session-wise XLA doesn't seem to work on tf 1.10
        # if args.xla:
        #     config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
        if self.mode == 'NER':
            return Estimator(model_fn=ner_model_fn, config=RunConfig(session_config=config))
        elif self.mode == 'BERT':
            return Estimator(model_fn=model_fn, config=RunConfig(session_config=config))
        elif self.mode == 'CLASS':
            return Estimator(model_fn=classification_model_fn, config=RunConfig(session_config=config))

    def run(self):
        self._run()

    @zmqd.socket(zmq.PUSH)
    @multi_socket(zmq.PULL, num_socket='num_concurrent_socket')
    def _run(self, sink, *receivers):
        # Windows does not support logger in MP environment, thus get a new logger
        # inside the process for better compatibility
        logger = set_logger(colored('WORKER-%d' % self.worker_id, 'yellow'), self.verbose)

        logger.info('use device %s, load graph from %s' %
                    ('cpu' if self.device_id < 0 else ('gpu: %d' % self.device_id), self.graph_path))

        tf = import_tf(self.device_id, self.verbose, use_fp16=self.use_fp16)
        estimator = self.get_estimator(tf)
        for sock, addr in zip(receivers, self.worker_address):
            sock.connect(addr)

        sink.connect(self.sink_address)
        for r in estimator.predict(input_fn=self.input_fn_builder(receivers, tf), yield_single_examples=False):
            if self.mode == 'BERT':
                send_ndarray(sink, r['client_id'], r['encodes'])
                logger.info('job done\tsize: %s\tclient: %s' % (r['encodes'].shape, r['client_id']))
            elif self.mode == 'NER':
                pred_label_result, pred_ids_result = ner_result_to_json(r['encodes'], self.id2label)
                rst = send_ndarray(sink, r['client_id'], pred_label_result)
                # print('rst:', rst)
                logger.info('job done\tsize: %s\tclient: %s' % (r['encodes'].shape, r['client_id']))
            elif self.mode == 'CLASS':
                pred_label_result = [self.id2label.get(x, -1) for x in r['encodes']]
                pred_score_result = r['score'].tolist()
                to_client = {'pred_label': pred_label_result, 'score': pred_score_result}
                rst = send_ndarray(sink, r['client_id'], to_client)
                logger.info('job done\tsize: %s\tclient: %s' % (r['encodes'].shape, r['client_id']))

    def input_fn_builder(self, socks, tf):
        import sys
        sys.path.append('..')
        from bert_base.bert.extract_features import convert_lst_to_features
        from bert_base.bert.tokenization import FullTokenizer

        def gen():
            tokenizer = FullTokenizer(vocab_file=os.path.join(self.args.bert_model_dir, 'vocab.txt'))
            # Windows does not support logger in MP environment, thus get a new logger
            # inside the process for better compatibility
            logger = set_logger(colored('WORKER-%d' % self.worker_id, 'yellow'), self.verbose)

            poller = zmq.Poller()
            for sock in socks:
                poller.register(sock, zmq.POLLIN)

            logger.info('ready and listening!')
            while not self.exit_flag.is_set():
                events = dict(poller.poll())
                for sock_idx, sock in enumerate(socks):
                    if sock in events:
                        #接收来自客户端的消息
                        client_id, raw_msg = sock.recv_multipart()
                        msg = jsonapi.loads(raw_msg)
                        logger.info('new job\tsocket: %d\tsize: %d\tclient: %s' % (sock_idx, len(msg), client_id))
                        # check if msg is a list of list, if yes consider the input is already tokenized
                        # 对接收到的字符进行切词，并且转化为id格式
                        # logger.info('get msg:%s, type:%s' % (msg[0], type(msg[0])))
                        is_tokenized = all(isinstance(el, list) for el in msg)
                        tmp_f = list(convert_lst_to_features(msg, self.max_seq_len, tokenizer, logger,
                                                             is_tokenized, self.mask_cls_sep))
                        #print([f.input_ids for f in tmp_f])
                        yield {
                            'client_id': client_id,
                            'input_ids': [f.input_ids for f in tmp_f],
                            'input_mask': [f.input_mask for f in tmp_f],
                            'input_type_ids': [f.input_type_ids for f in tmp_f]
                        }

        def input_fn():
            return (tf.data.Dataset.from_generator(
                gen,
                output_types={'input_ids': tf.int32,
                              'input_mask': tf.int32,
                              'input_type_ids': tf.int32,
                              'client_id': tf.string},
                output_shapes={
                    'client_id': (),
                    'input_ids': (None, self.max_seq_len),
                    'input_mask': (None, self.max_seq_len), #.shard(num_shards=4, index=4)
                    'input_type_ids': (None, self.max_seq_len)}).prefetch(self.prefetch_size))

        return input_fn


class ServerStatistic:
    def __init__(self):
        self._hist_client = defaultdict(int)
        self._hist_msg_len = defaultdict(int)
        self._client_last_active_time = defaultdict(float)
        self._num_data_req = 0
        self._num_sys_req = 0
        self._num_total_seq = 0
        self._last_req_time = time.perf_counter()
        self._last_two_req_interval = []
        self._num_last_two_req = 200

    def update(self, request):
        client, msg, req_id, msg_len = request
        self._hist_client[client] += 1
        if ServerCommand.is_valid(msg):
            self._num_sys_req += 1
            # do not count for system request, as they are mainly for heartbeats
        else:
            self._hist_msg_len[int(msg_len)] += 1
            self._num_total_seq += int(msg_len)
            self._num_data_req += 1
            tmp = time.perf_counter()
            self._client_last_active_time[client] = tmp
            if len(self._last_two_req_interval) < self._num_last_two_req:
                self._last_two_req_interval.append(tmp - self._last_req_time)
            else:
                self._last_two_req_interval.pop(0)
            self._last_req_time = tmp

    @property
    def value(self):
        def get_min_max_avg(name, stat):
            if len(stat) > 0:
                return {
                    'avg_%s' % name: sum(stat) / len(stat),
                    'min_%s' % name: min(stat),
                    'max_%s' % name: max(stat),
                    'num_min_%s' % name: sum(v == min(stat) for v in stat),
                    'num_max_%s' % name: sum(v == max(stat) for v in stat),
                }
            else:
                return {}

        def get_num_active_client(interval=180):
            # we count a client active when its last request is within 3 min.
            now = time.perf_counter()
            return sum(1 for v in self._client_last_active_time.values() if (now - v) < interval)

        parts = [{
            'num_data_request': self._num_data_req,
            'num_total_seq': self._num_total_seq,
            'num_sys_request': self._num_sys_req,
            'num_total_request': self._num_data_req + self._num_sys_req,
            'num_total_client': len(self._hist_client),
            'num_active_client': get_num_active_client()},
            get_min_max_avg('request_per_client', self._hist_client.values()),
            get_min_max_avg('size_per_request', self._hist_msg_len.keys()),
            get_min_max_avg('last_two_interval', self._last_two_req_interval),
            get_min_max_avg('request_per_second', [1. / v for v in self._last_two_req_interval]),
        ]

        return {k: v for d in parts for k, v in d.items()}


def init_predict_var(path):
    """
    初始化NER所需要的一些辅助数据
    :param path:
    :return:
    """
    label_list_file = os.path.join(path, 'label_list.pkl')
    label_list = []
    if os.path.exists(label_list_file):
        with open(label_list_file, 'rb') as fd:
            label_list = pickle.load(fd)
    num_labels = len(label_list)

    with open(os.path.join(path, 'label2id.pkl'), 'rb') as rf:
        label2id = pickle.load(rf)
        id2label = {value: key for key, value in label2id.items()}
    return num_labels, label2id, id2label


def ner_result_to_json(predict_ids, id2label):
    """
    NER识别结果转化为真实标签结果进行返回
    :param predict_ids:
    :param id2label
    :return:
    """
    if False:
        return predict_ids
    pred_label_result, pred_ids_result =\
                convert_id_to_label(predict_ids, id2label, len(predict_ids))
    return pred_label_result, pred_ids_result

