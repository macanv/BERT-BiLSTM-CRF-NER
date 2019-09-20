# encoding=utf-8

"""
基于清华大学语料库的中文文本分类
Author:MaCan
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import codecs
import pickle
import numpy as np
import tensorflow as tf

import sys
# sys.path.append('..')
# from bert_base.server.helper import get_logger
from bert_base.bert import modeling
from bert_base.bert import optimization
from bert_base.bert import tokenization

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

if os.name == 'nt':
    bert_path = 'C:\迅雷下载\chinese_L-12_H-768_A-12'
    root_path = r'C:\workspace\python\BERT_Base'
else:
    bert_path = '/home/macan/ml/data/chinese_L-12_H-768_A-12'
    root_path = '/home/macan/ml/workspace/BERT_Base2'

flags = tf.flags
FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string("data_dir",  os.path.join(os.path.join(root_path, 'data'), 'classification'),
                    "The input data dir. Should contain the .tsv files (or other data files) for the task.")

flags.DEFINE_string(
    "bert_config_file", os.path.join(bert_path, 'bert_config.json'),
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", None, "The name of the task to train.")

flags.DEFINE_string("vocab_file", os.path.join(bert_path, 'vocab.txt'),
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", os.path.join(os.path.join(root_path, 'output'), 'classification'),
    "The output directory where the model checkpoints will be written.")

## Other parameters

flags.DEFINE_string(
    "init_checkpoint", os.path.join(bert_path, 'bert_model.ckpt'),
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 202,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool('clean', True, 'remove the files which created by last training')

flags.DEFINE_bool("do_train", True, "Whether to run training.")

flags.DEFINE_bool("do_eval", True, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", True,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float('dropout_keep_prob', 0.5, 'dropout probability')
flags.DEFINE_float("num_train_epochs", 5.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps",500,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_integer('save_summary_steps', 500, 'summary steps')

# logger = get_logger(os.path.join(FLAGS.output_dir, 'c.log'))
import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RestoreHook(tf.train.SessionRunHook):
    def __init__(self, init_fn):
        self.init_fn = init_fn

    def after_create_session(self, session, coord=None):
        if session.run(tf.train.get_or_create_global_step()) == 0:
            self.init_fn(session)

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with codecs.open(input_file, "r", encoding='utf-8') as f:
            lines = []
            for line in f:
                line = line.strip()
                if line == '':
                    continue
                line = line.split('__\t')
                if len(line) == 2:
                    line[0] = line[0].replace('__', '')
                    lines.append(line)
        return lines


class ThuProcessor(DataProcessor):
    """Processor for the Thu data set."""

    def __init__(self):
        self.labels = set()

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.txt")), 'train')

    def get_dev_examples(self, data_dir):
       return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.txt")), 'dev')

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.txt")), 'test')

    def get_labels(self):
        """在读取数据的时候，自动获取类别个数"""
        if not os.path.exists(os.path.join(FLAGS.output_dir, 'label_list.pkl')):
            with codecs.open(os.path.join(FLAGS.output_dir, 'label_list.pkl'), 'wb') as fd:
                pickle.dump(self.labels, fd)
        else:
            with codecs.open(os.path.join(FLAGS.output_dir, 'label_list.pkl'), 'rb') as fd:
                labels = pickle.load(fd)
            if len(labels) > len(self.labels):
                self.labels = labels
        return list(self.labels)

    def _create_examples(self, lines, set_type):
        examples = []
        np.random.shuffle(lines)
        for i, line in enumerate(lines):
            guid = '%s-%s' %(set_type, i)
            # if set_type == 'test':
            #     text_a = tokenization.convert_to_unicode(line[1])
            #     label = '0'
            # else:
            #     text_a = tokenization.convert_to_unicode(line[1])
            #     label = tokenization.convert_to_unicode(line[0])
            #     self.labels.add(label)
            text_a = tokenization.convert_to_unicode(line[1])
            label = tokenization.convert_to_unicode(line[0])
            self.labels.add(label)
            examples.append(
                InputExample(guid=guid, text_a=text_a, label=label, text_b=None)
            )
        return examples

def conver_single_example(ex_index, example, label_list, max_seq_length, tokenizer, mode):
    """
    将一个训练样本转化为InputFeature，其中进行字符seg并且index化,和label的index转化
    :param ex_index:
    :param example:
    :param label_list:
    :param max_seq_length:
    :param tokenizer:
    :return:
    """
    # 1. 构建label->id的映射
    label_map = {}
    if os.path.exists(os.path.join(FLAGS.output_dir, 'label2id.pkl')):
        with codecs.open(os.path.join(FLAGS.output_dir, 'label2id.pkl'), 'rb') as fd:
            label_map = pickle.load(fd)
    else:
        for i, label in enumerate(label_list):
            label_map[label] = i
        with codecs.open(os.path.join(FLAGS.output_dir, 'label2id.pkl'), 'wb') as fd:
            pickle.dump(label_map, fd)
    # 不考虑seq pair 分类的情况
    tokens_a = tokenizer.tokenize(example.text_a)

    # 截断，因为有句首和句尾的标识符
    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[0:(max_seq_length-2)]

    tokens = []
    segment_ids = []
    tokens.append('[CLS]')
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append('[SEP]')
    segment_ids.append(0)
    #将字符转化为id形式
    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    input_mask = [1]*len(input_ids)
    #补全到max_seg_length
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        segment_ids.append(0)
        input_mask.append(0)
    if example.label is None:
        label_id = -1
    else:
        label_id = label_map[example.label]
    if ex_index < 2 and mode in ['train', 'dev']:
        logger.info("*** Example ***")
        logger.info("guid: %s" % (example.guid))
        logger.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        logger.info("label: %s (id = %d)" % (example.label, label_id))

    feature = InputFeatures(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids, label_id=label_id)
    return feature

def file_based_convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, output_file, mode):
    """
    将训练文件转化特征后，存储为tf_record格式，用于模型的读取
    :param examples:
    :param label_list:
    :param max_seq_length:
    :param tokenizer:
    :param output_file:
    :return:
    """
    writer = tf.python_io.TFRecordWriter(path=output_file)
    # 将每一个样本转化为idx特征，封装到map中后进行序列化存储为record
    for ex_index, example in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info('Writing example %d of %d' %(ex_index, len(examples)))
        feature = conver_single_example(ex_index, example, label_list, max_seq_length, tokenizer, mode)
        # 将输入数据转化为64位int 的list，这是必须的
        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f
        features = collections.OrderedDict()
        features['input_ids'] = create_int_feature(feature.input_ids)
        features['input_mask'] = create_int_feature(feature.input_mask)
        features['segment_ids'] = create_int_feature(feature.segment_ids)
        features['label_ids'] = create_int_feature([feature.label_id])
        # 转化为Example 协议内存块
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())

def file_based_input_fn_builder(input_file, seq_length, num_label, is_training, drop_remainder):
    """

    :param input_file:
    :param seq_length:
    :param is_training:
    :param drop_remainder: 是否丢弃较小的batch
    :return:
    """
    name_to_features = {
      "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
      "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "label_ids": tf.FixedLenFeature([], tf.int64),
    }

    def _decode_record(record, name_to_feature):
        # 解析一个record中的数据
        example = tf.parse_single_example(record, name_to_feature)
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t
        return example

    def input_fn(params):
        """
        模型输入函数
        :param params:
        :return:
        """
        batch_size = params['batch_size']
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=200)

        # tf.data.experimental.map_and_batch will be deprecated, the replace methods like bellow
        d = d.apply(tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder
        ))
        # d = d.apply(lambda record: _decode_record(record, name_to_features))
        # d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
        return d
    return input_fn


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids, labels, num_labels):
    """

    :param bert_config:
    :param is_training:
    :param input_ids:
    :param input_mask:
    :param segment_ids:
    :param labels:
    :param num_labels:
    :param use_one_hot_embedding:
    :return:
    """
    # 通过传入的训练数据，进行representation
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
    )

    embedding_layer = model.get_sequence_output()
    output_layer = model.get_pooled_output()
    hidden_size = output_layer.shape[-1].value

    # model = CNN_Classification(embedding_chars=embedding_layer,
    #                                labels=labels,
    #                                num_tags=num_labels,
    #                                sequence_length=FLAGS.max_seq_length,
    #                                embedding_dims=embedding_layer.shape[-1].value,
    #                                vocab_size=0,
    #                                filter_sizes=[3, 4, 5],
    #                                num_filters=3,
    #                                dropout_keep_prob=FLAGS.dropout_keep_prob,
    #                                l2_reg_lambda=0.001)
    # loss, predictions, probabilities = model.add_cnn_layer()

    output_weights = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
        if is_training:
            # I.e., 0.1 dropout
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        probabilities = tf.nn.softmax(logits, axis=-1)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)
    return (loss, per_example_loss, logits, probabilities)

def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate, num_train_steps,
                     num_warmup_steps):
    """

    :param bert_config:
    :param num_labels:
    :param init_checkpoint:
    :param learning_rate:
    :param num_train_steps:
    :param num_warmup_steps:
    :param use_one_hot_embeddings:
    :return:
    """
    def model_fn(features, labels, mode, params):
        logger.info("*** Features ***")
        for name in sorted(features.keys()):
            logger.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (total_loss, per_example_loss, logits, probabilities) = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
            num_labels)

        # resort variable from checkpoint file to init current graph
        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        init_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names) = \
                modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
            #variables_to_restore = tf.contrib.framework.get_model_variables()
            #init_fn = tf.contrib.framework.\
               # assign_from_checkpoint_fn(init_checkpoint,
                #                          variables_to_restore,
                 #                         ignore_missing_vars=True)

        # 打印变量名称
        logger.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            logger.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)



        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu=False)
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op
            )
            #    training_hooks=[RestoreHook(init_fn)])
        elif mode == tf.estimator.ModeKeys.EVAL:
            def metric_fn(per_example_loss, label_ids, logits):
                predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                accuracy = tf.metrics.accuracy(label_ids, predictions)
                loss = tf.metrics.mean(per_example_loss)
                return {
                    "eval_accuracy": accuracy,
                    "eval_loss": loss,
                }
            eval_metrics = metric_fn(per_example_loss, label_ids, logits)
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metric_ops=eval_metrics
                #evaluation_hooks=[RestoreHook(init_fn)]
                )
        else:
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode, predictions=probabilities)
        return output_spec
    return model_fn

def main(_):
    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
        raise ValueError(
            "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)

    processor = ThuProcessor()
    #定义分词器
    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
    # estimator 运行参数
    run_config = tf.estimator.RunConfig(
        model_dir=FLAGS.output_dir,
        save_summary_steps=FLAGS.save_summary_steps,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        keep_checkpoint_max=5,
        log_step_count_steps=500,
        session_config=tf.ConfigProto(log_device_placement=True)
        #session_config=tf.ConfigProto(log_device_placement=True,
        #                               device_count={'GPU': 1}))
    )

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None
    if FLAGS.do_train:
        train_examples = processor.get_train_examples(FLAGS.data_dir)
        num_train_steps = int(
            len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    # get_labels() must be called after get_train_examoles or other examples
    label_list = processor.get_labels()
    logger.info('************ label_list=', ' '.join(label_list))
    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list),
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps)

    # params是一个dict 里面的key是model_fn 里面用到的参数名称，value是对应的数据
    params = {
        'batch_size': FLAGS.train_batch_size,
    }

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config,
        params=params,
    )

    if FLAGS.do_train:
        train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
        file_based_convert_examples_to_features(
            train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file, 'train')
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", FLAGS.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=FLAGS.max_seq_length,
            num_label=len(label_list),
            is_training=True,
            drop_remainder=True)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    if FLAGS.do_eval:
        eval_examples = processor.get_dev_examples(FLAGS.data_dir)
        eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
        file_based_convert_examples_to_features(
            eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file, 'eval')
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", FLAGS.eval_batch_size)

        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=FLAGS.max_seq_length,
            num_label=len(label_list),
            is_training=False,
            drop_remainder=False)

        result = estimator.evaluate(input_fn=eval_input_fn)

        output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    if FLAGS.do_predict:
        predict_examples = processor.get_test_examples(FLAGS.data_dir)
        predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
        file_based_convert_examples_to_features(predict_examples, label_list,
                                                FLAGS.max_seq_length, tokenizer,
                                                predict_file, 'test')

        logger.info("***** Running prediction*****")
        logger.info("  Num examples = %d", len(predict_examples))
        logger.info("  Batch size = %d", FLAGS.predict_batch_size)

        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=FLAGS.max_seq_length,
            num_label=len(label_list),
            is_training=False,
            drop_remainder=False)

        result = estimator.predict(input_fn=predict_input_fn)

        output_predict_file = os.path.join(FLAGS.output_dir, "test_results.txt")
        with tf.gfile.GFile(output_predict_file, "w") as writer:
            logger.info("***** Predict results *****")
            for prediction in result:
                output_line = "\t".join(
                    str(class_probability) for class_probability in prediction) + "\n"
                writer.write(output_line)


def load_data():
    processer = ThuProcessor()
    example = processer.get_train_examples(FLAGS.data_dir)
    print()


if __name__ == "__main__":
    # flags.mark_flag_as_required("data_dir")
    # flags.mark_flag_as_required("vocab_file")
    # flags.mark_flag_as_required("bert_config_file")
    # flags.mark_flag_as_required("output_dir")
    tf.app.run()
    # load_data()





