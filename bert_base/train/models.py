# -*- coding: utf-8 -*-

"""
 一些公共模型代码
 @Time    : 2019/1/30 12:46
 @Author  : MaCan (ma_cancan@163.com)
 @File    : models.py
"""

from bert_base.train.lstm_crf_layer import BLSTM_CRF
from tensorflow.contrib.layers.python.layers import initializers


__all__ = ['InputExample', 'InputFeatures', 'decode_labels', 'create_model', 'convert_id_str',
           'convert_id_to_label', 'result_to_json', 'create_classification_model']

class Model(object):
    def __init__(self, *args, **kwargs):
        pass


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid=None, text=None, label=None):
        """Constructs a InputExample.
        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        self.label = label

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids, ):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        # self.label_mask = label_mask


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()


def create_model(bert_config, is_training, input_ids, input_mask,
                 segment_ids, labels, num_labels, use_one_hot_embeddings,
                 dropout_rate=1.0, lstm_size=1, cell='lstm', num_layers=1):
    """
    创建X模型
    :param bert_config: bert 配置
    :param is_training:
    :param input_ids: 数据的idx 表示
    :param input_mask:
    :param segment_ids:
    :param labels: 标签的idx 表示
    :param num_labels: 类别数量
    :param use_one_hot_embeddings:
    :return:
    """
    # 使用数据加载BertModel,获取对应的字embedding
    import tensorflow as tf
    from bert_base.bert import modeling
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings
    )
    # 获取对应的embedding 输入数据[batch_size, seq_length, embedding_size]
    embedding = model.get_sequence_output()
    max_seq_length = embedding.shape[1].value
    # 算序列真实长度
    used = tf.sign(tf.abs(input_ids))
    lengths = tf.reduce_sum(used, reduction_indices=1)  # [batch_size] 大小的向量，包含了当前batch中的序列长度
    # 添加CRF output layer
    blstm_crf = BLSTM_CRF(embedded_chars=embedding, hidden_unit=lstm_size, cell_type=cell, num_layers=num_layers,
                          dropout_rate=dropout_rate, initializers=initializers, num_labels=num_labels,
                          seq_length=max_seq_length, labels=labels, lengths=lengths, is_training=is_training)
    rst = blstm_crf.add_blstm_crf_layer(crf_only=True)
    return rst


def create_classification_model(bert_config, is_training, input_ids, input_mask, segment_ids, labels, num_labels):
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
    import tensorflow as tf
    from bert_base.bert import modeling
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

    # predict = CNN_Classification(embedding_chars=embedding_layer,
    #                                labels=labels,
    #                                num_tags=num_labels,
    #                                sequence_length=FLAGS.max_seq_length,
    #                                embedding_dims=embedding_layer.shape[-1].value,
    #                                vocab_size=0,
    #                                filter_sizes=[3, 4, 5],
    #                                num_filters=3,
    #                                dropout_keep_prob=FLAGS.dropout_keep_prob,
    #                                l2_reg_lambda=0.001)
    # loss, predictions, probabilities = predict.add_cnn_layer()

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

        if labels is not None:
            one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

            per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
            loss = tf.reduce_mean(per_example_loss)
        else:
            loss, per_example_loss = None, None
    return (loss, per_example_loss, logits, probabilities)


def decode_labels(labels, batch_size):
    new_labels = []
    for row in range(batch_size):
        label = []
        for i in labels[row]:
            i = i.decode('utf-8')
            if i == '**PAD**':
                break
            if i in ['[CLS]', '[SEP]']:
                continue
            label.append(i)
        new_labels.append(label)
    return new_labels


def convert_id_str(input_ids, batch_size):
    res = []
    for row in range(batch_size):
        line = []
        for i in input_ids[row]:
            i = i.decode('utf-8')
            if i == '**PAD**':
                break
            if i in ['[CLS]', '[SEP]']:
                continue

            line.append(i)
        res.append(line)
    return res


def convert_id_to_label(pred_ids_result, idx2label, batch_size):
    """
    将id形式的结果转化为真实序列结果
    :param pred_ids_result:
    :param idx2label:
    :return:
    """
    result = []
    index_result = []
    for row in range(batch_size):
        curr_seq = []
        curr_idx = []
        ids = pred_ids_result[row]
        for idx, id in enumerate(ids):
            if id == 0:
                break
            curr_label = idx2label[id]
            if curr_label in ['[CLS]', '[SEP]']:
                if id == 102 and (idx < len(ids) and ids[idx + 1] == 0):
                    break
                continue
            # elif curr_label == '[SEP]':
            #     break
            curr_seq.append(curr_label)
            curr_idx.append(id)
        result.append(curr_seq)
        index_result.append(curr_idx)
    return result, index_result


def result_to_json(self, string, tags):
    """
    将模型标注序列和输入序列结合 转化为结果
    :param string: 输入序列
    :param tags: 标注结果
    :return:
    """
    item = {"entities": []}
    entity_name = ""
    entity_start = 0
    idx = 0
    last_tag = ''

    for char, tag in zip(string, tags):
        if tag[0] == "S":
            self.append(char, idx, idx+1, tag[2:])
            item["entities"].append({"word": char, "start": idx, "end": idx+1, "type":tag[2:]})
        elif tag[0] == "B":
            if entity_name != '':
                self.append(entity_name, entity_start, idx, last_tag[2:])
                item["entities"].append({"word": entity_name, "start": entity_start, "end": idx, "type": last_tag[2:]})
                entity_name = ""
            entity_name += char
            entity_start = idx
        elif tag[0] == "I":
            entity_name += char
        elif tag[0] == "O":
            if entity_name != '':
                self.append(entity_name, entity_start, idx, last_tag[2:])
                item["entities"].append({"word": entity_name, "start": entity_start, "end": idx, "type": last_tag[2:]})
                entity_name = ""
        else:
            entity_name = ""
        entity_start = idx
        idx += 1
        last_tag = tag
    if entity_name != '':
        self.append(entity_name, entity_start, idx, last_tag[2:])
        item["entities"].append({"word": entity_name, "start": entity_start, "end": idx, "type": last_tag[2:]})
    return item
