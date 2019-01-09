# encoding=utf-8

"""
用于语料库的处理
1. 全部处理成小于max_seq_length的序列，这样可以避免解码出现不合法的数据或者在最后算结果的时候出现out of range 的错误。

@Author: Macan
"""


import os
import codecs
import argparse

def load_file(file_path):
    if not os.path.exists(file_path):
        return None
    with codecs.open(file_path, 'r', encoding='utf-8') as fd:
        for line in fd:
            yield line


def _cut(sentence):
    new_sentence = []
    sen = []
    for i in sentence:
        if i.split(' ')[0] in ['。', '！', '？'] and len(sen) != 0:
            sen.append(i)
            new_sentence.append(sen)
            sen = []
            continue
        sen.append(i)
    if len(new_sentence) == 1: #娄底那种一句话超过max_seq_length的且没有句号的，用,分割，再长的不考虑了。。。
        new_sentence = []
        sen = []
        for i in sentence:
            if i.split(' ')[0] in ['，'] and len(sen) != 0:
                sen.append(i)
                new_sentence.append(sen)
                sen = []
                continue
            sen.append(i)
    return new_sentence


def cut_sentence(file, max_seq_length):
    """
    句子截断
    :param file: 
    :param max_seq_length: 
    :return: 
    """
    context = []
    sentence = []
    cnt = 0
    for line in load_file(file):
        line = line.strip()
        if line == '' and len(sentence) != 0:
            # 判断这一句是否超过最大长度
            if len(sentence) > max_seq_length:
                sentence = _cut(sentence)
                context.extend(sentence)
            else:
                context.append(sentence)
            sentence = []
            continue
        cnt += 1
        sentence.append(line)
    print('token cnt:{}'.format(cnt))
    return context

def write_to_file(file, context):
    # 首先将源文件改名为新文件名，避免覆盖
    os.rename(file, '{}.bak'.format(file))
    with codecs.open(file, 'w', encoding='utf-8') as fd:
        for sen in context:
            for token in sen:
                fd.write(token + '\n')
            fd.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='data pre process')
    parser.add_argument('--train_data', type=str, default='./NERdata/train.txt')
    parser.add_argument('--dev_data', type=str, default='./NERdata/dev.txt')
    parser.add_argument('--test_data', type=str, default='./NERdata/test.txt')
    parser.add_argument('--max_seq_length', type=int, default=126)
    args = parser.parse_args()

    print('cut train data to max sequence length:{}'.format(args.max_seq_length))
    context = cut_sentence(args.train_data, args.max_seq_length)
    write_to_file(args.train_data, context)

    print('cut dev data to max sequence length:{}'.format(args.max_seq_length))
    context = cut_sentence(args.dev_data, args.max_seq_length)
    write_to_file(args.dev_data, context)

    print('cut test data to max sequence length:{}'.format(args.max_seq_length))
    context = cut_sentence(args.test_data, args.max_seq_length)
    write_to_file(args.test_data, context)