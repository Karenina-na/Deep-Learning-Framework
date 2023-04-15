import os
import torch
import numpy as np
from nltk import word_tokenize
import nltk
from collections import Counter
from torch.autograd import Variable
import torch.utils.data as Data


# download nltk data
# nltk.download()


class PrepareData(Data.Dataset):
    def __init__(self, train_file_path, dev_file_path):
        super(PrepareData, self).__init__()

        self.max_len_en = 0
        self.max_len_cn = 0

        self.start_word = '[BOS]'
        self.end_word = '[EOS]'
        self.pad_word = '[PAD]'
        self.unk_word = '[UNK]'

        # 读取数据 并分词
        self.train_en, self.train_cn = self.load_data(train_file_path)
        self.dev_en, self.dev_cn = self.load_data(dev_file_path)

        # 构建单词表
        self.en_word_dict, self.en_total_words, self.en_index_dict = self.build_dict(self.train_en)
        self.cn_word_dict, self.cn_total_words, self.cn_index_dict = self.build_dict(self.train_cn)

        # id化
        self.train_en, self.train_cn = self.wordToID(self.train_en, self.train_cn, self.en_word_dict, self.cn_word_dict)
        self.dev_en, self.dev_cn = self.wordToID(self.dev_en, self.dev_cn, self.en_word_dict, self.cn_word_dict)

        self.src_vocab_size = len(self.en_word_dict)
        self.tgt_vocab_size = len(self.cn_word_dict)

        self.start_token_id = self.en_word_dict[self.start_word]
        self.end_token_id = self.en_word_dict[self.end_word]
        self.pad_token_id = self.en_word_dict[self.pad_word]


    def __len__(self):
        return self.train_en.__len__()

    def __getitem__(self, idx):
        return self.train_en[idx], self.train_cn[idx]

    def load_data(self, path):
        en = []
        cn = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip().split('\t')
                # 去除行尾的空白符，然后使用制表符(\t)将行分割为两个部分，即英文和中文。
                # 将英文部分转换为小写，并使用nltk库中的word_tokenize函数将其分词。
                # 在分词的结果的前后分别添加BOS和EOS标记，分别表示该句子的开始和结束。
                en.append([self.start_word] + word_tokenize(line[0].lower()) + [self.end_word])
                cn.append([self.start_word] + word_tokenize(" ".join([w for w in line[1]])) + [self.end_word])
        # [['BOS', 'i', 'am', 'a', 'student', '.', 'EOS'], ...]
        # [['BOS', '我', '是', '一名', '学生', '。', 'EOS'], ...]
            # 找到最大的句子长度
            self.max_len_en = max([len(s) for s in en])
            self.max_len_cn = max([len(s) for s in cn])
            # 填充[PAD]，使得每个句子的长度都相同
            for i in range(len(en)):
                en[i] = en[i] + [self.pad_word] * (self.max_len_en - len(en[i]))
            for i in range(len(cn)):
                cn[i] = cn[i] + [self.pad_word] * (self.max_len_cn - len(cn[i]))
        return en, cn

    def build_dict(self, sentences, max_words=50000):
        word_count = Counter()

        for sentence in sentences:
            for s in sentence:
                word_count[s] += 1  # 统计词频

        ls = word_count.most_common(max_words)  # 选取词频最高的前max_words个词
        total_words = len(ls) + 1  # 词典大小+2，因为有两个特殊符号UNK和PAD

        # {key: value} = {word: frequency}
        word_dict = {w[0]: index for index, w in enumerate(ls)}
        word_dict[self.unk_word] = len(word_dict)  # 未知词

        # {key: value} = {frequency: word} # 按照词频从大到小编码
        index_dict = {v: k for k, v in word_dict.items()}

        return word_dict, total_words, index_dict

    @staticmethod
    def wordToID(en, cn, en_dict, cn_dict, sort=True):
        length = len(en)

        # 将英文和中文分别转换为id，id的值为单词在词典中的索引

        out_en_ids = [[en_dict.get(w, 0) for w in sent] for sent in en]
        out_cn_ids = [[cn_dict.get(w, 0) for w in sent] for sent in cn]

        # sort sentences by english lengths
        def len_argsort(seq):
            return sorted(range(len(seq)), key=lambda x: len(seq[x]))

        # 把中文和英文按照同样的顺序排序
        if sort:
            sorted_index = len_argsort(out_en_ids)
            out_en_ids = [out_en_ids[i] for i in sorted_index]
            out_cn_ids = [out_cn_ids[i] for i in sorted_index]

        # 转换为tensor
        out_en_ids = torch.tensor(out_en_ids, dtype=torch.long)
        out_cn_ids = torch.tensor(out_cn_ids, dtype=torch.long)

        return out_en_ids, out_cn_ids


if __name__ == "__main__":
    train_file_path = "../Data/translate_chinese_to_english_simple/train.txt"
    dev_file_path = "../Data/translate_chinese_to_english_simple/dev.txt"
    data = PrepareData(train_file_path, dev_file_path)
    # print(data.train_en[0])
    # print(data.train_cn[0])
    # print(data.en_word_dict)
    # print(data.cn_word_dict)
    # print(data.en_total_words)
    # print(data.cn_total_words)
    # print(data.en_index_dict)
    # print(data.cn_index_dict)
    # print(data.src_vocab_size)
    # print(data.tgt_vocab_size)
    # print(data.pad_token_id)
    # print(data.start_token_id)
    # print(data.end_token_id)
    # print(data.max_len_en)
    # print(data.max_len_cn)
