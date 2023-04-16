import torch
import torch.nn as nn
import torch.optim as optim
from Models.Transformer.Transformer import Transformer, greedy_decoder
from DataProcess.TranslateSimpleNLP import PrepareData
from torch.utils.data import DataLoader
import pandas as pd

# Transformer Parameters
d_model = 512  # Embedding Size
d_ff = 2048  # FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q), V
n_layers = 6  # number of Encoder of Decoder Layer
n_heads = 8  # number of heads in Multi-Head Attention
lr = 0.001  # learning rate
BatchSize = 256  # Batch size
Epoch = 500  # Epoch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train():
    # 加载数据集
    train_file_path = "../../Data/translate_chinese_to_english_simple/train.txt"
    dev_file_path = "../../Data/translate_chinese_to_english_simple/dev.txt"
    data = PrepareData(train_file_path, dev_file_path)
    dataLoader = DataLoader(data, batch_size=BatchSize, shuffle=True, num_workers=10)

    # 构建模型
    model = Transformer(data.src_vocab_size, data.tgt_vocab_size, d_model, n_layers, d_ff, d_k, d_v, n_heads).to(device)
    model.train()
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.99)

    print("source: ", data.src_vocab_size)
    print("target: ", data.tgt_vocab_size)

    print("training start >>>>>\n")

    for epoch in range(Epoch):
        for (enc_inputs, dec_inputs) in dataLoader:
            # enc_inputs, dec_inputs, dec_outputs = enc_inputs.to(device), dec_inputs.to(device), dec_outputs.to(device)
            # outputs: [batch_size * tgt_len, tgt_vocab_size]
            enc_inputs, dec_inputs = enc_inputs.to(device), dec_inputs.to(device)
            outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
            loss = criterion(outputs, dec_inputs.view(-1))
            print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # end of epoch
    print("training end <<<<<\n")

    # 保存模型
    torch.save(model.state_dict(), "../../Result/checkpoints/transformer.pth")
    size = pd.DataFrame([data.src_vocab_size, data.tgt_vocab_size,
                         data.start_token_id, data.end_token_id, data.pad_token_id,
                         data.max_len_en, data.max_len_cn],
                        index=["src_vocab_size", "tgt_vocab_size",
                               "start_token_id", "end_token_id", "pad_token_id",
                               "src_sentence_max_len", "tgt_sentence_max_len"])
    size.to_csv("../../Result/checkpoints/transformer_size.csv")
    en_word_dict = pd.DataFrame([[k, v] for k, v in data.en_word_dict.items()])
    en_word_dict.to_csv("../../Result/checkpoints/en_word_dict.csv")
    ch_word_dict = pd.DataFrame([[k, v] for k, v in data.cn_word_dict.items()])
    ch_word_dict.to_csv("../../Result/checkpoints/ch_word_dict.csv")


def test():
    # 加载模型
    print("loading Data >>>>>\n")
    size = pd.read_csv("../../Result/checkpoints/transformer_size.csv")
    src_vocab_size = int(size.iloc[0, 1])
    tgt_vocab_size = int(size.iloc[1, 1])
    start_token_id = int(size.iloc[2, 1])  # src_start_token_id == tgt_start_token_id
    end_token_id = int(size.iloc[3, 1])  # src_end_token_id == tgt_end_token_id
    pad_token_id = int(size.iloc[4, 1])  # src_pad_token_id == tgt_pad_token_id
    src_sentence_max_len = int(size.iloc[5, 1])
    tgt_sentence_max_len = int(size.iloc[6, 1])

    # {word: id}
    enc_word_dict = pd.read_csv("../../Result/checkpoints/en_word_dict.csv", index_col=2).to_dict()['0']
    enc_word_dict = {value: key for key, value in enc_word_dict.items()}

    # {id: word}
    dec_word_dict = pd.read_csv("../../Result/checkpoints/ch_word_dict.csv", index_col=2).to_dict()['0']

    # 测试句子
    sentences = [
        ["I", "am", "a", "Chinese", "."],
        ["I", "love", "you", "."],
        ["I", "love", "Chinese", "people"],
        ["I", "love", "Chinese", "people", "."],
        ["I", "love", "Chinese", "people", "."],
        ["I", "love", "Chinese", "people", ".", "How", "about", "you", "?"],
    ]
    dec_sentences = sentences
    sentences = [[start_token_id] + [enc_word_dict.get(word.lower(), 0) for word in sentence] + [end_token_id]
                 for sentence in sentences]
    sentences = [sentence + [0] * (src_sentence_max_len - len(sentence)) for sentence in sentences]
    sentences = torch.tensor(sentences, dtype=torch.long).to(device)

    # 加载模型
    print("loading Model >>>>>\n")
    model = Transformer(src_vocab_size, tgt_vocab_size, d_model, n_layers, d_ff, d_k, d_v, n_heads)
    model.load_state_dict(torch.load("../../Result/checkpoints/transformer.pth"))
    model.to(device)
    model.eval()

    # 生成句子
    print("generating sentences >>>>>\n")
    for i in range(len(sentences)):
        greedy_dec_input = greedy_decoder(model, sentences[i].view(1, -1),
                                          src_start_symbol=start_token_id, tgt_end_symbol=end_token_id,
                                          device=device)
        predict, _, _, _ = model(sentences[i].view(1, -1), greedy_dec_input)
        predict = predict.data.max(1, keepdim=True)[1]
        print(dec_sentences[i], '->', [dec_word_dict[n.item()] for n in predict.squeeze()])


if __name__ == "__main__":
    train()
    # test()
