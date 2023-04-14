import torch
import torch.nn as nn
import torch.optim as optim
from Models.Transformer.Transformer import Transformer
from DataProcess.TranslateSimpleNLP import PrepareData
from torch.utils.data import DataLoader

# Transformer Parameters
d_model = 512  # Embedding Size
d_ff = 2048  # FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q), V
n_layers = 3  # number of Encoder of Decoder Layer
n_heads = 8  # number of heads in Multi-Head Attention
lr = 0.001  # learning rate
BatchSize = 256  # Batch size
Epoch = 300  # Epoch

if __name__ == "__main__":
    # 加载数据集
    train_file_path = "../../Data/translate_chinese_to_english_simple/train.txt"
    dev_file_path = "../../Data/translate_chinese_to_english_simple/dev.txt"
    data = PrepareData(train_file_path, dev_file_path)
    dataLoader = DataLoader(data, batch_size=BatchSize, shuffle=True, num_workers=10)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 构建模型
    model = Transformer(data.src_vocab_size, data.tgt_vocab_size, d_model, n_layers, d_ff, d_k, d_v, n_heads).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.99)

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

    # 保存模型
    torch.save(model.state_dict(), "../../Result/checkpoints/transformer.pth")