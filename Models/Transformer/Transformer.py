import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
import numpy as np
from Models.Transformer.Coder.Encoder import Encoder
from Models.Transformer.Coder.Decoder import Decoder


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, n_layers,
                 d_ff, d_k, d_v, n_heads):
        super(Transformer, self).__init__()
        self.encoder = Encoder(src_vocab_size, d_model, n_layers, d_ff, d_k, d_v, n_heads)
        self.decoder = Decoder(tgt_vocab_size, d_model, n_layers, d_ff, d_k, d_v, n_heads)
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False)

    def forward(self, enc_inputs, dec_inputs):
        """
        enc_inputs: [batch_size, src_len]
        dec_inputs: [batch_size, tgt_len]
        """
        # tensor to store decoder outputs
        # outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)

        # enc_outputs: [batch_size, src_len, d_model], enc_self_attns: [n_layers, batch_size, n_heads, src_len, src_len]
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attns: [n_layers, batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [n_layers, batch_size, tgt_len, src_len]
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)
        dec_logits = self.projection(dec_outputs)  # dec_logits: [batch_size, tgt_len, tgt_vocab_size]
        return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns


def greedy_decoder(model, enc_input, src_start_symbol, tgt_end_symbol, device, max_len=1000):
    """
    For simplicity, a Greedy Decoder is Beam search when K=1. This is necessary for inference as we don't know the
    target sequence input. Therefore we try to generate the target input word by word, then feed it into the transformer.
    Starting Reference: http://nlp.seas.harvard.edu/2018/04/03/attention.html#greedy-decoding
    :param model: Transformer Model
    :param enc_input: The encoder input
    :param src_start_symbol: The start symbol. In this example it is 'S' which corresponds to index 4
    :param tgt_end_symbol: The target word symbol. In this example it is '.' which corresponds to index 8
    :param device : CUDA or CPU
    :param max_len : max seq long
    :return: The target input
    """
    enc_outputs, enc_self_attns = model.encoder(enc_input)
    dec_input = torch.zeros(1, 0).type_as(enc_input.data)
    terminal = False
    next_symbol = src_start_symbol
    while not terminal and max_len > 0:
        dec_input = torch.cat([dec_input.detach(), torch.tensor([[next_symbol]], dtype=enc_input.dtype).to(device)], -1)
        dec_outputs, _, _ = model.decoder(dec_input, enc_input, enc_outputs)
        projected = model.projection(dec_outputs)
        prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
        next_word = prob.data[-1]
        next_symbol = next_word
        if next_symbol == tgt_end_symbol:
            terminal = True
        max_len -= 1
    return dec_input


if __name__ == "__main__":
    from MakeData_Test import LoadData, LoadTestData

    loader, src_vocab_size, tgt_vocab_size = LoadData()

    # Transformer Parameters
    d_model = 512  # Embedding Size
    d_ff = 2048  # FeedForward dimension
    d_k = d_v = 64  # dimension of K(=Q), V
    n_layers = 6  # number of Encoder of Decoder Layer
    n_heads = 8  # number of heads in Multi-Head Attention
    lr = 0.001  # learning rate

    model = Transformer(src_vocab_size, tgt_vocab_size, d_model, n_layers, d_ff, d_k, d_v, n_heads)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # [PAD] 忽略
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.99)

    print(model)
    print("training start")

    for epoch in range(30):
        for enc_inputs, dec_inputs, dec_outputs in loader:
            '''
            enc_inputs: [batch_size, [src_sentence]]
            dec_inputs: [batch_size, [tgt_sentence]
            dec_outputs: [batch_size, [tgt_sentence]
            '''
            # enc_inputs, dec_inputs, dec_outputs = enc_inputs.to(device), dec_inputs.to(device), dec_outputs.to(device)
            # outputs: [batch_size * tgt_len, tgt_vocab_size]
            outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
            loss = criterion(outputs, dec_outputs.view(-1))
            print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Test
    idx2word, tgt_vocab = LoadTestData()

    enc_inputs, _, _ = next(iter(loader))
    for i in range(len(enc_inputs)):
        greedy_dec_input = greedy_decoder(model, enc_inputs[i].view(1, -1),
                                          src_start_symbol=tgt_vocab["S"], tgt_end_symbol=tgt_vocab["."],
                                          device=torch.device("cpu"))
        predict, _, _, _ = model(enc_inputs[i].view(1, -1), greedy_dec_input)
        predict = predict.data.max(1, keepdim=True)[1]
        print(enc_inputs[i], '->', [idx2word[n.item()] for n in predict.squeeze()])
