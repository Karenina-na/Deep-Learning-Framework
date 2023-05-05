from torchinfo import summary
import torch


def summary_info(net, input_data, batch_dim=0,
                 col_names=("input_size", "output_size", "num_params",
                            "kernel_size", "mult_adds"), depth=4, verbose=2):
    """
    :param net: model
    :param input_data: input data
    :param batch_dim: batch dimension
    :param col_names: column names
    :param depth: depth of summary
    :param verbose: verbose
    :return: None
    """
    summary(model=net, input_data=input_data, batch_dim=batch_dim,
            col_names=col_names, depth=depth, verbose=verbose)


if __name__ == '__main__':
    from Models.Transformer.Transformer import Transformer

    # Transformer Parameters
    d_model = 512  # Embedding Size
    d_ff = 2048  # FeedForward dimension
    d_k = d_v = 64  # dimension of K(=Q), V
    n_layers = 6  # number of Encoder of Decoder Layer
    n_heads = 8  # number of heads in Multi-Head Attention
    lr = 0.001  # learning rate

    model = Transformer(10, 20, d_model, n_layers, d_ff, d_k, d_v, n_heads)
    # input data
    src = torch.randint(0, 10, (3, 5))
    tgt = torch.randint(0, 20, (3, 5))
    summary_info(model, input_data=[src, tgt], batch_dim=0)
