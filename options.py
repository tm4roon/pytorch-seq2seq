# -*- coding: utf-8 -*-


def train_opts(parser):
    group = parser.add_argument_group('Training')
    group.add_argument('--train', default='./sample_data/sample_train.tsv',
        help='path to a train data')
    group.add_argument('--valid', default='./sample_data/sample_valid.tsv',
        help='path to a validation data')
    group.add_argument('--batch-size', type=int, default=32, 
        help='batch size')
    group.add_argument('--savedir', default='./checkpoints', 
        help='path to save models')
    group.add_argument('--max-epoch', type=int, default=0, 
        help='number of epochs')
    group.add_argument('--max-update', type=int, default=0,
        help='number of updates')
    group.add_argument('--lr', type=float, default=0.25,
        help='learning rate')
    group.add_argument('--min-lr', type=float, default=1e-5, 
        help='minimum learning rate')
    group.add_argument('--clip', type=float, default=0.1,
        help='gradient cliping')
    group.add_argument('--tf-ratio', type=float, default=0.5,
        help='teaching force ratio')
    group.add_argument('--gpu', action='store_true',
        help='whether gpu is used')
    return group


def translate_opts(parser):
    group = parser.add_argument_group('Translation')
    group.add_argument('--model', default='./checkpoints/checkpoint_best.pt',
        help='model file for translation')
    group.add_argument('--input', default='./sample_data/sample_test.txt',
        help='input file')
    group.add_argument('--batch-size', type=int, default=32,
        help='batch size')
    group.add_argument('--maxlen', type=int, default=100,
        help='maximum length of output sentence')
    group.add_argument('--gpu', action='store_true',
        help='whether gpu is used')
    return group
    

def model_opts(parser):
    group = parser.add_argument_group('Model\'s hyper-parameters')
    group.add_argument('--embed-dim', type=int, default=200,
        help='dimension of word embeddings')
    group.add_argument('--src_min-freq', type=int, default=0,
        help='''map words of source side appearing less than 
                threshold times to unknown''')
    group.add_argument('--tgt_min-freq', type=int, default=0,
        help='''map words of target side appearing less than
              threshold times to unknown''')
    group.add_argument('--rnn', choices=['lstm'], default='lstm',
        help='rnn\'s architechture')
    group.add_argument('--hidden-dim', type=int, default=1024,
        help='number of hidden units per layer')
    group.add_argument('--n-layers', type=int, default=2,
        help='number of LSTM layers')
    group.add_argument('--bidirectional', action='store_true',
        help='whether use bidirectional LSTM for encoder')
    group.add_argument('--attn', choices=['dot', 'general', 'concat'],
        default='dot', help='attention type')
    group.add_argument('--dropout', type=float, default=0.2,
        help='dropout applied to layers (0 means no dropout)')
    group.add_argument('--tied', action='store_true',
        help='tie the word embedding and softmax weight')
    return group

