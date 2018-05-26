#! /usr/bin/env python
import os
import argparse
import datetime
import torch
import torchtext.data as data
import torchtext.datasets as datasets
import model
import train
import mydatasets

parser = argparse.ArgumentParser(description='Text classifier')

# Model hyperparameters
parser.add_argument('-lr', type=float, default=3e-4, help='initial learning rate [default: 3e-4]')
parser.add_argument('-epochs', type=int, default=256, help='number of epochs for train [default: 256]')
parser.add_argument('-batch-size', type=int, default=8, help='batch size for training [default: 8]')
parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
parser.add_argument('-max-norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
parser.add_argument('-use_pretrain', action='store_true', default=False, help='use pretrained embeddings' )
parser.add_argument('-use_att', action='store_true', default=False, help='use attention layer' )
parser.add_argument('-Hd', type=int, default=200, help='latent dimension of LSTM [default: 200]')

# pretrained embeddings (see -use_pretrain in the previous group)
parser.add_argument('-vector_cache', type=str, default=os.path.join(os.getcwd(), '.vector_cache/input_vectors.pt'))
parser.add_argument('-word_vectors', type=str, default='fasttext.en') 
parser.add_argument('-d_embed', type=int, default=300)
parser.add_argument('-data_cache', type=str, default=os.path.join(os.getcwd(), '.data_cache'))

# device
parser.add_argument('-device', type=int, default=-1, help='device to use for iterate data, -1 mean cpu [default: -1]')
parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu' )

# logistics
parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
parser.add_argument('-predict', type=str, default=None, help='predict the sentence given')
parser.add_argument('-test', action='store_true', default=False, help='train or test')
parser.add_argument('-log-interval',  type=int, default=1,   help='how many steps to wait before logging training status [default: 1]')
parser.add_argument('-test-interval', type=int, default=100, help='how many steps to wait before testing [default: 100]')
parser.add_argument('-save-interval', type=int, default=100, help='how many steps to wait before saving [default:100]')
parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
parser.add_argument('-shuffle', action='store_true', default=False, help='shuffle the data every epoch' )

args = parser.parse_args()


def legal(text_field, label_field, **kargs):
    l = mydatasets.Legal(text_field, label_field)
    train_data, dev_data, tst_data = l.splits(text_field, label_field)
    text_field.build_vocab(train_data, dev_data, tst_data, max_size=10000)

    # We load pretrained embeddings
    if args.word_vectors:
        if os.path.isfile(args.vector_cache):
            text_field.vocab.vectors = torch.load(args.vector_cache)
        else:
            text_field.vocab.load_vectors(args.word_vectors + '.' + str(args.d_embed) + 'd')
            os.makedirs(os.path.dirname(args.vector_cache), exist_ok=True)
            torch.save(text_field.vocab.vectors, args.vector_cache)

    label_field.build_vocab(train_data, dev_data)
    train_iter, dev_iter, tst_iter = data.Iterator.splits(
                                (train_data, dev_data, tst_data), 
                                batch_sizes=(args.batch_size, args.batch_size, args.batch_size),
                                **kargs)
    return train_iter, dev_iter, tst_iter


# Load data
print("\nLoading data...")
text_field = data.Field(lower=True)
label_field = data.Field(sequential=False)
train_iter, dev_iter, tst_iter = legal(text_field, label_field, device=-1, repeat=False)

# Update args and print
args.embed_num = len(text_field.vocab)
args.class_num = len(label_field.vocab) - 1
args.vectors = text_field.vocab.vectors
args.cuda = (not args.no_cuda) and torch.cuda.is_available(); del args.no_cuda
args.save_dir = os.path.join(args.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

print("\nParameters:")
for attr, value in sorted(args.__dict__.items()):
    print("\t{}={}".format(attr.upper(), value))

args.TF = text_field

# Model creation and load
if args.snapshot is None:
    net = model.LSTM_Text(args)
else :
    print('\nLoading model from [%s]...' % args.snapshot)
    try:
        #net = torch.load(args.snapshot)  Provocaba problemas al predecir
        net = model.LSTM_Text(args)
        net.load_state_dict(torch.load(args.snapshot))
    except :
        print("Sorry, This snapshot doesn't exist."); exit()

if args.cuda:
    net = net.cuda()
        
# Call the main function (train.py)
if args.predict is not None:
    label = train.predict(args.predict, net, text_field, label_field, args.cuda, args)
    print('\n[Text]  {}\n[Label] {}\n'.format(args.predict, label))
elif args.test :
    try:
        train.eval(test_iter, net, args) 
    except Exception as e:
        print("\nSorry. The test dataset doesn't  exist.\n")
else :
    print()
    try:
        train.train(train_iter, dev_iter, tst_iter, net, label_field, args)
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')