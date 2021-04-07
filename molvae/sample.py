import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.autograd import Variable

import math, random, sys
from optparse import OptionParser
from collections import deque
import rdkit
import rdkit.Chem as Chem
from rdkit.Chem import Draw


from jtnn.mol_tree import Vocab
from jtnn.jtnn_vae import JTNNVAE


def sample(hidden_size, latent_size, depth, nsample, stereo, vocab_path, model_path):
    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    vocab = [x.strip("\r\n ") for x in open(vocab_path)]
    vocab = Vocab(vocab)

    # hidden_size = int(opts.hidden_size)
    # latent_size = int(opts.latent_size)
    # depth = int(opts.depth)
    # nsample = int(opts.nsample)
    # stereo = True if int(opts.stereo) == 1 else False

    model = JTNNVAE(vocab, hidden_size, latent_size, depth, stereo=stereo)
    load_dict = torch.load(model_path, map_location=torch.device('cpu'))
    missing = {k: v for k, v in model.state_dict().items() if k not in load_dict}
    load_dict.update(missing)
    model.load_state_dict(load_dict)
    # model = model.cuda()

    torch.manual_seed(0)
    result = []
    for i in range(nsample):
        sampled = model.sample_prior(prob_decode=False)
        result.append(sampled)
        print(sampled)
    return result


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-n", "--nsample", dest="nsample")
    parser.add_option("-v", "--vocab", dest="vocab_path")
    parser.add_option("-m", "--model", dest="model_path")
    parser.add_option("-w", "--hidden", dest="hidden_size", default=200)
    parser.add_option("-l", "--latent", dest="latent_size", default=56)
    parser.add_option("-d", "--depth", dest="depth", default=3)
    parser.add_option("-e", "--stereo", dest="stereo", default=1)
    opts, args = parser.parse_args()
    print(opts, args)
    sample(
        hidden_size=int(opts.hidden_size),
        latent_size=int(opts.latent_size),
        depth=int(opts.depth),
        nsample=int(opts.nsample),
        stereo=True if int(opts.stereo) == 1 else False,
        vocab_path=opts.vocab_path,
        model_path=opts.model_path
    )
