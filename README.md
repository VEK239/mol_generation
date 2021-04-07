# Junction Tree Variational Autoencoder for Molecular Graph Generation. Usage example

Official implementation of Junction Tree Variational Autoencoder [https://arxiv.org/abs/1802.04364](https://arxiv.org/abs/1802.04364)

# Requirements
* Linux (We only tested on Ubuntu)
* RDKit (version >= 2017.09)
* Python (version == 2.7)
* PyTorch (version >= 0.2)

To install RDKit, please follow the instructions here [http://www.rdkit.org/docs/Install.html](http://www.rdkit.org/docs/Install.html)

We highly recommend you to use conda for package management.

# Quick Start
The following directories provides scripts for the experiments in original ICML paper
* `molvae/` includes scripts for training our VAE model only. Please read `molvae/README.md` for training our VAE model.
* `jtnn/` contains codes for model formulation.
* `notebooks/mol_generation_example.ipynb` contains the example of molecule generation and generated molecules summary.
