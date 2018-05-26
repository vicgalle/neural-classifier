# Requirements for the Lovelace HPC cluster

First we obtain the code
```git clone ssh://vcs-user@phab.icmat-datalab.es:2222/diffusion/18/legalnlp.git```
```cd legalnlp```

We load a suitable python installation
```module load python/anaconda/5.0.1```

Optionally, we may create a new conda environment
```conda create --name legalNLP```
```source activate legalNLP```

We install the two dependencies
```pip install --user http://download.pytorch.org/whl/cu75/torch-0.2.0.post3-cp36-cp36m-manylinux1_x86_64.whl```
```pip install --user torchtext```

Then, we may train or test some dataset, please see README.md. Executions can be launched via
```qsub -q gpu.q -V -b y -N gpu -l h_vmem=32G -cwd "python main.py"``` (see arguments in the readme or src/lstm/main.py).
