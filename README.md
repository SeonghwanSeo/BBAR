# Molecular generative model via retrosynthetically prepared chemical building block assembly

**Advanced Science** [[Paper](https://doi.org/10.1002/advs.202206674)] [[arXiv](https://arxiv.org/abs/2111.12907)]

Official github of ***Molecular generative model via retrosynthetically prepared chemical building block assembly*** by Seonghwan Seo\*, Jaechang Lim, Woo Youn Kim. (*Advanced Science*)

This repository is improved version(BBARv2) of [jaechang-hits/BBAR-pytorch](https://github.com/jaechang-hits/BBAR-pytorch) which contains codes and model weights to reproduce the results in paper. You can find the updated architectures at [`architecture`](/architecture).

If you have any problems or need help with the code, please add an issue or contact [shwan0106@kaist.ac.kr](mailto:shwan0106@kaist.ac.kr).

### Citation

```
@article{seo2023bbar,
  title = {Molecular Generative Model via Retrosynthetically Prepared Chemical Building Block Assembly},
  author = {Seo, Seonghwan and Lim, Jaechang and Kim, Woo Youn},
  journal = {Advanced Science},
  volume = {10},
  number = {8},
  pages = {2206674},
  doi = {https://doi.org/10.1002/advs.202206674},
  url = {https://onlinelibrary.wiley.com/doi/abs/10.1002/advs.202206674},
}
```



## Table of Contents

- [Installation](#installation)
- [Data](#data)
  - [Dataset Structure](#dataset-structure)
  - [Prepare Your Own Dataset](#prepare-your-own-dataset)
- [Model Training](#model-training)
  - [Preprocess](#preprocess)
  - [Training](#training)
- [Generation](#generation)

## Installation

The project can be installed by pip with `--find-links` arguments for torch-geometric package.
```bash
pip install -e . --find-links https://data.pyg.org/whl/torch-2.1.2+cu121.html # CUDA
pip install -e . --find-links https://data.pyg.org/whl/torch-2.1.2+cpu.html # CPU-only
```

## Data

### Dataset Structure

#### Data Directory Structure

Move to `data/` directory. Initially, the structure of directory `data/` is as follows.

```bash
├── data/
    ├── ZINC/               (ZINC15 Database)
    │   ├── smiles/
    │   │   ├── train.smi   (https://github.com/wengong-jin/icml18-jtnn/tree/master/data/zinc/train.txt)
    │   │   ├── valid.smi   (https://github.com/wengong-jin/icml18-jtnn/tree/master/data/zinc/valid.txt)
    │   │   └── test.smi    (https://github.com/wengong-jin/icml18-jtnn/tree/master/data/zinc/test.txt)
    │   ├── get_data.py
    │   └── library.csv
    └── 3CL_ZINC/           (Smina calculation result. (ligands: ZINC15, receptor: 7L13))
        ├── data.csv
        ├── split.csv
        └── library.csv     (Same to data/ZINC/library.csv)
```

- `data/ZINC/`, `data/3CL_ZINC/` : Dataset which used in our paper.

#### Prepare ZINC15 Dataset

Move to `data/ZINC` directory, and run `python get_data.py`. And then, `data.csv` and `split.csv` will be created. The dataset for 3CL docking is already prepared.

```bash
├── data/
    ├── ZINC/
        ├── smiles/
        ├── get_data.py
        ├── data.csv      new!
        ├── split.csv     new!
        └── library.csv
```

### Prepare Your Own Dataset

For your own dataset, you need to prepare `data.csv` and `split.csv` as follows.

- `./data/<OWN-DATA>/data.csv`

  ```
  SMILES,Property1,Property2,...
  c1ccccc1,10.25,32.21,...
  C1CCCC1,35.1,251.2,...
  ...
  ```

  - SMILES must be RDKit-readable.
  - If you want to train a single molecule set with different properties, you don't have to configure datasets separately for each property. You need to configure just one dataset file which contains all of property information. For example, `ZINC/data.csv` contains information about `mw`, `logp`, `tpsa`, `qed`, and you can train the model with property or properties, e.g. `mw`, `[mw, logp, tpsa]`.

- `./data/<OWN-DATA>/split.csv`

  ```
  train,0
  train,1
  ...
  val,125
  ...
  test,163
  ...
  ```

  - First column is data type (train, val, test), and second column is index of `data.csv`.

And then, you need to create a ***building block library***. Go to root directory and run `./script/get_library.py`.

```shell
cd <ROOT-DIR>
python ./script/get_library.py \
  --data_dir ./data/<OWN-DATA> \
  --cpus <N-CPUS>
```

After this step, the structure of directory `data/` is as follows.

```bash
├── data/
    ├── <OWN-DATA>/
        ├── data.csv
        ├── split.csv
        └── library.csv     new!
```



## Model Training

The model training requires less than <u>*12 hours*</u> with 1 GPU(RTX2080) and 4 CPUs(Intel Xeon Gold 6234).

### Preprocess (Optional)

You can skip data processing during train by pre-processing data with `./script/preprocess.py`.

```shell
cd <ROOT-DIR>
python ./script/preprocess.py \
  --data_dir ./data/<DATA-DIR> \
  --cpus <N-CPUS>
```

After preprocessing step, the structure of directory `data/` is as follows. `data.csv`, `split.csv` and `library.csv` are required, and `data.pkl` is optional.

```bash
├── data/
    ├── <DATA-DIR>/
        ├── data.csv
        ├── data.pkl      new!
        ├── split.csv
        └── library.csv
```



### Training

```shell
cd <ROOT-DIR>
python ./script/train.py -h
```

Training Script Format Example

Our training script reads model config files `./config/model.yaml`. You can change model size by modifying or creating new config files. You can find another arguments through running with `-h` flag.

```shell
python ./script/train.py \
    --name <exp-name> \
    --exp_dir <exp-dir-name> \          # default: ./result/
    --property <property1> <property2> ... \
    --data_dir <DATA-DIR> \             # default: ./data/ZINC/
    --model_config <model-config-path>  # default: ./config/model.yaml
```

Example running script

```shell
python ./script/train.py \
    --name 'logp-tpsa' \
    --exp_dir ./result/ZINC/ \
    --data_dir ./data/ZINC/ \
    --property logp tpsa

python ./script/train.py \
    --name '3cl_affinity' \
    --exp_dir ./result/3cl_affinity/ \
    --data_dir ./data/3CL_ZINC/ \
    --property affinity
```

## Generation

The model generates 20 to 30 molecules per 1 second with 1 CPU(Intel Xeon E5-2667 v4).

### Download Pretrained Models.

```shell
# Download Weights of pretrained models. (mw, logp, tpsa, qed, 3cl-affinity)
# Path: ./test/pretrained_model/
cd <ROOT-DIR>
sh ./download-weights.sh
```

### Generation

```shell
cd <ROOT-DIR>
python ./script/sample.py -h
```

Example running script.

```shell
# Output directory path
mkdir ./result_sample

# Scaffold-based generation. => use `-s` or `--scaffold`
python ./script/sample.py \
    -g ./test/generation_config/logp.yaml \
    -s "c1ccccc1" \
    --num_samples 100 \
    --logp 6 \
    -o ./result_sample/logp\=6.smi

# Scaffold-based generation. (From File) => use `-S` or `--scaffold_path`
python ./script/sample.py \
    --generator_config ./test/generation_config/mw.yaml \
    --scaffold_path ./test/start_scaffolds.smi \
    --num_samples 100 \
    --mw 300 \
    --o ./result_sample/mw\=300.smi \
    --seed 0 -q
```

Generator config (Yaml)

- generator config format (`./config/generator.yaml`)

  ```yaml
  # If library_builtin_model_path is not null, generator save or load library-builtin model.
  # The library-builtin model contains model parameters and library information.
  # 	(library information: SMILES and latent vector of building block)
  # During configuration process of generator, model vectorizes all building blocks in library.
  # This process requires about 30 seconds. With library-builtin model, this process is skipped.
  # When the file `library_builtin_model_path` exists, upper two parameters (`model_path`, `library_path`) are not needed.
  model_path: <MODEL_PATH>
  library_path: <LIBRARY_PATH>
  library_builtin_model_path: <LIBRARY_BUILTIN_MODEL_PATH>  # optional
  
  # Required
  window_size: 2000
  alpha: 0.75
  max_iteration: 10
  ```

- Example (`./test/generation_config/logp.yaml`)

  ```yaml
  model_path: ./test/pretrained_model/logp.tar
  library_path: ./data/ZINC/library.csv
  library_builtin_model_path: ./test/builtin_model/logp.tar
  
  window_size: 2000
  alpha: 0.75
  max_iteration: 10
  ```
