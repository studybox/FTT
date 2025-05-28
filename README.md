# FrenetTrajectoryTranslation

This is the official code repository for the paper:

**Multi-modal Trajectory Prediction With Frenet Trajectory Translation**  
*Boqi Li, Shaobing Xu, Huei Peng*

This work introduces a novel approach to multi-modal vehicle trajectory prediction by translating predicted trajectories into the Frenet coordinate space. It leverages both learned routing intent and map structure to improve prediction accuracy.

---

## 📦 Installation

We recommend using a conda environment:

```bash
conda create -n ftt python=3.9.19 numpy=1.23.0
conda activate ftt
# Install PyTorch and CUDA:
conda install pytorch=1.12.1 torchvision=0.13.1 cudatoolkit=11.3 -c pytorch
```

Install PyTorch Geometric dependencies:

```bash
mkdir pyg_depend && cd pyg_depend
wget https://data.pyg.org/whl/torch-1.12.0%2Bcu113/torch_cluster-1.6.0%2Bpt112cu113-cp39-cp39-linux_x86_64.whl
wget https://data.pyg.org/whl/torch-1.12.0%2Bcu113/torch_scatter-2.1.0%2Bpt112cu113-cp39-cp39-linux_x86_64.whl
wget https://data.pyg.org/whl/torch-1.12.0%2Bcu113/torch_sparse-0.6.16%2Bpt112cu113-cp39-cp39-linux_x86_64.whl
wget https://data.pyg.org/whl/torch-1.12.0%2Bcu113/torch_spline_conv-1.2.1%2Bpt112cu113-cp39-cp39-linux_x86_64.whl
python3 -m pip install torch_cluster-1.6.0+pt112cu113-cp39-cp39-linux_x86_64.whl
python3 -m pip install torch_scatter-2.1.0+pt112cu113-cp39-cp39-linux_x86_64.whl
python3 -m pip install torch_sparse-0.6.16+pt112cu113-cp39-cp39-linux_x86_64.whl
python3 -m pip install torch_spline_conv-1.2.1+pt112cu113-cp39-cp39-linux_x86_64.whl
python3 -m pip install torch_geometric
```

Install other dependencies:
```bash
pip install scikit-learn networkx 
pip install iso3166
pip install shapely==1.7.1
pip install tensorboard
pip install pytorch-lightning
pip install wandb
pip install dgl-cu113 -f https://data.dgl.ai/wheels/repo.html
```

---

## Directory structure:
```
FTT/
├── data/
│   ├── rounD/
│   │   ├── processed/
│   │   └── raw/
│   ├── interaction-CHN/
│   └── ...
├── datasets/
├── pretrained/
├── models/
│   └── ftt/
├── config.py
├── train.py
└── val.py
```
---

## 📂 Dataset and Pretrained Models

- Place **processed datasets** in the `data/` directory.
- Place **pretrained models** in the `pretrained/` directory.

👉 You can download the datasets and pretrained models from the following links:

- [📥 Dataset Download Link](#) *(insert actual link)*
- [📥 Pretrained Models](#) *(insert actual link)*

---

## 🚀 Training

To train a model using the **FTT** architecture:

```bash
python train.py --model ftt
```

To train other models for comparison:

```bash
python train.py --model smart
```

To evaluate a model:

```bash
python val.py --model ftt
```

---

## 📫 Contact

For questions, issues, or collaborations, feel free to reach out:

**Boqi Li**  
📧 boqili@umich.edu