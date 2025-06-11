# KomplexNet

**KomplexNet** is a PyTorch Lightning-based framework designed to explore complex-valued neural networks with phase-based synchronization mechanisms, particularly Kuramoto-style dynamics. The model is tested on datasets like MultiMNIST and integrates several backbone variants such as small real-valued, complex-valued, and ViT-inspired networks.

## 🧠 Project Highlights

- **Phase synchrony**: Incorporates Kuramoto oscillator dynamics for phase-based grouping of neurons.
- **Modular architecture**: Supports multiple model variants (real, complex, ViT).
- **Dataset ready**: Pre-configured for MultiMNIST and compatible with MultiMNIST with Cifar in the Backgrounf.
- **Logging and checkpoints**: Integrated with Weights & Biases (wandb) and PyTorch Lightning callbacks.

## 📁 Project Structure

```
KomplexNet/
├── main.py                     # Main training and evaluation script
├── model.py                    # Core network modules
├── smaller_complex_model.py    # Complex-valued model with Kuramoto coupling
├── smaller_model.py            # Real-valued baseline
├── smaller_vit.py              # Lightweight ViT-like model
├── dataloaders.py              # Dataloaders for supported datasets
├── make_multi_mnist.py         # Script to generate MultiMNIST dataset
├── opts.py                     # Argument parser
├── complex_functions.py        # Custom complex math ops (e.g., Kuramoto update)
└── pt_utils/                   # Pretrained kernels and weights
```

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/S4b1n3/KomplexNet.git
cd KomplexNet
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare the dataset

First, download MNIST and CIFAR dataset and save them in the `./data/` repository.

Then, to generate MultiMNIST:

```bash
python make_multi_mnist.py
```

Place the dataset in the format: `./data/<dataset>/<repo_name>/`.

## 🏃‍♀️ Running Training

```bash
python main.py --model small_complex --dataset MultiMNIST --in_repo default --out_repo output_name --filename test_run
```

### Key Arguments (via `opts.py`):
- `--model`: Choose from `small_real`, `small_complex`, `small_vit`
- `--dataset`: Dataset name (e.g., `MultiMNIST`)
- `--in_repo`: Folder name where data is stored
- `--out_repo`: Output folder name for logs/checkpoints
- `--filename`: Experiment name
- `--resume`: Resume from checkpoint
- `--epochs`: Number of training epochs

The `opts.py` file sets the hyper-parameters to the values presented in the paper.

## 📊 Logging

Experiments are logged using **Weights & Biases**. Make sure to log in:

```python
wandb.login()
```

## 📈 Evaluation

After training, testing is performed using `test.py` by loading the best checkpoint.

## 📚 Citation

If you use KomplexNet in your research, please cite the following paper:

```
@article{muzellec2025enhancing,
  title={Enhancing deep neural networks through complex-valued representations and Kuramoto synchronization dynamics},
  author={Muzellec, Sabine and Alamia, Andrea and Serre, Thomas and VanRullen, Rufin},
  journal={arXiv preprint arXiv:2502.21077},
  year={2025}
}
```



