# From Self-Attention Mechanism to Transformers

Interactive notebook exploring self-attention from first principles to a mini Transformer encoder–decoder with multi-head attention, plus hands-on exercises and applications.

- Notebook: [Attention_Transformer.ipynb](Attention_Transformer.ipynb)
- Requirements: [requirements.txt](requirements.txt)

## Project structure
- Attention_Transformer.ipynb — the full tutorial notebook (with visuals and exercises)
- requirements.txt — minimal dependencies (numpy, matplotlib, seaborn, tensorflow, jupyter)
- .gitignore — standard ignores for Python/Jupyter

## Quick start (recommended: Python 3.11)
The notebook includes a setup cell that installs packages into the active kernel and verifies TensorFlow/GPU. For the smoothest experience, create a fresh Python 3.11 environment.

### Option A: venv + pip
- Windows (PowerShell):
  1) python -m venv .venv
  2) .\.venv\Scripts\Activate.ps1
  3) python -m pip install -U pip
  4) pip install -r requirements.txt

- macOS/Linux (bash):
  1) python3 -m venv .venv
  2) source .venv/bin/activate
  3) python -m pip install -U pip
  4) pip install -r requirements.txt

### Option B: Conda (Windows/macOS/Linux)
1) conda create -n attn python=3.11 -y
2) conda activate attn
3) pip install -r requirements.txt

Register the environment as a Jupyter kernel (optional but useful):
- pip install ipykernel
- python -m ipykernel install --user --name attn --display-name "Python 3.11 (attn)"

## Run the notebook
- VS Code:
  - Open the folder, open [Attention_Transformer.ipynb](Attention_Transformer.ipynb)
  - Select the "Python 3.11 (attn)" kernel
  - Run “Run All” (the first cell installs/validates deps automatically)
- Terminal:
  - jupyter lab
  - Open [Attention_Transformer.ipynb](Attention_Transformer.ipynb), run all cells

Note: An active internet connection is required on first run to install packages.

## Optional GPU notes (Windows)
The setup cell tries tensorflow-directml when on Windows + Python 3.11 and falls back to CPU TensorFlow if unavailable. To check GPU:
- python -c "import tensorflow as tf; print(tf.__version__); print('built_with_cuda:', tf.test.is_built_with_cuda()); print('GPUs:', tf.config.list_physical_devices('GPU'))"

GPU is not required for this educational notebook (arrays + heatmaps). CPU runs are fine.

## What you’ll learn inside the notebook
- Self-Attention basics (NumPy/TensorFlow) with detailed visualizations
- Sentence-level attention and simple embedding strategies
- Encoder pipeline: self-attention, residuals, layer norm, feed-forward
- Encoder–decoder with cross-attention
- Multi-head attention (mini Transformer)
- Practical exercises:
  - Input shapes, scaling, masking, multi-head
  - Positional encoding, Xavier init
  - Varying dimensions/heads, dropout
  - Pre-/Post-LayerNorm, alternative activations
  - Realistic applications: sentiment heatmaps, extractive summarization via sentence-level attention, and QA cross-attention

Core self-attention equation used throughout:
- $\\text{Attention}(Q,K,V) = \\text{softmax}\\left(\\dfrac{QK^\\top}{\\sqrt{d_k}}\\right) V$

## Troubleshooting
- Packages fail to install
  - Upgrade tools: python -m pip install -U pip setuptools wheel
  - Then: pip install -r requirements.txt
- Wrong kernel in Jupyter/VS Code
  - Switch to the environment you created (e.g., “Python 3.11 (attn)”)
- TensorFlow GPU not used
  - This tutorial does not require GPU; see the “Why TensorFlow is not using the GPU” section in the notebook for options.

## Credits
By Abdelaziz LOUNES, Univ. Paris 8