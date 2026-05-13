# Recursive Flux Engine: Colab T4 PoC Guide

This guide explains how to run a 500M parameter **Recursive Flux Engine** PoC on a free Google Colab T4 instance.

## 1. Setup
Open a new notebook in Colab and set the runtime to **T4 GPU**.

## 2. Installation
Run the following in a cell to ensure the environment is ready:
```python
!pip install torch torchvision torchaudio
```

## 3. Upload Code
Upload `recursive_flux_engine.py` and `flux_poc_colab.py` to your Colab workspace.

## 4. Run the PoC
Run the following command in a cell:
```python
!python flux_poc_colab.py
```

## 5. Performance Expectations on T4 (16GB VRAM)
- **Model Size**: ~500M parameters.
- **Memory Usage**: ~8-10GB during training with `autocast` and `batch_size=4`.
- **Latency**: ~0.5s - 1s per training step (including the recursive passes).

## 6. Tuning for Stability
- If you encounter **Out of Memory (OOM)**:
  - Reduce `batch_size` to 1 or 2 in `flux_poc_colab.py`.
  - Reduce `d_model` to 768 in `RecursiveFluxEngine`.
- If you encounter **Gradient Explosion**:
  - The model already includes `clip_grad_norm_`, but you can further reduce the learning rate to `5e-5`.

## 7. What this PoC demonstrates
1. **SSM Memory**: The Mamba-3 blocks ingest and compress long sequences.
2. **Phase Detection**: The LNN layer identifies the logical "trigger" token.
3. **Recursive Recall**: The RWKV-7 HeadQK core recursively "searches" its state to recall the target token after a long sequence of zeros (the gap).
