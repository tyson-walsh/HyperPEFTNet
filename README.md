# HyperPEFTNet
Parameter-Efficient Hypernetworks for Persona Synthesis (AAAI-26)

HyperPEFTNet is the official implementation and reproduction package for  
“HyperPEFTNet: Parameter-Efficient Hypernetworks for Persona Synthesis.”  
The codebase contains:

* **Data pipeline** – scripts to harvest, sanitize, and feature-engineer the 2010-2016 Reddit slice.  
* **Model zoo** – PyTorch implementations of LoRA, Adapter, Bias, and Prefix PEFT back-bones plus the 11-layer hypernetwork generator.  
* **Training & evaluation** – single-GPU and multi-GPU launchers, Optuna sweeps, full metric pipeline (CE/PPL, BLEU, silhouette, ROC-AUC).  
* **Reproducibility** – Dockerfile (CUDA 11.8, PyTorch 2.3), YAML config snapshots, SHA-256 manifest, and an AAAI-26 reproducibility checklist.

> **Status:** code accompanying the AAAI-26 submission.  
> **Paper:** arXiv link | AAAI-26 camera-ready (pending).  
> **Citation:** see `CITATION.cff`.