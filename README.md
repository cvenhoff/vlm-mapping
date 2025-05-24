# Understanding Feature Mappings in VLMs

This repository implements our research on understanding how vision-language models achieve cross-modal representation alignment through a controlled setup with frozen LLM and ViT components. By analyzing the mapping between visual and linguistic representations using sparse autoencoders, we investigate how visual features gradually align with language features across different layers.

## Requirements

All experiments and training were performed on a single NVIDIA A100 GPU. To set up the environment:

```bash
conda env create --file=environment.yaml
```

## 1. Training Frozen CLIP-Gemma

To train the model:

1. Install the training data:
```bash
./install_data.sh
```

2. Navigate to the training directory:
```bash
cd train
```

3. Set up the accelerate configuration according to your specifications

4. Start the training:
```bash
./train.sh
```

For convenience, we provide pre-trained projector weights in `train/weights` to facilitate reproduction of our results.

## 2. Feature Mapping Investigation

To run the feature mapping experiments:

1. Create a `.env` file with the following environment variables:
   - `HF_TOKEN`: Your Hugging Face token for accessing Gemma 2
   - `OPENAI_API_KEY`: Your OpenAI API key for running semantic alignment experiments

2. Run the investigation script:
```bash
./run.sh
```

This script performs the following steps:
- Caches 5000 activations for analysis
- Computes sparsity and reconstruction error across all activations using `sae_reconstruction.py`
- Evaluates semantic alignment using `sae_description_eval.py`

All results are saved in the `results` directory, providing insights into layer-wise progression of visual-language alignment, reconstruction error patterns, sparsity characteristics, and semantic alignment measurements.

## Citation

```bibtex
@article{venhoff2025how,
  title     = {How Visual Representations Map to Language Feature Space in Multimodal LLMs},
  author    = {Constantin Venhoff and Ashkan Khakzar and Sonia Joseph and Philip Torr and Neel Nanda},
  booktitle = {The 4th Explainable AI for Computer Vision (XAI4CV) Workshop at CVPR 2025},
  year      = {2025},
}
```