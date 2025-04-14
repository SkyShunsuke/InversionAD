# Reconstruction-Free Anomaly Detection with Diffusion Models via Direct Latent Likelihood Evaluation <br><sub>Official PyTorch Implementation</sub>

[![arXiv](https://img.shields.io/badge/arXiv%20paper-2406.11838-b31b1b.svg)](https://arxiv.org/abs/2504.05662)&nbsp;
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/reconstruction-free-anomaly-detection-with/anomaly-detection-on-mvtec-ad)](https://paperswithcode.com/sota/anomaly-detection-on-mvtec-ad?p=reconstruction-free-anomaly-detection-with)

<p align="center">
  <img src="demo/method.png" width="720">
</p>

This is a PyTorch/GPU implementation of the paper [Reconstruction-Free Anomaly Detection with Diffusion Models via Direct Latent Likelihood Evaluation](https://arxiv.org/abs/2504.05662):

```
@article{sakai2025inversionad,
      title={Reconstruction-Free Anomaly Detection with Diffusion Models via Direct Latent Likelihood Evaluation}, 
      author={Shunsuke Sakai and Tatsuhito Hasegawa},
      year={2025},
      url={https://arxiv.org/abs/2504.05662}, 
}
```

## Preparation

### Installation

Download the code:
```
git clone https://github.com/SkyShunsuke/InversionAD
cd InversionAD
```

A suitable [conda](https://conda.io/) environment named `invad` can be created and activated with:

```
conda env create -f environment.yaml
conda activate invad
```

### Dataset
Download [MVTecAD](https://www.mvtec.com/company/research/datasets/mvtec-ad) dataset with the following command:
```bash
bash scripts/download_datasets.sh
```
and place it in the `data` directory. 

### Wandb Logging [Optional]
If you want to use Weights & Biases for logging, please make .env file in the root directory and set your wandb API key as follows:
```
WANDB_API_KEY=your_api_key
```
You can get your API key from [here](https://wandb.ai/authorize).
You can also set the project name and entity from the .env file:
```
WANDB_PROJECT=your_project_name
WANDB_ENTITY=your_entity_name
```

## Usage

### Training
To train the model, run the following command:

```
bash scripts/train.sh
```

This will train the model on the MVTecAD dataset. The training process will save the model checkpoints in the `results` directory.

And you can change training parameters in `configs/xxx.yaml` file.

### Evaluation
To evaluate the model, run the following command:
```
bash scripts/eval.sh
```
This will evaluate the model on the MVTecAD dataset.

## Acknowledgements
We thank for very nice diffusion implementation from [OpenAI](https://github.com/openai/guided-diffusion). 

## Contact

If you have any questions, feel free to contact me through email (mf240599@g.u-fukui.ac.jp).




