# POT

This repository is the official implementation of the SIGIR 2022 Paper [Personalized Abstractive Opinion Tagging](https://staff.fnwi.uva.nl/m.derijke/wp-content/papercite-data/pdf/zhao-2022-personalized.pdf).

If you have any question, please open an issue or contact <keninazhao@163.com>.

## PATag Dataset 
We provide a runnable version of the PATag data. You can run this program directly using `data.pkl` and `matrix.pkl` in [Google Drive](https://drive.google.com/drive/folders/1ST6maKXhkab6bEuPdJtgRbPg2IjXaiDz?usp=sharing). You should download and move it under`./DataSet/`. 


You can view the details of the relevant data through `./DataSet/data_info.txt`, and `./DataSet/sample_for_data_pair.txt` is an instance of data pair.

If you want to get raw data and more processing details, please use this [Google form](https://docs.google.com/forms/d/e/1FAIpQLSc-SkZnd2rJqjSkPYOvi5ShvCHlbnYA8viS6459yEy27dPdYQ/viewform?usp=sf_link) to submit your information and request access to PATag.

## Prerequisites
```console
- CUDA >= 10.0
- Python >= 3.6
- PyTorch >= 1.7
```

## Setup
Check the packages needed or simply run the command:
```console
pip install -r requirements.txt
```

## Experiments
You can run the program with the following command: 
```bash
bash script/run.sh model_name gpus is_train
```

'model_name' can be 'POT', 'POT_woBehavior' or 'POT_woHGAT', to correspond to our proposed model, as well as two variants.

We support and recommend using multiple GPUs for training and a single GPU for testing.

### Train
```bash
bash script/run.sh POT 0,1,2,3 1
```

### Test
For reproducibility purposes, we place the model checkpoints at [Google Drive](https://drive.google.com/drive/folders/1ggxkJCFDW30gyZG4tXv5ecU0CNPr-0ko?usp=sharing). You should download and move it under `./Output/model_name/`, then you can run the trained models to test by using `best.pkl` and `best_memory.p`.

```bash
bash script/run.sh POT 0 0
```

## Reference
If you find our code useful, please cite our paper as follows:
```bibtex
@article{zhao2022personalized,
  title={Personalized Abstractive Opinion Tagging},
  author={Zhao, Mengxue and Yang, Yang and Li, Miao and Wang, Jingang and Wu, Wei and Ren, Pengjie and de Rijke, Maarten and Ren, Zhaochun},
  year={2022}
}
```
