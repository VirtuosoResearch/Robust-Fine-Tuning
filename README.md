### Overview

In this repository, we provide the implementation to evaluate Hessian-based measurements of fine-tuned models, including traces and Hessian vector products, based on results presented at ICML'22, DC. Our observation is that by incorporating Hessians with distance from initialization, the Hessian distance measure provides a nonvacuous measure of the generalization gaps of fine-tuned models in the sense that the measures have the same scale as the empirically observed generalization gaps.

### Requirements

To install requirements:

```bash
pip install -r requirements.txt
```

### Data Preparation

We use six domains of object classification tasks from the DomainNet (Peng et al., 2019) dataset and create labels using weak supervision approaches following the work of Mazzetto et al. (2021). For completeness, we provide a copy of the DomainNet dataset under the directory `./exps_on_image_datasets/data/`. We refer the reader to the [implementation](https://github.com/BatsResearch/amcl) of Mazzetto et al. (2021) for the process of generating the labels. 

Besides the DomainNet dataset, we also use several other datasets in our experiments. We list the link for downloading the datasets below:

- [Caltech-256](http://www.vision.caltech.edu/Image_Datasets/Caltech256/): download and extract into `./exps_on_image_datasets/data/caltech256/`

- [MIT-Indoor](http://web.mit.edu/torralba/www/indoor.html): download and extract into `./exps_on_image_datasets/data/Indoor/`
- The data of CIFAR-10, CIFAR-100, MRPC, and SST2 datasets will be automatically downloaded. 

Our code automatically handles the processing of the datasets. 

### Usage

We provide the implementations of fine-tuning on both image and text classification tasks. Run the corresponding experiments under the directory of `./exps_on_image_datasets/` and `./exps_on_text_datasets/`. 

#### **Fine-tuning on Image Classification Tasks**

**Fine-tuning ResNet models under label noise.**

Use `train_label_noise.py` to run experiments of fine-tuning on noisy labels. Follow the bash script example to run the commands. Choose the domain from clipart, infograph, painting, quickdraw, real, and sketch.  Before running the command, please create a directory `./saved_label_noise` for saving the checkpoints and logs. 

```bash
cd exps_on_image_datasets
mkdir saved_label_noise

python train_label_noise.py --config configs/config_constraint_domain_net.json \
    --model ResNet18 \
    --lr 0.0001 --batch_size 8 --runs 10 --device 1 \
    --domain $domain_name --sample 1 \
    --constraint_reweight --reweight_noise_rate $rate\
    --reg_method constraint --reg_norm frob --reg_extractor $constraint_for_extractor --reg_predictor $constraint_for_predictor
# Use --load_matrix to apply estimated label confusion matrix
```

**Fine-tuning Vision Transformer on noisy labels.**

```bash
python train_label_noise_ws.py --config configs/config_constraint_domain_net.json \
    --model VisionTransformer --is_vit --img_size 224 --vit_type ViT-B_16 --vit_pretrained_dir pretrained/imagenet21k_ViT-B_16.npz \
    --lr 0.0001 --batch_size 8 --runs 5 --device 0 \
    --domain $domain_name --sample 1 \
    --constraint_reweight --reweight_noise_rate $rate\
    --reg_method constraint --reg_norm frob --reg_extractor $constraint_for_extractor --reg_predictor $constraint_for_predictor
```

Please follow the instructions in [ViT-pytorch](https://github.com/jeonsworld/ViT-pytorch) to download the pre-trained models.

**Evaluating the Hessian-based measures.** 

Use the following scripts to compute noise stability and Hessian-based measures. We use Hessian vector multiplication tools from PyHessian (Yao et al., 2020).

- `compute_noise_stability.py` computes the averaged noise stability with a given perturbation scale $\sigma$
- `compute_hessian_traces.py` computes the trace of loss's Hessian of each layer in a neural network. 

- `compute_hessian_norms.py` computes the Hessian-based measure developed in this paper. 

Follow the bash script examples to run the commands. Specify the config file under the directory of `./configs`. Create a directory named `./traces` for storing the quantities.

```bash
mkdir traces

python compute_noise_stability.py --config $config_file_name \
    --model $model_name --batch_size 4 \
    --checkpoint_dir $checkpoint_dir_for_the_model \
    --checkpoint_name $checkpoint_name_for_the_model \
    --sample_size $sample_size --eps $perturbation --device 0
    
python compute_hessian_traces.py --config $config_file_name \
    --model $model_name --batch_size 4 \
    --checkpoint_dir $checkpoint_dir_for_the_model \
    --checkpoint_name $checkpoint_name_for_the_model \
    --save_name $name_for_saved_file --sample_size $sample_size --num_layers $number_of_layers_in_model --device 0
    
python compute_hessian_measures.py --config $config_file_name \
   --model $model_name --batch_size 4 \
    --checkpoint_dir $checkpoint_dir_for_the_model \
    --checkpoint_name $checkpoint_name_for_the_model \
    --save_name $name_for_saved_file --sample_size $sample_size --num_layers $number_of_layers_in_model --device 0
```

#### **Fine-tuning on Text Classification Tasks**

**Fine-tuning RoBERTa/BERT model on noisy labels**

Use `train_glue_label_noise.py` to run the experiments of fine-tuning Roberta-Base. Follow the bash script example to run the command. 

```bash
python train_glue_label_noise.py --config configs/config_glue.json --task_name mrpc \
    --epochs 10 --runs 3 --device 0 --noise_rate $noise_rate --model_name_or_path roberta-base \
    --constraint_reweight --reweight_noise_rate $reweight_rate \
    --reg_method constraint --reg_norm frob \
    --reg_attention $constraint_for_attention_layer \
    --reg_linear $constraint_for_linear_layer\
    --reg_predictor $constraint_for_task_predictor
```

**Evaluating the Hessian-based measures on BERT/RoBERTa**

Use the following scripts to compute noise stability and Hessian-based measures. We use Hessian vector multiplication tools from PyHessian (Yao et al., 2020).

- `compute_noise_stability.py` computes the averaged noise stability with a given perturbation scale $\sigma$
- `compute_hessian_traces.py` computes the trace of loss's Hessian of each layer in a neural network. 

- `compute_hessian_norms.py` computes the Hessian-based measure developed in this paper. 

Follow the bash script examples to run the commands. Create a directory named `./traces` for storing the quantities. 

```bash
mkdir traces

python compute_noise_stability.py --config configs/config_glue.json --task_name mrpc --device 0 \
--checkpoint_dir $specify_a_checkpoint_dir --checkpoint_name $specify_a_checkpoint_name --sample_size $sample_size --eps $eps

python compute_hessian_traces.py --config configs/config_glue.json --task_name mrpc --device 0 \
    --checkpoint_dir $specify_a_checkpoint_dir --checkpoint_name $specify_a_checkpoint_name --save_name $specify_a_save_filename --sample_size $sample_size

python compute_hessian_measures.py --config configs/config_glue.json --task_name mrpc --device 0 \
    --checkpoint_dir $specify_a_checkpoint_dir --checkpoint_name $specify_a_checkpoint_name --save_name $specify_a_save_filename --sample_size $sample_size
```

### Citation

If you find this repository useful or happen to use it in a research paper, please cite our work with the following bib information.

```latex
@article{ju2022robust,
  title={Robust Fine-Tuning of Deep Neural Networks with Hessian-based Generalization Guarantees},
  author={Ju, Haotian and Li, Dongyue and Zhang, Hongyang R},
  journal={International Conference on Machine Learning},
  year={2022}
}
```

### Acknowledgment

Thanks to the authors of the following repositories for providing their implementation publicly available.

- **[PyHessian](https://github.com/amirgholami/PyHessian)**
- **[Adversarial-Multi-Class-Labeling](https://github.com/BatsResearch/amcl)**
- **[ViT-pytorch](https://github.com/jeonsworld/ViT-pytorch)**
