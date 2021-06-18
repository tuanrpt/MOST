# MOST: Multi-Source Domain Adaptation via Optimal Transport for Student-Teacher Learning

This is the implementation of **Multi-Source Domain Adaptation via Optimal Transport for Student-Teacher Learning** (MOST) model which was accepted at UAI 2021.

## Setup

### Install Package Dependencies

**Install manually**

```
Python Environment: >= 3.5
Tensorflow: >= 1.9
```

**Install automatically from YAML file**

```
pip install --upgrade pip
conda env create --file tf1.9py3.5.yml
```

**[UPDATE] Install tensorbayes**

Please note that tensorbayes 0.4.0 is out of date. Please copy a newer version to the *env* folder (tf1.9py3.5) using **tensorbayes.tar**

```
pip install tensorbayes
tar -xvf tensorbayes.tar
cp -rf /tensorbayes/* /opt/conda/envs/tf1.9py3.5/lib/python3.5/site-packages/tensorbayes/
```

### Install Datasets

To run on Digits-five dataset, in the root folder, please create a new folder named *datasets*.  

At the next step, user downloads Digits-five dataset [here](https://drive.google.com/file/d/12dUT_xBfikgsjYI6w9FyAKvx1UIz-Ccg/view?usp=sharing) and save it into *datasets* folder.

## Training

We first navigate to *model* folder, and then run *run_most.py* file as bellow:

```python
cd model
```

1. "*--> **mm***'' task

```python
python run_most.py 1 "mnist32_60_10,usps32,svhn,syn32" mnistm32_60_10 format mat num_iters 80000 phase1_iters 0 summary_freq 800 learning_rate 0.0002 batch_size 200 src_class_trade_off 1.0 src_domain_trade_off "1.0,1.0,1.0,1.0" ot_trade_off 0.1 domain_trade_off 1.0 trg_vat_troff 0.1 trg_ent_troff 0.1 data_shift_troff 10.0 mimic_trade_off 0.1 cast_data True cnn_size small theta 0.1 sample_size 5
```

2. ''*--> **mt***'' task
```python
python run_most.py 1 "mnistm32_60_10,usps32,svhn,syn32" mnist32_60_10 format mat num_iters 80000 phase1_iters 0 summary_freq 800 learning_rate 0.0002 batch_size 200 src_class_trade_off 1.0 src_domain_trade_off "1.0,1.0,1.0,1.0" ot_trade_off 0.1 domain_trade_off 1.0 trg_vat_troff 0.1 trg_ent_troff 0.1 data_shift_troff 10.0 mimic_trade_off 1.0 cast_data True cnn_size small theta 0.1 sample_size 5
```

3. ''*--> **up***'' task
```python
python run_most.py 1 "mnistm32_60_10,mnist32_60_10,svhn,syn32" usps32 format mat num_iters 80000 phase1_iters 0 summary_freq 800 learning_rate 0.0002 batch_size 200 src_class_trade_off 1.0 src_domain_trade_off "1.0,1.0,1.0,1.0" ot_trade_off 0.1 domain_trade_off 1.0 trg_vat_troff 0.1 trg_ent_troff 0.1 data_shift_troff 10.0 mimic_trade_off 1.0 cast_data True cnn_size small theta 0.1 sample_size 5
```

4. ''*--> **sv***'' task
```python
python run_most.py 1 "mnistm32_60_10,mnist32_60_10,usps32,syn32" svhn format mat num_iters 80000 phase1_iters 0 summary_freq 800 learning_rate 0.0002 batch_size 200 src_class_trade_off 1.0 src_domain_trade_off "1.0,1.0,1.0,1.0" ot_trade_off 0.1 domain_trade_off 1.0 trg_vat_troff 0.1 trg_ent_troff 0.0 data_shift_troff 10.0 mimic_trade_off 1.0 cast_data True cnn_size small theta 0.1 sample_size 5
```

5. ''*--> **sy***'' task
```python
python run_most.py 1 "mnistm32_60_10,mnist32_60_10,usps32,svhn" syn32 format mat num_iters 80000 phase1_iters 0 summary_freq 800 learning_rate 0.0002 batch_size 200 src_class_trade_off 1.0 src_domain_trade_off "1.0,1.0,1.0,1.0" ot_trade_off 0.1 domain_trade_off 1.0 trg_vat_troff 0.1 trg_ent_troff 0.0 data_shift_troff 10.0 mimic_trade_off 1.0 cast_data True cnn_size small theta 0.1 sample_size 5
```

## Results

|     Methods     |  --> mm  |  --> mt  |  --> us  |  --> sv  |  --> sy  |   Avg    |
| :-------------: | :------: | :------: | :------: | :------: | :------: | :------: |
|    MDAN [1]     |   69.5   |   98.0   |   92.4   |   69.2   |   87.4   |   83.3   |
|    DCTN [2]     |   70.5   |   96.2   |   92.8   |   77.6   |   86.8   |   84.8   |
|    M3SDA [3]    |   72.8   |   98.4   |   96.1   |   81.3   |   89.6   |   87.7   |
|    MDDA [4]     |   78.6   |   98.8   |   93.9   |   79.3   |   89.7   |   88.1   |
|  LtC-MSDA [5]   |   85.6   |   99.0   |   98.3   |   83.2   |   93.0   |   91.8   |
| **MOST** (ours) | **91.5** | **99.6** | **98.4** | **90.9** | **96.4** | **95.4** |

## References

[1] H. Zhao, S. Zhang, G. Wu, J. M. F. Moura, J. P. Costeira, and G. J Gordon. Adversarial multiple source domain adaptation. In S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. CesaBianchi, and R. Garnett, editors, Advances in Neural Information Processing Systems 31, pages 8559-8570. Curran Associates, Inc., 2018 .

[2] R. Xu, Z. Chen, W. Zuo, J. Yan, and L. Lin. Deep cocktail network: Multi-source unsupervised domain adaptation with category shift. In 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3964-3973, 2018.  

[3] X. Peng, Q. Bai, X. Xia, Z. Huang, K. Saenko, and B. Wang. Moment matching for multi-source domain adaptation. In Proceedings of the IEEE International Conference on Computer Vision, pages 1406-1415, 2019.  

[4] S. Zhao, G. Wang, S. Zhang, Y. Gu, Y. Li,
Z. Song, P. Xu, R. Hu, H. Chai, and K. Keutzer. Multi-source distilling domain adaptation. In The Thirty-Fourth AAAI Conference on Artificial Intelligence, AAAI 2020, The Thirty-Second Innovative Applications of Artificial Intelligence Conference, IAAI 2020, The Tenth AAAI Symposium on Educational Advances in Artificial Intelligence, EAAI 2020, New York, NY, USA, February 7-12, 2020, pages 12975-12983. AAAI Press, 2020.

[5] H. Wang, M. Xu, B. Ni, and W. Zhang. Learning to combine: Knowledge aggregation for multisource domain adaptation. In Computer Vision - ECCV, 2020. 

