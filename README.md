# Effective Approaches to Attention-based Neural Machine Translation
Encoder-Decoder model with global attention mechanismのpytorch実装。

## Model Details
- LSTM-based encoder-decoder model
- global attention (see Figure 2 in original paper)
- scheduled sampling


## Usages
学習
```sh
python train.py \
    --gpu
    --train ./sample_data/sample_train.py
    --valid ./sample_data/sample_valid.py
    --tf-ratio 0.5
    --savedir ./checkpoints
```

翻訳
```sh
python translate.py \
    --gpu
    --model ./checkpoints/checkpoint_best.pt
    --input ./sample_data/sample_test.txt
```

## References
- [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/pdf/1508.04025.pdf)
- [Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks](https://arxiv.org/pdf/1506.03099.pdf)
