# CAD

paper: Comparative Learning Based Stance Agreement Detection Framework for Multi-Target Stance Detection.

## Abstract

Multi-target stance detection is the detection of the stance of multiple targets in text. Currently, most multi-target stance detection methods only detect the stance of two targets individually and do not make the two targets complement each other to take full advantage of the relevant semantic information between the two targets. In this paper, we propose a comparative learning based stance agreement detection framework. We applied contrastive learning to stance agreement detection, it enabled the model to learn more information about the features of the target and to strengthen the links between the semantic information of the targets so that they assist each other in stance detection. In addition, we fine-tuned a new model as our encoder to more fully exploit the semantic information between hidden contexts. We also apply joint training as a multi-task learning approach, allowing models to share domain-specific information based on the dataset. By comparing different methods, experimental results show that our method achieves state-of-the-art results on multi-target benchmark datasets. In the concluding sections of our paper, we conducted error analysis experiments on the proposed methodology, elucidating its inherent limitations and furnishing invaluable insights conducive to future enhancements.

## Run

First, configure the environment:
```
$ pip install -r requirements.txt
```
For CAD in `merged` training setting, run
```
cd src/
python train_model.py \
    --input_target all \
    --model_select Bertweet \
    --train_mode unified \
    --col Stance1 \
    --lr 2e-5 \
    --batch_size 32 \
    --epochs 20 \
    --dropout 0. \
    --alpha 0.5
```
`input_target` can take one of the following target-pairs [`trump_hillary`, `trump_ted`, `hillary_bernie`] in adhoc setting and take [`all`] in merged setting.

`col` indicates the target in each target-pair. For example, for the target-pair `Trump-Clinton`, we have `Stance1` for Trump and `Stance2` for Clinton.

## Contact Info

Please contact Guantong Liu at lgt@dlmu.edu.cn with any questions.
