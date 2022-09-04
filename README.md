# QFM-BERT

codes for our paper [Semantics in Quantum Systems: A Quantum-inspired Fine-tuning Model for Aspect-base Sentiment Analysis]()

## Requirement

- pytorch >= 1.0
- python >= 3.7
- transformers >= 1.2


To install requirements, run `pip install -r requirements.txt`.


## Datasets
SemEval 2014 Task 4 dataset: http://alt.qcri.org/semeval2014/task4/index.php?id=data-and-tools

## Training
Execute the following command to train QFM-BERT and the graphics card used in the experiment is NVIDIA RTX A6000:
```shell script
python train.py --model_name qfm_bert --dataset restaurant --batch_size 32
```

See [train.py](./train.py) for more training arguments.



## Acknowledgement
The baseline of our code is from https://github.com/songyouwei/ABSA-PyTorch.


## Licence

MIT