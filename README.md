# FNC-1 DistilBERT
Playing with DistilBERT and FNC-1 task.

## Task
[Fake News Chellange](http://www.fakenewschallenge.org/) consists in detecting _fake news_. 
More specifically, the goal is to construct a classifier that can classify pair of headline
and article body into one of 4 classes: `agree`, `disagree`, `discuss` and `unrelated`.

## Model
To solve the task, I've chosen [DistilBERT](https://arxiv.org/abs/1910.01108) architecture. Using
a smaller and distilled version of [BERT](https://arxiv.org/pdf/1810.04805.pdf) allowed me to
train the model on my own laptop.

I've used pre-trained model from [transformers](https://github.com/huggingface/transformers)
package. For the training transformer's weights were _frozen_. This means that only the final dense
layers were updated and _DistilBert_ served as a feature extractor.

BERT architecture expects 2 texts as the input. Therefore, headline and article body were
used. The final input vector was truncated to the length of 512.


## Training
Due to the lack of time and computational resources I've trained only 3 models on provided by
the organizers train dataset (with 0.2 - 0.8 val - train split) for 10 epochs, each time using
different set of hyperparameters. The initial parameters were inspired by parameters commonly used for
training of this kind of architectures. Then I've chosen the best performing set of
the hyperparameters to train the model on the whole train set and evaluated it on the official test set.

## How to run?
Python3.8 is required. 

### Installation
1. Create or activate your [virtual environment](https://docs.python.org/3/tutorial/venv.html).
2. Install dependencies:
```shell
pip install -r requirements.txt
```
3. Follow [the instructions](https://www.tensorflow.org/install/gpu)
   from TensorFlow docs to enable CUDA support (if it isn't already enabled on your machine).

### Run
The main script is `run.py`. It supports running in two modes: `train-val` and `train-test`.
The first one trains the model on the 80% of the train set and evaluates on the rest of it.
The second mode performs training on the whole train set and evaluates the model on the official
test set. Results and models are saved in `results` folder.

To run the script execute:
```shell
python run.py <mode> <args>
```
For example:
```shell
python run.py train-val --epochs 10
```
To see the rest of available arguments run:
```shell
python run.py --help
```

Final model was trained using the following command:
```shell
python run.py train-test --epochs 20 --seed 123 --weight-decay 0.01 --lr 1e-4 --name lr1e-4_scheduler --use-scheduler
```

### Repository structure
Folder `data` contains train and test data. Results, models' weights, test
predictions and used parameters are stored in `results`. Notebooks `eda.ipynb`
and `eval.ipynb` contain initial data analysis and models evaluation.
Script `run.py` performs the training, and the rest of Python files contain
training source code.

*Note*: models trained in my experiments aren't included in this repository.

### TensorBoard
Training/validation loss and accuracy can be visualized using TensorBoard. To run it execute:
```shell
tensorboard --logdir fnc-1-distilbert/results
```
TensorBoard dashboard will be then available under `http://localhost:6006` address in your browser.
