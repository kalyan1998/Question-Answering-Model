Spoken SQuAD Question Answering with BERT and DistilBERT

Base Model

    Dataset: Spoken SQuAD - A version of the SQuAD dataset where documents are in spoken form and questions are in text form.
    Model: BERT model implemented using the Huggingface library.
    Training Parameters:
        Learning Rate: 2e-5
        Weight Decay: 2e-2
        Batch Size: 16
        Epochs: 6
        Optimizer: AdamW
        Max Sequence Length: 512 tokens
    Loss Function: Focal loss is used to compute the loss during training by comparing the model's predictions to the true start and end positions of the answer spans.
    Evaluation Metrics: Word Error Rate (WER) and F1 Score.

Improvements with DistilBERT

In addition to the base BERT model, the DistilBERT model from Huggingface has also been used with the following improvements:

    Doc Stride: Implemented to handle long passages.
    Scheduler: An ExponentialLR scheduler is applied to adjust the learning rate during training.

The Jupyter Notebooks used for testing on noisy data includes two separate cells: one for evaluating on Noise V1 with a 44% WER, and another for evaluating on Noise V2 with a 54% WER.

Evaluation

The scripts are designed to train and evaluate the model if a saved version isn't found. Alternatively, you can skip the training phase by downloading pre-trained models from the following link and proceed directly to evaluation:
https://drive.google.com/drive/u/1/folders/1eP2yFAuo-JMGP-redinPNHFU47IiKqQb
