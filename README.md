# Spoken SQuAD Question Answering with BERT and DistilBERT

## Project Description

This project implements a question-answering system using BERT and DistilBERT models to handle spoken documents. The models are fine-tuned on the Spoken SQuAD dataset, which includes spoken form documents and text questions, to improve their ability to understand and generate accurate answers in noisy conditions.

## Workflow Overview

1. **Data Collection and Preparation**
   - **Dataset**: Spoken SQuAD - A version of the SQuAD dataset where documents are in spoken form and questions are in text form.
   - **Preprocessing Steps**:
     - Data Extraction: Reads the dataset from JSON, extracting and lowercasing text for passages, questions, and answers, and noting answer locations.
     - Custom Dataset: Defines a PyTorch Dataset class, SQAD, for model input management based on tokenized data.
     - Position Calculation: Identifies the token positions for answer starts and ends within tokenized contexts.
     - Text Normalization: Cleans text by stripping extraneous characters and standardizing format for evaluation.

2. **Base Model (BERT)**
   - **Model**: BERT model implemented using the Huggingface library.
   - **Training Parameters**:
     - Learning Rate: 2e-5
     - Weight Decay: 2e-2
     - Batch Size: 16
     - Epochs: 6
     - Optimizer: AdamW
     - Max Sequence Length: 512 tokens
   - **Loss Function**: Focal loss is used to compute the loss during training by comparing the model's predictions to the true start and end positions of the answer spans.
   - **Evaluation Metrics**: Word Error Rate (WER) and F1 Score.

3. **Improvements with DistilBERT**
   - **Model**: DistilBERT model from Huggingface
   - **Improvements**:
     - **Doc Stride**: Implemented to handle long passages by splitting them into smaller chunks with overlap.
     - **Scheduler**: An ExponentialLR scheduler is applied to adjust the learning rate during training.
   - **Evaluation on Noisy Data**:
     - Evaluated on Noise V1 with a 44% WER.
     - Evaluated on Noise V2 with a 54% WER.

4. **Results**
   - Performance comparison across clean and noisy conditions:
     - **Base Model (BERT)**: Evaluated without improvements.
     - **Improvement 1**: Added Doc Stride.
     - **Improvement 2**: Added Doc Stride and Scheduler.
     - **Improvement 3**: Added Doc Stride, Scheduler, and used a stronger pretrained model (bert-large-uncased-whole-word-masking-finetuned-squad).

## File Descriptions

### `data_preprocessing.py`
- Handles data extraction, normalization, and creation of custom datasets for training and evaluation.

### `train_and_test.py`
- Defines model configurations, training loops, and evaluation metrics for both BERT and DistilBERT models.

### `evaluate_model.py`
- Contains the evaluation logic to compute WER and F1 scores on test datasets.

## Instructions to Execute Code

### 1) Training the Model

To train the model from scratch, provide the path to the training data directory and the training labels JSON file as arguments to the shell script. For example:
```bash
sh hw3_train.sh /path/to/training_data training_label.json
```
Replace /path/to/training_data with the actual path to your training data directory. The script will train the model using the specified data and labels, generating necessary files such as the model file and object files (*.obj) for later use.

### 2) Testing the Model
To test the model, specify the path to the testing data directory, which should include a feat subdirectory containing the video features. Additionally, provide the name of the output file where the test results will be stored. For instance:

```bash
sh hw3_test.sh /path/to/testing_data output.txt
```
Replace /path/to/testing_data with the actual path to your testing data directory containing a feat directory with video features. The script will use the pre-trained model to generate captions for the test data and save the results in output.txt.

### 3) Downloading Necessary Files
Before testing the model, ensure you have downloaded all necessary files, including the model file (.h5), object files (.obj), and the testing_label.json file. If these files are not available, you can train the model again as described in step 1, which will generate these files.

Note:

The model file is saved as Shiva_HW3_model0.h5 by default. If you have a different model file, you may need to modify the script to load the correct file.
If you encounter an error with the model downloaded using wget from hw3_train.sh, it could be due to extraction issues. In such cases, use the direct download link from Google Drive provided https://drive.google.com/drive/u/1/folders/1eP2yFAuo-JMGP-redinPNHFU47IiKqQb. Additionally, ensure that you run the model on a device with CUDA support, as it was trained using CUDA technology.
