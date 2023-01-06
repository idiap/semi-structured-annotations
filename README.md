# semi-structured-annotations
This repository contains the code and links to the data and trained models for the paper [A Corpus and Evaluation for Predicting Semi-Structured Human Annotations](#) presented at the GEM workshop at EMNLP 2022.

## Contents
1. [Short Description](#short-description)
2. [Data](#data)
3. [Models](#models)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Contact](#contact)
7. [Authors and Acknowledgments](#authors-and-acknowledgments)
8. [Citation](#citation)

## Short Description
Our goal is to teach seq2seq models to interpret policy announcements. We present the FOMC dataset on the monetary policy of the Federal Reserve, where source documents are policy announcements and targets are selected and annotated sentences from New York Times articles. We train seq2seq models (Transformer, BERT, BART) to generate the annotated targets conditioned on the source documents. We also introduce an evaluation method called *equivalence classes evaluation*. Equivalence classes group semantically interchangeable values from a specific annotation category. The seq2seq model then has to identify the true continuation among two possibilities from different equivalence classes.    

## Data
Please contact me at andreas.marfurt [at] idiap.ch to get access to the data.

## Models
We provide the following checkpoints of models finetuned on the FOMC dataset:
- [Transformer](https://drive.switch.ch/index.php/s/WJKe5e7XUOO3BXA): Randomly initialized Transformer
- [BERT](https://drive.switch.ch/index.php/s/1bgFsYSFq7WgX4L): BERT encoder, Transformer decoder
- [BART](https://drive.switch.ch/index.php/s/Zsj9LP3SaHZC93C): BART model
- [FilterBERT](https://drive.switch.ch/index.php/s/5zLt95szMXg1Yu3): BERT-based model for filtering source documents

The models are shared under a [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license.

## Installation
First, install conda, e.g. from [Miniconda](https://docs.conda.io/en/latest/miniconda.html). Then create and activate the environment:
```
conda env create -f environment.yml
conda activate semi-structured-annotations
```

## Usage
### Training
To train a model, use the [main.py](main.py) script. The default arguments are set to the hyperparameter values we used in our experiments. Here's an example of how to train BART with the parameters we used ourselves:
```
python main.py \
--model bart \
--data_dir data_fomc_bart \
--model_dir models/bart \
--default_root_dir logs/bart \
--deterministic \
--gpus 1 \
--batch_size 2 \
--accumulate_grad_batches 2 \
--max_epochs 20 \
--min_epochs 10 \
--max_steps 16000
```

### Filtering
You can filter source documents with FilterBERT, or the Oracle/Lead strategies. For the former, use the [filter_bert.py](filter_bert.py) script:
```
python filter_bert.py \
--model_dir models/filterbert \
--pretrained_dir bert-base-uncased \
--data_dir data_fomc_bert \
--default_root_dir logs/filterbert \
--deterministic \
--gpus 1 \
--batch_size 5 \
--max_epochs 10 \
--min_epochs 5 \
--max_steps 17000
```

For Oracle/Lead filtering, use [filter_source_docs_with_tokenizer.py](filter_source_docs_with_tokenizer.py) and specify a HuggingFace tokenizer.

### Saving Model Outputs
To run text generation evaluation, you have to first save a model's outputs in text format. Run [save_model_outputs.py](save_model_outputs.py) for Transformer/BERT models or [save_bart_outputs.py](save_bart_outputs.py) for a BART model with the default parameters. Don't forget to specify `model_dir` and `output_dir`. 

### Text Generation Evaluation
Use the [evaluations.py](evaluations.py) script to run the text generation evaluations and specify the path to your model outputs as the `input_dir`. Results will be saved as a json file in the same directory.

### Equivalence Classes Evaluation
Our definition of equivalence classes can be found in [equivalence_classes.json](equivalence_classes.json). We provide the evaluation instances we used in [data_fomc_equiv](data_fomc_equiv). If you want to generate your own, use the [create_equivalance_classes_examples.py](create_equivalence_classes_examples.py) script.

Run the evaluation with the [equivalence_classes_evaluation.py](equivalence_classes_evaluation.py) file by specifying the path to your evaluation data, the model directory and the output path.

## Contact
In case of problems or questions open a Github issue or write an email to andreas.marfurt [at] idiap.ch.

## Authors and Acknowledgments
Our paper was written by Andreas Marfurt, Ashley Thornton, David Sylvan, Lonneke van der Plas and James Henderson.

The work was supported as a part of the grant Automated interpretation of political and economic policy documents: Machine learning using semantic and syntactic information, funded by the Swiss National Science Foundation (grant number CRSII5_180320), and led by the co-PIs James Henderson, Jean-Louis Arcand and David Sylvan. We would also like to thank Maria Kamran, Alessandra Romani, Julia Greene, Clarisse Labbé, Shekhar Hari Kumar, Claire Ransom, Daniele Rinaldo, Eugenia Zena and Raphael Leduc for their invaluable data collection and annotation efforts.

## Citation
If you use our code, data or models, please cite us.
```
@inproceedings{marfurt-etal-2022-corpus,
    title = "A Corpus and Evaluation for Predicting Semi-Structured Human Annotations",
    author = "Marfurt, Andreas  and
      Thornton, Ashley  and
      Sylvan, David  and
      van der Plas, Lonneke  and
      Henderson, James",
    booktitle = "Proceedings of the Second Workshop on Generation, Evaluation and Metrics",
    month = dec,
    year = "2022",
    publisher = "Association for Computational Linguistics",    
}
```
