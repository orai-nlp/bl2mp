# How Well Can BERT Learn the Grammar of an Agglutinative and Flexible-Order Language? The Case of Basque.

Data (BL2MP dataset and pretraining corpora), models and evaluation scripts from our work [*How Well Can BERT Learn the Grammar of an Agglutinative and Flexible-Order Language? The Case of Basque.*]() accepted at LREC-COLING2024.


## BL2MP (Basque L2 student-based Minimal Pairs):

We introduce BL2MP (Basque L2 student-based Minimal Pairs), designed to assess the grammatical knowledge of language Models in the Basque language, inspired by the BLiMP benchmark. The BL2MP dataset includes examples sourced from the bai&by language academy, derived from essays written by students enrolled at the academy. These instances provide a wealth of authentic and natural grammatical errors, representing genuine mistakes made by learners and thus offering a realistic reflection of real-world language errors.

*BL2MP* is also available on [HuggingFace 🤗](https://huggingface.co/datasets/orai-nlp/bl2mp)

## Pretraining corpora:

We employed three corpora of different sizes (5M, 25M, 125M) in our experiments, and we share it in its raw and lemmatized versions:

Raw versions:
[5M](https://storage.googleapis.com/orai-nlp/bl2mp/corpora/eu_5M.txt), 
[25M](https://storage.googleapis.com/orai-nlp/bl2mp/corpora/eu_25M.txt), 
[125M](https://storage.googleapis.com/orai-nlp/bl2mp/corpora/eu_125M.txt)

Lemmatized versions: [5M_lemma](https://storage.googleapis.com/orai-nlp/bl2mp/corpora/eu_5M_lemma.txt), 
[25M_lemma](https://storage.googleapis.com/orai-nlp/bl2mp/corpora/eu_25M_lemma.txt), 
[125M_lemma](https://storage.googleapis.com/orai-nlp/bl2mp/corpora/eu_125M_lemma.txt)

MLM validation datasets:
[MLM_val](https://storage.googleapis.com/orai-nlp/bl2mp/corpora/eu_argia_test.txt), 
[MLM_val_lemma](https://storage.googleapis.com/orai-nlp/bl2mp/corpora/eu_argia_test_lemma.txt)

## Models

We trained 3 BERT models of different sizes, namely mini, medium and base (with 4L, 8L and 12L respectively) with each corpus.

We share the best performing checkpoints for each model:

[bert_mini_eu_5M](https://storage.googleapis.com/orai-nlp/bl2mp/models/bert_mini_eu_5M_ckpt-409600.tar.xz),
[bert_mini_eu_25M](https://storage.googleapis.com/orai-nlp/bl2mp/models/bert_mini_eu_25M_ckpt-512000.tar.xz),
[bert_mini_eu_125M](https://storage.googleapis.com/orai-nlp/bl2mp/models/bert_mini_eu_125M_ckpt-640000.tar.xz)

[bert_medium_eu_5M](https://storage.googleapis.com/orai-nlp/bl2mp/models/bert_medium_eu_5M_ckpt-51200.tar.xz),
[bert_medium_eu_25M](https://storage.googleapis.com/orai-nlp/bl2mp/models/bert_medium_eu_25M_ckpt-256000.tar.xz),
[bert_medium_eu_125M](https://storage.googleapis.com/orai-nlp/bl2mp/models/bert_medium_eu_125M_ckpt-640000.tar.xz)

[bert_base_eu_5M](https://storage.googleapis.com/orai-nlp/bl2mp/models/bert_base_eu_5M_ckpt-51200.tar.xz),
[bert_base_eu_25M](https://storage.googleapis.com/orai-nlp/bl2mp/models/bert_base_eu_25M_ckpt-128000.tar.xz),
[bert_base_eu_125M](https://storage.googleapis.com/orai-nlp/bl2mp/models/bert_base_eu_125M_ckpt-640000.tar.xz)

We also share the models trained with the lemmatized version: 

[bert_medium_eu_5M_lemma](https://storage.googleapis.com/orai-nlp/bl2mp/models/bert_medium_eu_5M_lemma_ckpt-51200.tar.xz),
[bert_medium_eu_25M_lemma](https://storage.googleapis.com/orai-nlp/bl2mp/models/bert_medium_eu_25M_lemma_ckpt-512000.tar.xz),
[bert_medium_eu_125M_lemma](https://storage.googleapis.com/orai-nlp/bl2mp/models/bert_medium_eu_125M_lemma_ckpt-640000.tar.xz)



## Evaluation Script usage:

We used [minicons](https://github.com/kanishkamisra/minicons) to evaluate (0-shot) our MLMs on BL2MP, which uses Salazar et al. (2020) to score sentences.

It can be installed with pip:

```
pip install minicons
```

And the we evaluate a MLMs as follows:

```
python3 mlm-score.py  --input bl2mpjsonl --output_dir output/ --lm orai-nlp/ElhBERTeu-medium --device cuda:0
```

There are different versions of the dataset and evaluation script, created for different experiments, but all of them use the same call to minicons, and differ only in reading the input data, and the conditions set to filter minimal-pairs to compute the final accuracy score.


Authors
-----------
Gorka Urbizu [1] [2], Muitze Zulaika [1], Xabier Saralegi [1], Ander Corral [1]

Affiliation of the authors: 

[1] Orai NLP Technologies

[2] University of the Basque Country



Licensing
-------------

Copyright (C) by Orai NLP Technologies. 

The corpora, datasets, models and scripts created in this work, are licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0.

International License (CC BY-NC-SA 4.0). To view a copy of this license, visit [http://creativecommons.org/licenses/by-nc-sa/4.0/](https://creativecommons.org/licenses/by-nc-sa/4.0/).



Acknowledgements
-------------------
If you use these corpora, datasets or models please cite the following paper:

- G. Urbizu, M. Zulaika, X. Saralegi, A. Corral. How Well Can BERT Learn the Grammar of an Agglutinative and Flexible-Order Language? The Case of Basque. The 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING2024). May, 2024. Torino, Italia



Contact information
-----------------------
Gorka Urbizu, Muitze Zulaika: {g.urbizu,m.zulaika}@orai.eus