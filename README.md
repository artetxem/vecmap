VecMap (bilingual word embedding mappings)
==============

This is an open source implementation of our framework to learn bilingual word embedding mappings, described in the following papers:
- Mikel Artetxe, Gorka Labaka, and Eneko Agirre. 2016. **[Learning principled bilingual mappings of word embeddings while preserving monolingual invariance](https://aclweb.org/anthology/D16-1250)**. In *Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing*, pages 2289-2294.
- Mikel Artetxe, Gorka Labaka, and Eneko Agirre. 2017. **[Learning bilingual word embeddings with (almost) no bilingual data](http://aclweb.org/anthology/P17-1042)**. In *Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 451-462.

The first paper describes the general framework, whereas the second introduces a self-learning extension that enables training under very weak bilingual supervision (as little as a 25 word dictionary or an automatically generated list of numerals) with comparable results.

The package includes the tools necessary to map embeddings from one language into another as described in both papers, evaluation tools for word translation induction, word analogy and word similarity/relatedness, and scripts to reproduce the results reported there.

If you use this software for academic research, please cite the relevant paper(s):
```
@inproceedings{artetxe2016learning,
  author    = {Artetxe, Mikel  and  Labaka, Gorka  and  Agirre, Eneko},
  title     = {Learning principled bilingual mappings of word embeddings while preserving monolingual invariance},
  booktitle = {Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing},
  year      = {2016},
  pages     = {2289--2294}
}

@inproceedings{artetxe2017learning,
  author    = {Artetxe, Mikel  and  Labaka, Gorka  and  Agirre, Eneko},
  title     = {Learning bilingual word embeddings with (almost) no bilingual data},
  booktitle = {Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  year      = {2017},
  pages     = {451--462}
}
```


Requirements
--------
- Python 3
- NumPy
- SciPy

If you are having performance issues, make sure that your NumPy installation is properly linked to BLAS/LAPACK.


Usage
--------

Using the package typically involves the following 3 steps:

1. Normalize the source and target embeddings (`normalize_embeddings.py`). We recommend using length normalization followed by dimension-wise mean centering for best results as follows:
```
python3 normalize_embeddings.py unit center -i SRC_EMBEDDINGS.TXT -o SRC_EMBEDDINGS.NORMALIZED.TXT
python3 normalize_embeddings.py unit center -i TRG_EMBEDDINGS.TXT -o TRG_EMBEDDINGS.NORMALIZED.TXT
```
2. Map the source embeddings into the target embedding space (`map_embeddings.py`). We recommend using an orthogonal mapping (`--orthogonal`) for best results as follows:
```
python3 map_embeddings.py --orthogonal SRC_EMBEDDINGS.NORMALIZED.TXT TRG_EMBEDDINGS.NORMALIZED.TXT SRC_EMBEDDINGS.MAPPED.TXT TRG_EMBEDDINGS.MAPPED.TXT -d TRAIN_DICTIONARY.TXT
```
If your seed dictionary is small, you should enable the self-learning extension as follows (note that this might take a few hours):
```
python3 map_embeddings.py --orthogonal SRC_EMBEDDINGS.NORMALIZED.TXT TRG_EMBEDDINGS.NORMALIZED.TXT SRC_EMBEDDINGS.MAPPED.TXT TRG_EMBEDDINGS.MAPPED.TXT -d TRAIN_DICTIONARY.TXT --self_learning -v
```
If you have no dictionary at all, you can use shared numerals instead as follows:
```
python3 map_embeddings.py --orthogonal SRC_EMBEDDINGS.NORMALIZED.TXT TRG_EMBEDDINGS.NORMALIZED.TXT SRC_EMBEDDINGS.MAPPED.TXT TRG_EMBEDDINGS.MAPPED.TXT --numerals --self_learning -v
```
3. Evaluate the mapped embeddings. You can use `eval_translation.py` for evaluation in word translation induction, `eval_analogy.py` for evaluation in word analogy and `eval_similarity.py` for evaluation in word similarity/relatedness as follows:
```
python3 eval_translation.py SRC_EMBEDDINGS.MAPPED.TXT TRG_EMBEDDINGS.MAPPED.TXT -d TEST_DICTIONARY.TXT
python3 eval_analogy.py -l SRC_EMBEDDINGS.MAPPED.TXT -i TEST_ANALOGIES.TXT -t 30000
python3 eval_similarity.py -l --backoff 0 SRC_EMBEDDINGS.MAPPED.TXT TRG_EMBEDDINGS.MAPPED.TXT -i TEST_SIMILARITY.TXT
```

For additional options and more details on each of the tools, run them with the `--help` flag. We also recommend having a look at the experiment scripts described in the following section to understand how to run the most standard variants as described in the papers.


Reproducing results
--------

If you want to reproduce the results reported in our papers, simply clone the repository, download the datasets with the provided script, and run the appropriate experiment script as follows:

```
git clone https://github.com/artetxem/vecmap.git
cd vecmap
./get_data.sh
./reproduce_emnlp2016.sh
./reproduce_acl2017.sh
```

Note that the scripts save copies of all embeddings they produce, so you will need around 70GB of disk space in order to run all experiments.

The EMNLP 2016 script runs in a few minutes and produces the following output, which corresponds to Table 1 in the paper:

```
ORIGINAL EMBEDDINGS
  - EN AN  |  Coverage: 64.98%  Accuracy: 76.66% (sem: 79.66%, syn: 75.36%)
--------------------------------------------------------------------------------
UNCONSTRAINED MAPPING (Mikolov et al., 2013)
  - EN-IT  |  Coverage:100.00%  Accuracy: 34.93%
  - EN AN  |  Coverage: 64.98%  Accuracy: 73.80% (sem: 73.07%, syn: 74.12%)
UNCONSTRAINED MAPPING + LENGTH NORMALIZATION
  - EN-IT  |  Coverage:100.00%  Accuracy: 33.80%
  - EN AN  |  Coverage: 64.98%  Accuracy: 73.61% (sem: 72.92%, syn: 73.91%)
UNCONSTRAINED MAPPING + LENGTH NORMALIZATION + MEAN CENTERING
  - EN-IT  |  Coverage:100.00%  Accuracy: 38.47%
  - EN AN  |  Coverage: 64.98%  Accuracy: 73.71% (sem: 73.07%, syn: 73.99%)
--------------------------------------------------------------------------------
ORTHOGONAL MAPPING (Zhang et al., 2016)
  - EN-IT  |  Coverage:100.00%  Accuracy: 36.73%
  - EN AN  |  Coverage: 64.98%  Accuracy: 76.66% (sem: 79.66%, syn: 75.36%)
ORTHOGONAL MAPPING + LENGTH NORMALIZATION (Xing et al., 2015)
  - EN-IT  |  Coverage:100.00%  Accuracy: 36.87%
  - EN AN  |  Coverage: 64.98%  Accuracy: 76.66% (sem: 79.66%, syn: 75.36%)
ORTHOGONAL MAPPING + LENGTH NORMALIZATION + MEAN CENTERING (best)
  - EN-IT  |  Coverage:100.00%  Accuracy: 39.27%
  - EN AN  |  Coverage: 64.98%  Accuracy: 76.59% (sem: 79.63%, syn: 75.27%)
```

The ACL 2017 script runs in about 4 days in our small cluster and produces the following output, which corresponds to Table 1 and 2 in the paper (note that you might see some minor differences for the 25 word seed dictionary due to the underdetermined mapping when the dictionary size is smaller than the dimensionality of the embeddings):

```
--------------------------------------------------------------------------------
ENGLISH-ITALIAN
--------------------------------------------------------------------------------
5,000 WORD DICTIONARY
  - Mikolov et al. (2013a)  |  Translation: 34.93%  MWS353: 62.66%
  - Xing et al. (2015)      |  Translation: 36.87%  MWS353: 61.41%
  - Zhang et al. (2016)     |  Translation: 36.73%  MWS353: 61.62%
  - Artetxe et al. (2016)   |  Translation: 39.27%  MWS353: 61.74%
  - Proposed method         |  Translation: 39.67%  MWS353: 62.35%
25 WORD DICTIONARY
  - Mikolov et al. (2013a)  |  Translation:  0.00%  MWS353: -6.42%
  - Xing et al. (2015)      |  Translation:  0.00%  MWS353: 19.49%
  - Zhang et al. (2016)     |  Translation:  0.07%  MWS353: 15.52%
  - Artetxe et al. (2016)   |  Translation:  0.07%  MWS353: 17.45%
  - Proposed method         |  Translation: 37.27%  MWS353: 62.64%
NUMERAL DICTIONARY
  - Mikolov et al. (2013a)  |  Translation:  0.00%  MWS353: 28.75%
  - Xing et al. (2015)      |  Translation:  0.13%  MWS353: 27.75%
  - Zhang et al. (2016)     |  Translation:  0.27%  MWS353: 27.38%
  - Artetxe et al. (2016)   |  Translation:  0.40%  MWS353: 24.85%
  - Proposed method         |  Translation: 39.40%  MWS353: 62.82%

--------------------------------------------------------------------------------
ENGLISH-GERMAN
--------------------------------------------------------------------------------
5,000 WORD DICTIONARY
  - Mikolov et al. (2013a)  |  Translation: 35.00%  MWS353: 52.75%  RG65: 64.29%
  - Xing et al. (2015)      |  Translation: 41.27%  MWS353: 59.54%  RG65: 70.03%
  - Zhang et al. (2016)     |  Translation: 40.80%  MWS353: 59.63%  RG65: 70.41%
  - Artetxe et al. (2016)   |  Translation: 41.87%  MWS353: 59.71%  RG65: 71.56%
  - Proposed method         |  Translation: 40.87%  MWS353: 61.62%  RG65: 74.20%
25 WORD DICTIONARY
  - Mikolov et al. (2013a)  |  Translation:  0.00%  MWS353: -6.30%  RG65: -9.13%
  - Xing et al. (2015)      |  Translation:  0.07%  MWS353: 13.13%  RG65: 21.71%
  - Zhang et al. (2016)     |  Translation:  0.13%  MWS353: 11.72%  RG65: 18.96%
  - Artetxe et al. (2016)   |  Translation:  0.13%  MWS353:  9.08%  RG65: 25.84%
  - Proposed method         |  Translation: 39.60%  MWS353: 61.19%  RG65: 74.91%
NUMERAL DICTIONARY
  - Mikolov et al. (2013a)  |  Translation:  0.07%  MWS353: 24.17%  RG65: 30.71%
  - Xing et al. (2015)      |  Translation:  0.53%  MWS353: 29.03%  RG65: 34.57%
  - Zhang et al. (2016)     |  Translation:  0.87%  MWS353: 27.92%  RG65: 32.82%
  - Artetxe et al. (2016)   |  Translation:  0.73%  MWS353: 30.45%  RG65: 35.91%
  - Proposed method         |  Translation: 40.27%  MWS353: 60.38%  RG65: 73.88%

--------------------------------------------------------------------------------
ENGLISH-FINNISH
--------------------------------------------------------------------------------
5,000 WORD DICTIONARY
  - Mikolov et al. (2013a)  |  Translation: 25.91%
  - Xing et al. (2015)      |  Translation: 28.23%
  - Zhang et al. (2016)     |  Translation: 28.16%
  - Artetxe et al. (2016)   |  Translation: 30.62%
  - Proposed method         |  Translation: 28.72%
25 WORD DICTIONARY
  - Mikolov et al. (2013a)  |  Translation:  0.00%
  - Xing et al. (2015)      |  Translation:  0.07%
  - Zhang et al. (2016)     |  Translation:  0.14%
  - Artetxe et al. (2016)   |  Translation:  0.21%
  - Proposed method         |  Translation: 28.16%
NUMERAL DICTIONARY
  - Mikolov et al. (2013a)  |  Translation:  0.00%
  - Xing et al. (2015)      |  Translation:  0.56%
  - Zhang et al. (2016)     |  Translation:  0.42%
  - Artetxe et al. (2016)   |  Translation:  0.77%
  - Proposed method         |  Translation: 26.47%
```


License
-------

Copyright (C) 2016-2017, Mikel Artetxe

Licensed under the terms of the GNU General Public License, either version 3 or (at your option) any later version. A full copy of the license can be found in LICENSE.txt.
