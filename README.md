VecMap (bilingual word embedding mappings)
==============

This is an open source implementation of our framework to learn bilingual word embedding mappings, described in the following paper:

Mikel Artetxe, Gorka Labaka, and Eneko Agirre. 2016. **Learning principled bilingual mappings of word embeddings while preserving monolingual invariance**. In *Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing (EMNLP 2016)*.

The package includes the tools necessary to project embeddings from one language into another as described in the paper, evaluation tools for word analogy and word translation induction, and a script to reproduce the results reported there.

If you use this software for academic research, please cite the paper in question:
```
@inproceedings{artetxe2016learning,
  title={Learning principled bilingual mappings of word embeddings while preserving monolingual invariance},
  author={Artetxe, Mikel and Labaka, Gorka and Agirre, Eneko},
  booktitle={Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing},
  year={2016}
}
```


Requirements
--------
- Python 3
- NumPy


Usage
--------

If you want to reproduce the results reported in the paper, simply clone the repository and run the experiment script:

```
git clone https://github.com/artetxem/vecmap.git
cd vecmap
./run_experiment.sh
```

The script will automatically download the appropriate English-Italian dataset, train different projections on it, and evaluate them on English-Italian word translation induction and English word analogy. Take a coffee or two and, when you come back, you should see the following results, which correspond to Table 1 in the paper:

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
ORTHOGONAL MAPPING
  - EN-IT  |  Coverage:100.00%  Accuracy: 36.73%
  - EN AN  |  Coverage: 64.98%  Accuracy: 76.66% (sem: 79.66%, syn: 75.36%)
ORTHOGONAL MAPPING + LENGTH NORMALIZATION (Xing et al., 2015)
  - EN-IT  |  Coverage:100.00%  Accuracy: 36.87%
  - EN AN  |  Coverage: 64.98%  Accuracy: 76.66% (sem: 79.66%, syn: 75.36%)
ORTHOGONAL MAPPING + LENGTH NORMALIZATION + MEAN CENTERING (best)
  - EN-IT  |  Coverage:100.00%  Accuracy: 39.27%
  - EN AN  |  Coverage: 64.98%  Accuracy: 76.59% (sem: 79.63%, syn: 75.27%)
```

If you want to work with your own settings or dataset instead, you should follow the following steps:

1. Normalize the source and target embeddings (`normalize_embeddings.py`). We recommend using length normalization followed by dimension-wise mean centering for best results.
2. Project the source embeddings into the target embedding space (`project_embeddings.py`). We recommend using an orthogonal mapping for best results.
3. Evaluate the projected embeddings (`eval_translation.py` for bilingual evaluation in word translation induction and `eval_analogy.py` for monolingual evaluation in analogy).

This can be done running the following commands:
```
# Normalize the source and target embeddings
python3 normalize_embeddings.py unit center -i SRC_EMBEDDINGS.TXT -o SRC_EMBEDDINGS.NORMALIZED.TXT
python3 normalize_embeddings.py unit center -i TRG_EMBEDDINGS.TXT -o TRG_EMBEDDINGS.NORMALIZED.TXT

# Project the source embeddings into the target embedding space
python3 project_embeddings.py --orthogonal SRC_EMBEDDINGS.NORMALIZED.TXT TRG_EMBEDDINGS.NORMALIZED.TXT -d TRAIN_DICTIONARY.TXT -o SRC_EMBEDDINGS.NORMALIZED.PROJECTED.TXT

# Evaluate the projected embeddings in a bilingual word translation induction task
python3 eval_translation.py SRC_EMBEDDINGS.NORMALIZED.PROJECTED.TXT TRG_EMBEDDINGS.NORMALIZED.TXT -d TEST_DICTIONARY.TXT

# Evaluate the projected embeddings in a monolingual analogy task
python3 eval_analogy.py -l SRC_EMBEDDINGS.NORMALIZED.PROJECTED.TXT -i TEST_ANALOGIES.TXT -t 30000
```

For more details on each of the tools, run them with the `--help` flag. We also recommend having a look at the experiment script to understand how to run the most standard variants as described in the paper.


License
-------

Copyright (C) 2016-2017, Mikel Artetxe

Licensed under the terms of the GNU General Public License, either version 3 or (at your option) any later version. A full copy of the license can be found in LICENSE.txt.
