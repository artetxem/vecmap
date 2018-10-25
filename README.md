VecMap (cross-lingual word embedding mappings)
==============

This is an open source implementation of our framework to learn cross-lingual word embedding mappings, described in the following papers:
- Mikel Artetxe, Gorka Labaka, and Eneko Agirre. 2018. **[A robust self-learning method for fully unsupervised cross-lingual mappings of word embeddings](https://aclweb.org/anthology/P18-1073)**. In *Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*.
- Mikel Artetxe, Gorka Labaka, and Eneko Agirre. 2018. **[Generalizing and improving bilingual word embedding mappings with a multi-step framework of linear transformations](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16935/16781)**. In *Proceedings of the Thirty-Second AAAI Conference on Artificial Intelligence (AAAI-18)*, pages 5012-5019.
- Mikel Artetxe, Gorka Labaka, and Eneko Agirre. 2017. **[Learning bilingual word embeddings with (almost) no bilingual data](https://aclweb.org/anthology/P17-1042)**. In *Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 451-462.
- Mikel Artetxe, Gorka Labaka, and Eneko Agirre. 2016. **[Learning principled bilingual mappings of word embeddings while preserving monolingual invariance](https://aclweb.org/anthology/D16-1250)**. In *Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing*, pages 2289-2294.

The package includes a script to build cross-lingual word embeddings with or without parallel data as described in the papers, as well as evaluation tools in word translation induction, word similarity/relatedness and word analogy.

If you use this software for academic research, [please cite the relevant paper(s)](#publications).


Requirements
--------
- Python 3
- NumPy
- SciPy
- CuPy (optional, only required for CUDA support)


Usage
--------

In order to build your own cross-lingual word embeddings, you should first train monolingual word embeddings for each language using your favorite tool (e.g. [word2vec](https://github.com/tmikolov/word2vec) or [fasttext](https://github.com/facebookresearch/fastText)) and then map them to a common space with our software as described below. Having done that, you can evaluate the resulting cross-lingual embeddings using our included tools as discussed next.

#### Mapping

The mapping software offers 4 main modes with our recommended settings for different scenarios:

- **Supervised** (recommended if you have a large training dictionary):
```
python3 map_embeddings.py --supervised TRAIN.DICT SRC.EMB TRG.EMB SRC_MAPPED.EMB TRG_MAPPED.EMB
```
- **Semi-supervised** (recommended if you have a small seed dictionary):
```
python3 map_embeddings.py --semi_supervised TRAIN.DICT SRC.EMB TRG.EMB SRC_MAPPED.EMB TRG_MAPPED.EMB
```
- **Identical** (recommended if you have no seed dictionary but can rely on identical words):
```
python3 map_embeddings.py --identical SRC.EMB TRG.EMB SRC_MAPPED.EMB TRG_MAPPED.EMB
```
- **Unsupervised** (recommended if you have no seed dictionary and do not want to rely on identical words):
```
python3 map_embeddings.py --unsupervised SRC.EMB TRG.EMB SRC_MAPPED.EMB TRG_MAPPED.EMB
```

`SRC.EMB` and `TRG.EMB` refer to the input monolingual embeddings, which should be in the word2vec text format, whereas `SRC_MAPPED.EMB` and `TRG_MAPPED.EMB` refer to the output cross-lingual embeddings. The training dictionary `TRAIN.DICT`, if any, should be given as a text file with one entry per line (source word + whitespace + target word).

If you have a NVIDIA GPU, append the `--cuda` flag to the above commands to make things faster.

For most users, the above settings should suffice. Choosing the right mode should be straightforward depending on the resources available: as a general rule, you should prefer the mode with the highest supervision for the resources you have, although it is advised to try different variants in case of doubt.

In addition to these recommended modes, the software also offers additional options to adjust different aspects of the mapping method as described in the papers. While most users should not need to deal with those, you can learn more about them by running the tool with the `--help` flag. You can either use one of the recommended modes and modify a few options on top of it, or do not use any recommended mode and set all options yourself. In fact, if you dig into the code, you will see that the above modes simply set recommended defaults for all the different options.

#### Evaluation

You can evaluate your mapped embeddings in bilingual lexicon extraction (aka dictionary induction or word translation) as follows:
```
python3 eval_translation.py SRC_MAPPED.EMB TRG_MAPPED.EMB -d TEST.DICT
```
The above command uses standard nearest neighbor retrieval by default. For best results, it is recommended that you use CSLS retrieval instead:
```
python3 eval_translation.py SRC_MAPPED.EMB TRG_MAPPED.EMB -d TEST.DICT --retrieval csls
```
While better, CSLS is also significantly slower than nearest neighbor, so do not forget to append the `--cuda` flag to the above command if you have a NVIDIA GPU.

In addition to bilingual lexicon extraction, you can also evaluate your mapped embeddings in cross-lingual word similarity as follows:
```
python3 eval_similarity.py -l --backoff 0 SRC_MAPPED.EMB TRG_MAPPED.EMB -i TEST_SIMILARITY.TXT
```

Finally, we also offer an evaluation tool for monolingual word analogies, which mimics the one included with word2vec but should run significantly faster:
```
python3 eval_analogy.py -l SRC_MAPPED.EMB -i TEST_ANALOGIES.TXT -t 30000
```


Dataset
--------
You can use the following script to download the main dataset used in our papers, which is an extension of that of [Dinu et al. (2014)](http://clic.cimec.unitn.it/~georgiana.dinu/down/):
```
./get_data.sh
```


Reproducing results
--------

While we always recommend to use the above settings for best results when working with your own embeddings, we also offer additional modes to replicate the systems from our different papers as follows:
- **ACL 2018** (currently equivalent to the unsupervised mode):
```
python3 map_embeddings.py --acl2018 SRC.EMB TRG.EMB SRC_MAPPED.EMB TRG_MAPPED.EMB
```
- **AAAI 2018** (currently equivalent to the supervised mode, except for minor differences in re-weighting, normalization and dimensionality reduction):
```
python3 map_embeddings.py --aaai2018 TRAIN.DICT SRC.EMB TRG.EMB SRC_MAPPED.EMB TRG_MAPPED.EMB
```
- **ACL 2017** (superseded by our ACL 2018 system; offers 2 modes depending on the initialization):
```
python3 map_embeddings.py --acl2017 SRC.EMB TRG.EMB SRC_MAPPED.EMB TRG_MAPPED.EMB
python3 map_embeddings.py --acl2017_seed TRAIN.DICT SRC.EMB TRG.EMB SRC_MAPPED.EMB TRG_MAPPED.EMB
```
- **EMNLP 2016** (superseded by our AAAI 2018 system):
```
python3 map_embeddings.py --emnlp2016 TRAIN.DICT SRC.EMB TRG.EMB SRC_MAPPED.EMB TRG_MAPPED.EMB
```


FAQ
--------

##### How long does training take?

- The supervised mode (`--supervised`) should run in around 2 minutes in either CPU or GPU.
- The rest of recommended modes (either `--semi_supervised`, `--identical` or `--unsupervised`) should run in around 5 hours in CPU, or 10 minutes in GPU (Titan Xp or similar).


##### This is running much slower for me! What can I do?

1. If you have a GPU, do not forget the `--cuda` flag.
2. Make sure that your NumPy installation is properly linked to BLAS/LAPACK. This is particularly important if you are working on CPU, as it can have a huge impact in performance if not properly set up.
3. There are different settings that affect the execution time of the algorithm and can thus be adjusted to make things faster: the batch size (`--batch_size`), the vocabulary cutoff (`--vocabulary_cutoff`), the stochastic dictionary induction settings (`--stochastic_initial`, `--stochastic_multiplier` and `--stochastic_interval`) and the convergence threshold (`--threshold`), among others. However, most of these settings will have a direct impact in the quality of the resulting embeddings, so you should not play with them unless you really know what you are doing.


##### Prior versions of this software included nice scripts to reproduce the exact same results reported in your papers. Why are those missing now?

As the complexity of the software (and the number of publications/results to reproduce) increased, maintaining those nice scripts became very tedious. Moreover, with the inclusion of CUDA support and FP32 precision, reproducing the exact same results on different platforms became inviable due to minor numerical variations in the underlying computations, which were magnified by self-learning (e.g. the exact same command is likely to produce a slightly different output on CPU and GPU). While the effect in the final results is negligible (the observed variations are around 0.1-0.2 accuracy points), this made it unfeasible to reproduce the exact same results in different platforms.

Instead of that, we now provide an [easy interface to run all the systems proposed in our different papers](#reproducing-results). We think that this might be even more useful than the previous approach: the most skeptical user should still be able to easily verify our results, while we also provide a simple interface to test our different systems in other datasets.


##### The ablation test in your ACL 2018 paper reports 0% accuracies for removing CSLS, but I am getting better results. Why is that?

After publishing the paper, we discovered a bug in the code that was causing those 0% accuracies. Now that the bug is fixed, the effect of removing CSLS is not that dramatic, although it still has a big negative impact. At the same time, the effect of removing the bidirectional dictionary induction in that same ablation test is slightly smaller.


See also
--------

VecMap is a basic building block of [Monoses](https://github.com/artetxem/monoses), our Unsupervised Statistical Machine Translation system. You can use them in combination to train your own machine translation model from monolingual corpora alone.


Publications
--------

If you use this software for academic research, please cite the relevant paper(s) as follows (in case of doubt, please cite the ACL 2018 paper, or the AAAI 2018 paper if you use the supervised mode):
```
@inproceedings{artetxe2018acl,
  author    = {Artetxe, Mikel  and  Labaka, Gorka  and  Agirre, Eneko},
  title     = {A robust self-learning method for fully unsupervised cross-lingual mappings of word embeddings},
  booktitle = {Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  year      = {2018},
  pages     = {789--798}
}

@inproceedings{artetxe2018aaai,
  author    = {Artetxe, Mikel  and  Labaka, Gorka  and  Agirre, Eneko},
  title     = {Generalizing and improving bilingual word embedding mappings with a multi-step framework of linear transformations},
  booktitle = {Proceedings of the Thirty-Second AAAI Conference on Artificial Intelligence},
  year      = {2018},
  pages     = {5012--5019}
}

@inproceedings{artetxe2017acl,
  author    = {Artetxe, Mikel  and  Labaka, Gorka  and  Agirre, Eneko},
  title     = {Learning bilingual word embeddings with (almost) no bilingual data},
  booktitle = {Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  year      = {2017},
  pages     = {451--462}
}

@inproceedings{artetxe2016emnlp,
  author    = {Artetxe, Mikel  and  Labaka, Gorka  and  Agirre, Eneko},
  title     = {Learning principled bilingual mappings of word embeddings while preserving monolingual invariance},
  booktitle = {Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing},
  year      = {2016},
  pages     = {2289--2294}
}
```


License
-------

Copyright (C) 2016-2018, Mikel Artetxe

Licensed under the terms of the GNU General Public License, either version 3 or (at your option) any later version. A full copy of the license can be found in LICENSE.txt.
