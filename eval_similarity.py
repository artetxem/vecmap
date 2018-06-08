# Copyright (C) 2017-2018  Mikel Artetxe <artetxem@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import embeddings

import argparse
import numpy as np
import os
import scipy.stats
import sys


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate embeddings in word similarity/relatedness')
    parser.add_argument('src_embeddings', help='the source language embeddings')
    parser.add_argument('trg_embeddings', nargs='?', help='the target language embeddings')
    parser.add_argument('-i', '--input', default=[sys.stdin.fileno()], nargs='+', help='the input datasets (defaults to stdin)')
    parser.add_argument('-l', '--lowercase', action='store_true', help='lowercase the words in the test files')
    parser.add_argument('--backoff', default=None, type=float, help='use a backoff similarity score for OOV entries')
    parser.add_argument('--encoding', default='utf-8', help='the character encoding for input/output (defaults to utf-8)')
    parser.add_argument('--precision', choices=['fp16', 'fp32', 'fp64'], default='fp32', help='the floating-point precision (defaults to fp32)')
    parser.add_argument('--sim', nargs='*', help='the names of the datasets to include in the similarity results')
    parser.add_argument('--rel', nargs='*', help='the names of the datasets to include in the relatedness results')
    parser.add_argument('--all', nargs='*', help='the names of the datasets to include in the total results')
    args = parser.parse_args()

    # Choose the right dtype for the desired precision
    if args.precision == 'fp16':
        dtype = 'float16'
    elif args.precision == 'fp32':
        dtype = 'float32'
    elif args.precision == 'fp64':
        dtype = 'float64'

    # Parse test files
    word_pairs = []
    golds = []
    for filename in args.input:
        f = open(filename, encoding=args.encoding, errors='surrogateescape')
        word_pairs.append([])
        golds.append([])
        for line in f:
            if args.lowercase:
                line = line.lower()
            src, trg, score = line.split('\t')
            word_pairs[-1].append((src, trg))
            golds[-1].append(float(score))

    # Build vocabularies
    src_vocab = {pair[0] for pairs in word_pairs for pair in pairs}
    trg_vocab = {pair[1] for pairs in word_pairs for pair in pairs}

    # Read embeddings
    srcfile = open(args.src_embeddings, encoding=args.encoding, errors='surrogateescape')
    trgfile = open(args.src_embeddings if args.trg_embeddings is None else args.trg_embeddings, encoding=args.encoding, errors='surrogateescape')
    src_words, src_matrix = embeddings.read(srcfile, vocabulary=src_vocab, dtype=dtype)
    trg_words, trg_matrix = embeddings.read(trgfile, vocabulary=trg_vocab, dtype=dtype)

    # Length normalize embeddings so their dot product effectively computes the cosine similarity
    embeddings.length_normalize(src_matrix)
    embeddings.length_normalize(trg_matrix)

    # Build word to index map
    src_word2ind = {word: i for i, word in enumerate(src_words)}
    trg_word2ind = {word: i for i, word in enumerate(trg_words)}

    # Compute system scores and correlations
    results = []
    for i in range(len(golds)):
        system = []
        gold = []
        oov = 0
        for gold_score, (src, trg) in zip(golds[i], word_pairs[i]):
            try:
                cos = np.dot(src_matrix[src_word2ind[src]], trg_matrix[trg_word2ind[trg]])
                system.append(cos)
                gold.append(gold_score)
            except KeyError:
                if args.backoff is None:
                    oov += 1
                else:
                    system.append(args.backoff)
                    gold.append(gold_score)
        name = os.path.splitext(os.path.basename(args.input[i]))[0]
        coverage = len(system) / (len(system) + oov)
        pearson = scipy.stats.pearsonr(gold, system)[0]
        spearman = scipy.stats.spearmanr(gold, system)[0]
        results.append((name, coverage, pearson, spearman))
        print('Coverage:{0:7.2%}  Pearson:{1:7.2%}  Spearman:{2:7.2%} | {3}'.format(coverage, pearson, spearman, name))

    # Compute and print total (averaged) results
    if len(results) > 1:
        print('-'*80)
        if args.sim is not None:
            sim = list(zip(*[res for res in results if res[0] in args.sim]))
            print('Coverage:{0:7.2%}  Pearson:{1:7.2%}  Spearman:{2:7.2%} | sim.'.format(np.mean(sim[1]), np.mean(sim[2]), np.mean(sim[3])))
        if args.rel is not None:
            rel = list(zip(*[res for res in results if res[0] in args.rel]))
            print('Coverage:{0:7.2%}  Pearson:{1:7.2%}  Spearman:{2:7.2%} | rel.'.format(np.mean(rel[1]), np.mean(rel[2]), np.mean(rel[3])))
        if args.all is not None:
            results = [res for res in results if res[0] in args.all]
        results = list(zip(*results))
        print('Coverage:{0:7.2%}  Pearson:{1:7.2%}  Spearman:{2:7.2%} | all'.format(np.mean(results[1]), np.mean(results[2]), np.mean(results[3])))


if __name__ == '__main__':
    main()
