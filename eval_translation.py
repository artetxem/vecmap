# Copyright (C) 2016-2017  Mikel Artetxe <artetxem@gmail.com>
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
import collections
import numpy as np
import sys


BATCH_SIZE = 1000


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate embeddings of two languages in a shared space in word translation induction')
    parser.add_argument('src_embeddings', help='the source language embeddings')
    parser.add_argument('trg_embeddings', help='the target language embeddings')
    parser.add_argument('-d', '--dictionary', default=sys.stdin.fileno(), help='the test dictionary file (defaults to stdin)')
    parser.add_argument('--dot', action='store_true', help='use the dot product in the similarity computations instead of the cosine')
    parser.add_argument('--encoding', default='utf-8', help='the character encoding for input/output (defaults to utf-8)')
    args = parser.parse_args()

    # Read input embeddings
    srcfile = open(args.src_embeddings, encoding=args.encoding, errors='surrogateescape')
    trgfile = open(args.trg_embeddings, encoding=args.encoding, errors='surrogateescape')
    src_words, src_matrix = embeddings.read(srcfile)
    trg_words, trg_matrix = embeddings.read(trgfile)

    # Length normalize embeddings so their dot product effectively computes the cosine similarity
    if not args.dot:
        src_matrix = embeddings.length_normalize(src_matrix)
        trg_matrix = embeddings.length_normalize(trg_matrix)

    # Build word to index map
    src_word2ind = {word: i for i, word in enumerate(src_words)}
    trg_word2ind = {word: i for i, word in enumerate(trg_words)}

    # Read dictionary and compute coverage
    f = open(args.dictionary, encoding=args.encoding, errors='surrogateescape')
    src2trg = collections.defaultdict(set)
    oov = set()
    vocab = set()
    for line in f:
        src, trg = line.split()
        try:
            src_ind = src_word2ind[src]
            trg_ind = trg_word2ind[trg]
            src2trg[src_ind].add(trg_ind)
            vocab.add(src)
        except KeyError:
            oov.add(src)
    oov -= vocab  # If one of the translation options is in the vocabulary, then the entry is not an oov
    coverage = len(src2trg) / (len(src2trg) + len(oov))

    # Compute accuracy
    correct = 0
    src, trg = zip(*src2trg.items())
    for i in range(0, len(src2trg), BATCH_SIZE):
        j = min(i + BATCH_SIZE, len(src2trg))
        similarities = src_matrix[list(src[i:j])].dot(trg_matrix.T)
        nn = np.argmax(similarities, axis=1).tolist()
        for k in range(j-i):
            if nn[k] in trg[i+k]:
                correct += 1
    print('Coverage:{0:7.2%}  Accuracy:{1:7.2%}'.format(coverage, correct / len(src2trg)))


if __name__ == '__main__':
    main()
