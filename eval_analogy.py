# Copyright (C) 2016-2018  Mikel Artetxe <artetxem@gmail.com>
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
import sys


BATCH_SIZE = 1000


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate embeddings in word analogy')
    parser.add_argument('embeddings', help='the word embeddings')
    parser.add_argument('-t', '--threshold', type=int, default=0, help='reduce vocabulary of the model for fast approximate evaluation (0 = off, otherwise typical value is 30,000)')
    parser.add_argument('-i', '--input', default=sys.stdin.fileno(), help='the test file (defaults to stdin)')
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose output (give category specific results)')
    parser.add_argument('-l', '--lowercase', action='store_true', help='lowercase the words in the test file')
    parser.add_argument('--encoding', default='utf-8', help='the character encoding for input/output (defaults to utf-8)')
    parser.add_argument('--precision', choices=['fp16', 'fp32', 'fp64'], default='fp32', help='the floating-point precision (defaults to fp32)')
    args = parser.parse_args()

    # Choose the right dtype for the desired precision
    if args.precision == 'fp16':
        dtype = 'float16'
    elif args.precision == 'fp32':
        dtype = 'float32'
    elif args.precision == 'fp64':
        dtype = 'float64'

    # Read input embeddings
    f = open(args.embeddings, encoding=args.encoding, errors='surrogateescape')
    words, matrix = embeddings.read(f, threshold=args.threshold, dtype=dtype)

    # Build word to index map
    word2ind = {word: i for i, word in enumerate(words)}

    # Length normalize embeddings
    embeddings.length_normalize(matrix)

    # Parse test file
    f = open(args.input, encoding=args.encoding, errors='surrogateescape')
    categories = []
    src1 = []
    trg1 = []
    src2 = []
    trg2 = []
    for line in f:
        if line.startswith(': '):
            name = line[2:-1]
            is_syntactic = name.startswith('gram')
            categories.append({'name': name, 'is_syntactic': is_syntactic, 'total': 0, 'oov': 0})
        else:
            try:
                ind = [word2ind[word.lower() if args.lowercase else word] for word in line.split()]
                src1.append(ind[0])
                trg1.append(ind[1])
                src2.append(ind[2])
                trg2.append(ind[3])
                categories[-1]['total'] += 1
            except KeyError:
                categories[-1]['oov'] += 1
    total = len(src1)

    # Compute nearest neighbors using efficient matrix multiplication
    nn = []
    for i in range(0, total, BATCH_SIZE):
        j = min(i + BATCH_SIZE, total)
        similarities = (matrix[src2[i:j]] - matrix[src1[i:j]] + matrix[trg1[i:j]]).dot(matrix.T)
        similarities[range(j-i), src1[i:j]] = -1
        similarities[range(j-i), trg1[i:j]] = -1
        similarities[range(j-i), src2[i:j]] = -1
        nn += np.argmax(similarities, axis=1).tolist()
    nn = np.array(nn)

    # Compute and print accuracies
    semantic = {'correct': 0, 'total': 0, 'oov': 0}
    syntactic = {'correct': 0, 'total': 0, 'oov': 0}
    ind = 0
    for category in categories:
        current = syntactic if category['is_syntactic'] else semantic
        correct = np.sum(nn[ind:ind+category['total']] == trg2[ind:ind+category['total']])
        current['correct'] += correct
        current['total'] += category['total']
        current['oov'] += category['oov']
        ind += category['total']
        if args.verbose:
            print('Coverage:{0:7.2%}  Accuracy:{1:7.2%} | {2}'.format(
                category['total'] / (category['total'] + category['oov']),
                correct / category['total'],
                category['name']))
    if args.verbose:
        print('-'*80)
    print('Coverage:{0:7.2%}  Accuracy:{1:7.2%} (sem:{2:7.2%}, syn:{3:7.2%})'.format(
        (semantic['total'] + syntactic['total']) / (semantic['total'] + syntactic['total'] + semantic['oov'] + syntactic['oov']),
        (semantic['correct'] + syntactic['correct']) / (semantic['total'] + syntactic['total']),
        semantic['correct'] / semantic['total'],
        syntactic['correct'] / syntactic['total']))


if __name__ == '__main__':
    main()
