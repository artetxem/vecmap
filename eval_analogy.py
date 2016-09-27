# Copyright (C) 2016  Mikel Artetxe <artetxem@gmail.com>
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


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate embeddings in word analogy')
    parser.add_argument('embeddings', help='the word embeddings')
    parser.add_argument('-t', '--threshold', type=int, default=0, help='reduce vocabulary of the model for fast approximate evaluation (0 = off, otherwise typical value is 30,000)')
    parser.add_argument('-i', '--input', default=sys.stdin.fileno(), help='the test file (defaults to stdin)')
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose output (give category specific results)')
    parser.add_argument('-l', '--lowercase', action='store_true', help='lowercase the words in the test file')
    parser.add_argument('--encoding', default='utf-8', action='store_true', help='the character encoding for input/output (defaults to utf-8)')
    args = parser.parse_args()

    # Read input embeddings
    f = open(args.embeddings, encoding=args.encoding, errors='surrogateescape')
    words, matrix = embeddings.read(f, threshold=args.threshold)

    # Length normalize embeddings
    matrix = embeddings.length_normalize(matrix)

    # Build word to index map
    word2ind = {word: i for i, word in enumerate(words)}

    # Compute accuracy and coverage and print results
    category = category_name = None
    semantic = {'correct': 0, 'total': 0, 'oov': 0}
    syntactic = {'correct': 0, 'total': 0, 'oov': 0}
    f = open(args.input, encoding=args.encoding, errors='surrogateescape')
    for line in f:
        if line.startswith(': '):
            if args.verbose and category is not None:
                print('Coverage:{0:7.2%}  Accuracy:{1:7.2%} | {2}'.format(
                    category['total'] / (category['total'] + category['oov']),
                    category['correct'] / category['total'],
                    category_name))
            category_name = line[2:-1]
            current = syntactic if category_name.startswith('gram') else semantic
            category = {'correct': 0, 'total': 0, 'oov': 0}
        else:
            try:
                src1, trg1, src2, trg2 = [word2ind[word.lower() if args.lowercase else word] for word in line.split()]
                similarities = np.dot(matrix, matrix[src2] - matrix[src1] + matrix[trg1])
                similarities[[src1, trg1, src2]] = -1
                closest = np.argmax(similarities)
                if closest == trg2:
                    category['correct'] += 1
                    current['correct'] += 1
                category['total'] += 1
                current['total'] += 1
            except KeyError:
                category['oov'] += 1
                current['oov'] += 1
    if args.verbose:
        print('Coverage:{0:7.2%}  Accuracy:{1:7.2%} | {2}'.format(
            category['total'] / (category['total'] + category['oov']),
            category['correct'] / category['total'],
            category_name))
        print('-'*80)
    print('Coverage:{0:7.2%}  Accuracy:{1:7.2%} (sem:{2:7.2%}, syn:{3:7.2%})'.format(
        (semantic['total'] + syntactic['total']) / (semantic['total'] + syntactic['total'] + semantic['oov'] + syntactic['oov']),
        (semantic['correct'] + syntactic['correct']) / (semantic['total'] + syntactic['total']),
        semantic['correct'] / semantic['total'],
        syntactic['correct'] / syntactic['total']))


if __name__ == '__main__':
    main()