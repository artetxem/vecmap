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
import numpy as np
import sys


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Project the source embeddings into the target embedding space maximizing the squared Euclidean distances for the given dictionary')
    parser.add_argument('src_embeddings', help='the source embeddings')
    parser.add_argument('trg_embeddings', help='the target embeddings')
    parser.add_argument('-c', '--orthogonal', dest='orthogonal', action='store_true', help='use orthogonal constrained mapping (default)')
    parser.add_argument('-u', '--unconstrained', dest='orthogonal', action='store_false', help='use unconstrained mapping')
    parser.add_argument('-d', '--dictionary', default=sys.stdin.fileno(), help='the training dictionary file (defaults to stdin)')
    parser.add_argument('-o', '--output', default=sys.stdout.fileno(), help='the output projected embedding file (defaults to stdout)')
    parser.add_argument('--encoding', default='utf-8', action='store_true', help='the character encoding for input/output (defaults to utf-8)')
    parser.set_defaults(orthogonal=True)
    args = parser.parse_args()

    # Read input embeddings
    srcfile = open(args.src_embeddings, encoding=args.encoding, errors='surrogateescape')
    trgfile = open(args.trg_embeddings, encoding=args.encoding, errors='surrogateescape')
    src_words, src_matrix = embeddings.read(srcfile)
    trg_words, trg_matrix = embeddings.read(trgfile)

    # Build word to index map
    src_word2ind = {word: i for i, word in enumerate(src_words)}
    trg_word2ind = {word: i for i, word in enumerate(trg_words)}

    # Read dictionary
    f = open(args.dictionary, encoding=args.encoding, errors='surrogateescape')
    src_indices = []
    trg_indices = []
    for line in f:
        src, trg = line.split()
        try:
            src_ind = src_word2ind[src]
            trg_ind = trg_word2ind[trg]
            src_indices.append(src_ind)
            trg_indices.append(trg_ind)
        except KeyError:
            print('WARNING: OOV dictionary entry ({0} - {1})'.format(src, trg), file=sys.stderr)

    # Learn the linear transformation minimizing the squared Euclidean distances (see paper)
    x = src_matrix[src_indices]
    z = trg_matrix[trg_indices]
    if args.orthogonal:  # orthogonal mapping
        u, s, vt = np.linalg.svd(np.dot(z.T, x))
        w = np.dot(vt.T, u.T)
    else:  # unconstrained mapping
        x_pseudoinv = np.dot(np.linalg.inv(np.dot(x.T, x)), x.T)
        w = np.dot(x_pseudoinv, z)

    # Project and write source embeddings
    f = open(args.output, mode='w', encoding=args.encoding, errors='surrogateescape')
    embeddings.write(src_words, np.dot(src_matrix, w), f)


if __name__ == '__main__':
    main()
