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
import sys


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Normalize word embeddings')
    parser.add_argument('actions', choices=['unit', 'center', 'unitdim', 'centeremb'], nargs='*', default=[], help='the actions to perform in order')
    parser.add_argument('-i', '--input', default=sys.stdin.fileno(), help='the input word embedding file (defaults to stdin)')
    parser.add_argument('-o', '--output', default=sys.stdout.fileno(), help='the output word embedding file (defaults to stdout)')
    parser.add_argument('--encoding', default='utf-8', help='the character encoding for input/output (defaults to utf-8)')
    args = parser.parse_args()

    # Read input embeddings
    f = open(args.input, encoding=args.encoding, errors='surrogateescape')
    words, matrix = embeddings.read(f)

    # Perform normalization actions
    for action in args.actions:
        if action == 'unit':
            matrix = embeddings.length_normalize(matrix)
        elif action == 'center':
            matrix = embeddings.mean_center(matrix)
        elif action == 'unitdim':
            matrix = embeddings.length_normalize_dimensionwise(matrix)
        elif action == 'centeremb':
            matrix = embeddings.mean_center_embeddingwise(matrix)

    # Write normalized embeddings
    f = open(args.output, mode='w', encoding=args.encoding, errors='surrogateescape')
    embeddings.write(words, matrix, f)


if __name__ == '__main__':
    main()