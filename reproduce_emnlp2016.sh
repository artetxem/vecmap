#!/bin/bash
#
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

ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DATA="$ROOT/data"
OUTPUT="$ROOT/output/emnlp2016"
TRAIN_DICTIONARY="$DATA/dictionaries/en-it.train.txt"
TEST_DICTIONARY="$DATA/dictionaries/en-it.test.txt"
TEST_ANALOGIES="$DATA/analogies/questions-words.txt"

mkdir -p "$OUTPUT/unconstrained"
mkdir -p "$OUTPUT/unconstrained-unit"
mkdir -p "$OUTPUT/unconstrained-unit-center"
mkdir -p "$OUTPUT/orthogonal"
mkdir -p "$OUTPUT/orthogonal-unit"
mkdir -p "$OUTPUT/orthogonal-unit-center"

echo 'ORIGINAL EMBEDDINGS'
echo -n '  - EN AN  |  '
python3 "$ROOT/eval_analogy.py" -l "$DATA/embeddings/original/en.emb.txt" -i "$TEST_ANALOGIES" -t 30000

echo '--------------------------------------------------------------------------------'
echo 'UNCONSTRAINED MAPPING (Mikolov et al., 2013)'
python3 "$ROOT/map_embeddings.py" --unconstrained -d "$TRAIN_DICTIONARY" "$DATA/embeddings/original/en.emb.txt" "$DATA/embeddings/original/it.emb.txt" "$OUTPUT/unconstrained/en.emb.txt" "$OUTPUT/unconstrained/it.emb.txt"
echo -n '  - EN-IT  |  '
python3 "$ROOT/eval_translation.py" "$OUTPUT/unconstrained/en.emb.txt" "$OUTPUT/unconstrained/it.emb.txt" -d "$TEST_DICTIONARY"
echo -n '  - EN AN  |  '
python3 "$ROOT/eval_analogy.py" -l "$OUTPUT/unconstrained/en.emb.txt" -i "$TEST_ANALOGIES" -t 30000

echo 'UNCONSTRAINED MAPPING + LENGTH NORMALIZATION'
python3 "$ROOT/map_embeddings.py" --unconstrained -d "$TRAIN_DICTIONARY" "$DATA/embeddings/unit/en.emb.txt" "$DATA/embeddings/unit/it.emb.txt" "$OUTPUT/unconstrained-unit/en.emb.txt" "$OUTPUT/unconstrained-unit/it.emb.txt"
echo -n '  - EN-IT  |  '
python3 "$ROOT/eval_translation.py" "$OUTPUT/unconstrained-unit/en.emb.txt" "$OUTPUT/unconstrained-unit/it.emb.txt" -d "$TEST_DICTIONARY"
echo -n '  - EN AN  |  '
python3 "$ROOT/eval_analogy.py" -l "$OUTPUT/unconstrained-unit/en.emb.txt" -i "$TEST_ANALOGIES" -t 30000

echo 'UNCONSTRAINED MAPPING + LENGTH NORMALIZATION + MEAN CENTERING'
python3 "$ROOT/map_embeddings.py" --unconstrained -d "$TRAIN_DICTIONARY" "$DATA/embeddings/unit-center/en.emb.txt" "$DATA/embeddings/unit-center/it.emb.txt" "$OUTPUT/unconstrained-unit-center/en.emb.txt" "$OUTPUT/unconstrained-unit-center/it.emb.txt"
echo -n '  - EN-IT  |  '
python3 "$ROOT/eval_translation.py" "$OUTPUT/unconstrained-unit-center/en.emb.txt" "$OUTPUT/unconstrained-unit-center/it.emb.txt" -d "$TEST_DICTIONARY"
echo -n '  - EN AN  |  '
python3 "$ROOT/eval_analogy.py" -l "$OUTPUT/unconstrained-unit-center/en.emb.txt" -i "$TEST_ANALOGIES" -t 30000

echo '--------------------------------------------------------------------------------'
echo 'ORTHOGONAL MAPPING (Zhang et al., 2016)'
python3 "$ROOT/map_embeddings.py" --orthogonal -d "$TRAIN_DICTIONARY" "$DATA/embeddings/original/en.emb.txt" "$DATA/embeddings/original/it.emb.txt" "$OUTPUT/orthogonal/en.emb.txt" "$OUTPUT/orthogonal/it.emb.txt"
echo -n '  - EN-IT  |  '
python3 "$ROOT/eval_translation.py" "$OUTPUT/orthogonal/en.emb.txt" "$OUTPUT/orthogonal/it.emb.txt" -d "$TEST_DICTIONARY"
echo -n '  - EN AN  |  '
python3 "$ROOT/eval_analogy.py" -l "$OUTPUT/orthogonal/en.emb.txt" -i "$TEST_ANALOGIES" -t 30000

echo 'ORTHOGONAL MAPPING + LENGTH NORMALIZATION (Xing et al., 2015)'
python3 "$ROOT/map_embeddings.py" --orthogonal -d "$TRAIN_DICTIONARY" "$DATA/embeddings/unit/en.emb.txt" "$DATA/embeddings/unit/it.emb.txt" "$OUTPUT/orthogonal-unit/en.emb.txt" "$OUTPUT/orthogonal-unit/it.emb.txt"
echo -n '  - EN-IT  |  '
python3 "$ROOT/eval_translation.py" "$OUTPUT/orthogonal-unit/en.emb.txt" "$OUTPUT/orthogonal-unit/it.emb.txt" -d "$TEST_DICTIONARY"
echo -n '  - EN AN  |  '
python3 "$ROOT/eval_analogy.py" -l "$OUTPUT/orthogonal-unit/en.emb.txt" -i "$TEST_ANALOGIES" -t 30000

echo 'ORTHOGONAL MAPPING + LENGTH NORMALIZATION + MEAN CENTERING (best)'
python3 "$ROOT/map_embeddings.py" --orthogonal -d "$TRAIN_DICTIONARY" "$DATA/embeddings/unit-center/en.emb.txt" "$DATA/embeddings/unit-center/it.emb.txt" "$OUTPUT/orthogonal-unit-center/en.emb.txt" "$OUTPUT/orthogonal-unit-center/it.emb.txt"
echo -n '  - EN-IT  |  '
python3 "$ROOT/eval_translation.py" "$OUTPUT/orthogonal-unit-center/en.emb.txt" "$OUTPUT/orthogonal-unit-center/it.emb.txt" -d "$TEST_DICTIONARY"
echo -n '  - EN AN  |  '
python3 "$ROOT/eval_analogy.py" -l "$OUTPUT/orthogonal-unit-center/en.emb.txt" -i "$TEST_ANALOGIES" -t 30000
