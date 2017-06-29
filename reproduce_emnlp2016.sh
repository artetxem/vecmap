#!/bin/bash
#
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

DATA='data'
OUTPUT='output'
SRC_EMBEDDINGS="$DATA/EN.200K.cbow1_wind5_hs0_neg10_size300_smpl1e-05.txt"
TRG_EMBEDDINGS="$DATA/IT.200K.cbow1_wind5_hs0_neg10_size300_smpl1e-05.txt"
TRAIN_DICTIONARY="$DATA/OPUS_en_it_europarl_train_5K.txt"
TEST_DICTIONARY="$DATA/OPUS_en_it_europarl_test.txt"
TEST_ANALOGIES="$DATA/questions-words.txt"

# Download data
mkdir -p $DATA
if [ ! -f $SRC_EMBEDDINGS ] || [ ! -f $TRG_EMBEDDINGS ] || [ ! -f $TRAIN_DICTIONARY ] || [ ! -f $TEST_DICTIONARY ]; then
    wget -q http://clic.cimec.unitn.it/~georgiana.dinu/down/resources/transmat.zip -O $DATA/transmat.zip
    unzip -p $DATA/transmat.zip data/EN.200K.cbow1_wind5_hs0_neg10_size300_smpl1e-05.txt > $SRC_EMBEDDINGS
    unzip -p $DATA/transmat.zip data/IT.200K.cbow1_wind5_hs0_neg10_size300_smpl1e-05.txt > $TRG_EMBEDDINGS
    unzip -p $DATA/transmat.zip $DATA/OPUS_en_it_europarl_train_5K.txt > $TRAIN_DICTIONARY
    unzip -p $DATA/transmat.zip $DATA/OPUS_en_it_europarl_test.txt > $TEST_DICTIONARY
fi
if [ ! -f $TEST_ANALOGIES ]; then
    wget -q https://storage.googleapis.com/google-code-archive-source/v2/code.google.com/word2vec/source-archive.zip -O $DATA/word2vec.zip
    unzip -p $DATA/word2vec.zip word2vec/trunk/questions-words.txt > $TEST_ANALOGIES
fi

# Normalize embeddings for all the different configurations
mkdir -p $OUTPUT/original
cp $SRC_EMBEDDINGS $OUTPUT/original/src.emb.txt
cp $TRG_EMBEDDINGS $OUTPUT/original/trg.emb.txt
mkdir -p $OUTPUT/unit
python3 normalize_embeddings.py unit -i $SRC_EMBEDDINGS -o $OUTPUT/unit/src.emb.txt
python3 normalize_embeddings.py unit -i $TRG_EMBEDDINGS -o $OUTPUT/unit/trg.emb.txt
mkdir -p $OUTPUT/unit-center
python3 normalize_embeddings.py unit center -i $SRC_EMBEDDINGS -o $OUTPUT/unit-center/src.emb.txt
python3 normalize_embeddings.py unit center -i $TRG_EMBEDDINGS -o $OUTPUT/unit-center/trg.emb.txt

echo 'ORIGINAL EMBEDDINGS'
echo -n '  - EN AN  |  '
python3 eval_analogy.py -l $OUTPUT/original/src.emb.txt -i $TEST_ANALOGIES -t 30000

echo '--------------------------------------------------------------------------------'
echo 'UNCONSTRAINED MAPPING (Mikolov et al., 2013)'
python3 map_embeddings.py --unconstrained $OUTPUT/original/src.emb.txt $OUTPUT/original/trg.emb.txt -d $TRAIN_DICTIONARY -o $OUTPUT/original/src.mapped-unconstrained.emb.txt
echo -n '  - EN-IT  |  '
python3 eval_translation.py $OUTPUT/original/src.mapped-unconstrained.emb.txt $OUTPUT/original/trg.emb.txt -d $TEST_DICTIONARY
echo -n '  - EN AN  |  '
python3 eval_analogy.py -l $OUTPUT/original/src.mapped-unconstrained.emb.txt -i $TEST_ANALOGIES -t 30000

echo 'UNCONSTRAINED MAPPING + LENGTH NORMALIZATION'
python3 map_embeddings.py --unconstrained $OUTPUT/unit/src.emb.txt $OUTPUT/unit/trg.emb.txt -d $TRAIN_DICTIONARY -o $OUTPUT/unit/src.mapped-unconstrained.emb.txt
echo -n '  - EN-IT  |  '
python3 eval_translation.py $OUTPUT/unit/src.mapped-unconstrained.emb.txt $OUTPUT/unit/trg.emb.txt -d $TEST_DICTIONARY
echo -n '  - EN AN  |  '
python3 eval_analogy.py -l $OUTPUT/unit/src.mapped-unconstrained.emb.txt -i $TEST_ANALOGIES -t 30000

echo 'UNCONSTRAINED MAPPING + LENGTH NORMALIZATION + MEAN CENTERING'
python3 map_embeddings.py --unconstrained $OUTPUT/unit-center/src.emb.txt $OUTPUT/unit-center/trg.emb.txt -d $TRAIN_DICTIONARY -o $OUTPUT/unit-center/src.mapped-unconstrained.emb.txt
echo -n '  - EN-IT  |  '
python3 eval_translation.py $OUTPUT/unit-center/src.mapped-unconstrained.emb.txt $OUTPUT/unit-center/trg.emb.txt -d $TEST_DICTIONARY
echo -n '  - EN AN  |  '
python3 eval_analogy.py -l $OUTPUT/unit-center/src.mapped-unconstrained.emb.txt -i $TEST_ANALOGIES -t 30000

echo '--------------------------------------------------------------------------------'
echo 'ORTHOGONAL MAPPING'
python3 map_embeddings.py --orthogonal $OUTPUT/original/src.emb.txt $OUTPUT/original/trg.emb.txt -d $TRAIN_DICTIONARY -o $OUTPUT/original/src.mapped-orthogonal.emb.txt
echo -n '  - EN-IT  |  '
python3 eval_translation.py $OUTPUT/original/src.mapped-orthogonal.emb.txt $OUTPUT/original/trg.emb.txt -d $TEST_DICTIONARY
echo -n '  - EN AN  |  '
python3 eval_analogy.py -l $OUTPUT/original/src.mapped-orthogonal.emb.txt -i $TEST_ANALOGIES -t 30000

echo 'ORTHOGONAL MAPPING + LENGTH NORMALIZATION (Xing et al., 2015)'
python3 map_embeddings.py --orthogonal $OUTPUT/unit/src.emb.txt $OUTPUT/unit/trg.emb.txt -d $TRAIN_DICTIONARY -o $OUTPUT/unit/src.mapped-orthogonal.emb.txt
echo -n '  - EN-IT  |  '
python3 eval_translation.py $OUTPUT/unit/src.mapped-orthogonal.emb.txt $OUTPUT/unit/trg.emb.txt -d $TEST_DICTIONARY
echo -n '  - EN AN  |  '
python3 eval_analogy.py -l $OUTPUT/unit/src.mapped-orthogonal.emb.txt -i $TEST_ANALOGIES -t 30000

echo 'ORTHOGONAL MAPPING + LENGTH NORMALIZATION + MEAN CENTERING (best)'
python3 map_embeddings.py --orthogonal $OUTPUT/unit-center/src.emb.txt $OUTPUT/unit-center/trg.emb.txt -d $TRAIN_DICTIONARY -o $OUTPUT/unit-center/src.mapped-orthogonal.emb.txt
echo -n '  - EN-IT  |  '
python3 eval_translation.py $OUTPUT/unit-center/src.mapped-orthogonal.emb.txt $OUTPUT/unit-center/trg.emb.txt -d $TEST_DICTIONARY
echo -n '  - EN AN  |  '
python3 eval_analogy.py -l $OUTPUT/unit-center/src.mapped-orthogonal.emb.txt -i $TEST_ANALOGIES -t 30000
