#!/bin/bash
#
# Copyright (C) 2018  Mikel Artetxe <artetxem@gmail.com>
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
EMBEDDINGS="$ROOT/data/embeddings/original"
DICTIONARIES="$ROOT/data/dictionaries"
TMP=$(mktemp -dp "$ROOT")

map () {
    lang=$1
    python3 "$ROOT/map_embeddings.py" "$EMBEDDINGS/en.emb.txt" "$EMBEDDINGS/$lang.emb.txt" "$TMP/src.emb.txt" "$TMP/trg.emb.txt" -d "$DICTIONARIES/en-$lang.train.txt" "${@:2}" $AAAI2018_SETTINGS
}

evaluate () {
    lang=$1
    python3 "$ROOT/eval_translation.py" "$TMP/src.emb.txt" "$TMP/trg.emb.txt" -d "$DICTIONARIES/en-$lang.test.txt" "${@:2}" $AAAI2018_SETTINGS | tail -c 7 | tr -d '\n'
}

run_all () {
    languages=( it de fi es )
    for lang in "${languages[@]}"
    do
        python3 "$ROOT/map_embeddings.py" "$EMBEDDINGS/en.emb.txt" "$EMBEDDINGS/$lang.emb.txt" "$TMP/src.emb.txt" "$TMP/trg.emb.txt" -d "$DICTIONARIES/en-$lang.train.txt" "$@" $AAAI2018_SETTINGS
        python3 "$ROOT/eval_translation.py" "$TMP/src.emb.txt" "$TMP/trg.emb.txt" -d "$DICTIONARIES/en-$lang.test.txt" $AAAI2018_SETTINGS | tail -c 7 | tr -d '\n'
        if [ "$lang" != 'es' ]; then echo -n '   '; fi
    done
    echo
}

echo    '--------------------------------------------------------------------------'
echo    '                                 TABLE 2                                  '
echo    '--------------------------------------------------------------------------'
echo    ' Motivation   S1   S4 (src)   S4 (trg)   EN-IT    EN-DE    EN-FI    EN-ES '
echo    '--------------------------------------------------------------------------'
echo -n '   Orth.                                 '
run_all --normalize unit center
echo    '--------------------------------------------------------------------------'
echo -n '    CCA       x                          '
run_all --normalize unit center --whiten
echo    '--------------------------------------------------------------------------'
echo -n '    OLS       x      src        src      '
run_all --normalize unit center --whiten --src_dewhiten src --trg_dewhiten src
echo -n '    OLS       x      trg        trg      '
run_all --normalize unit center --whiten --src_dewhiten trg --trg_dewhiten trg
echo    '--------------------------------------------------------------------------'
echo -n '    New       x      src        trg      '
run_all --normalize unit center --whiten --src_dewhiten src --trg_dewhiten trg
echo    '--------------------------------------------------------------------------'
echo
echo    '------------------------------------------------------'
echo    '                       TABLE 3                        '
echo    '------------------------------------------------------'
echo    '  Motivation   S3    EN-IT    EN-DE    EN-FI    EN-ES '
echo    '------------------------------------------------------'
echo -n ' Orth. / CCA         '
run_all --normalize unit center --whiten --src_dewhiten src --trg_dewhiten trg
echo    '------------------------------------------------------'
echo -n '    OLS        src   '
run_all --normalize unit center --whiten --src_dewhiten src --trg_dewhiten trg --src_reweight
echo -n '    OLS        trg   '
run_all --normalize unit center --whiten --src_dewhiten src --trg_dewhiten trg --trg_reweight
echo    '------------------------------------------------------'
echo
echo    '---------------------------------------------'
echo    '                   TABLE 4                   '
echo    '---------------------------------------------'
echo    ' S3    S5   EN-IT    EN-DE    EN-FI    EN-ES '
echo    '---------------------------------------------'
echo -n '            '
run_all --normalize unit center --whiten --src_dewhiten src --trg_dewhiten trg
echo -n '       x    '
map it --normalize unit center --whiten --src_dewhiten src --trg_dewhiten trg --dim_reduction 177; evaluate it;  echo -n '   '
map de --normalize unit center --whiten --src_dewhiten src --trg_dewhiten trg --dim_reduction 224; evaluate de;  echo -n '   '
map fi --normalize unit center --whiten --src_dewhiten src --trg_dewhiten trg --dim_reduction 233; evaluate fi;  echo -n '   '
map es --normalize unit center --whiten --src_dewhiten src --trg_dewhiten trg --dim_reduction 246; evaluate es;  echo
echo    '---------------------------------------------'
echo -n ' trg        '
run_all --normalize unit center --whiten --src_dewhiten src --trg_dewhiten trg --trg_reweight
echo -n ' trg   x    '
map it --normalize unit center --whiten --src_dewhiten src --trg_dewhiten trg --trg_reweight --dim_reduction 242; evaluate it;  echo -n '   '
map de --normalize unit center --whiten --src_dewhiten src --trg_dewhiten trg --trg_reweight --dim_reduction 272; evaluate de;  echo -n '   '
map fi --normalize unit center --whiten --src_dewhiten src --trg_dewhiten trg --trg_reweight --dim_reduction 268; evaluate fi;  echo -n '   '
map es --normalize unit center --whiten --src_dewhiten src --trg_dewhiten trg --trg_reweight --dim_reduction 216; evaluate es;  echo
echo    '---------------------------------------------'
echo
echo    '--------------------------------------------------------------'
echo    '                           TABLE 5                            '
echo    '--------------------------------------------------------------'
echo    '     Retrieval method        EN-IT    EN-DE    EN-FI    EN-ES '
echo    '--------------------------------------------------------------'
echo -n '     Nearest neighbor        '
map it --normalize unit center --whiten --src_dewhiten src --trg_dewhiten trg --trg_reweight --dim_reduction 242; evaluate it;  echo -n '   '
map de --normalize unit center --whiten --src_dewhiten src --trg_dewhiten trg --trg_reweight --dim_reduction 272; evaluate de;  echo -n '   '
map fi --normalize unit center --whiten --src_dewhiten src --trg_dewhiten trg --trg_reweight --dim_reduction 268; evaluate fi;  echo -n '   '
map es --normalize unit center --whiten --src_dewhiten src --trg_dewhiten trg --trg_reweight --dim_reduction 216; evaluate es;  echo
echo -n ' Inverted nearest neighbor   '
map it --normalize unit center --whiten --src_dewhiten src --trg_dewhiten trg --trg_reweight --dim_reduction 242; evaluate it --retrieval invnn;  echo -n '   '
map de --normalize unit center --whiten --src_dewhiten src --trg_dewhiten trg --trg_reweight --dim_reduction 272; evaluate de --retrieval invnn;  echo -n '   '
map fi --normalize unit center --whiten --src_dewhiten src --trg_dewhiten trg --trg_reweight --dim_reduction 268; evaluate fi --retrieval invnn;  echo -n '   '
map es --normalize unit center --whiten --src_dewhiten src --trg_dewhiten trg --trg_reweight --dim_reduction 216; evaluate es --retrieval invnn;  echo
echo -n '     Inverted softmax        '
map it --normalize unit center --whiten --src_dewhiten src --trg_dewhiten trg --trg_reweight --dim_reduction 242; evaluate it --retrieval invsoftmax --inv_temperature 12;  echo -n '   '
map de --normalize unit center --whiten --src_dewhiten src --trg_dewhiten trg --trg_reweight --dim_reduction 272; evaluate de --retrieval invsoftmax --inv_temperature  1;  echo -n '   '
map fi --normalize unit center --whiten --src_dewhiten src --trg_dewhiten trg --trg_reweight --dim_reduction 268; evaluate fi --retrieval invsoftmax --inv_temperature  1;  echo -n '   '
map es --normalize unit center --whiten --src_dewhiten src --trg_dewhiten trg --trg_reweight --dim_reduction 216; evaluate es --retrieval invsoftmax --inv_temperature  1;  echo
echo    '--------------------------------------------------------------'
echo
echo    '-----------------------------------------------------------------------'
echo    '                                TABLE 6                                '
echo    '-----------------------------------------------------------------------'
echo    '                                      EN-IT    EN-DE    EN-FI    EN-ES '
echo    '-----------------------------------------------------------------------'
echo -n ' Mikolov, Le, and Sutskever (2013)    '
run_all --whiten --src_reweight --src_dewhiten trg --trg_dewhiten trg
echo -n ' Shigeto et al. (2015)                '
run_all --whiten --trg_reweight --src_dewhiten src --trg_dewhiten src
echo -n ' Xing et al. (2015)                   '
run_all --normalize unit
echo -n ' Zhang et al. (2016)                  '
run_all
echo -n ' Artetxe, Labaka, and Agirre (2016)   '
run_all --normalize unit center
echo -n ' Smith et al. (2017)                  '
map it --normalize unit --dim_reduction 206; evaluate it --retrieval invsoftmax --inv_temperature 17;  echo -n '   '
map de --normalize unit --dim_reduction 240; evaluate de --retrieval invsoftmax --inv_temperature  1;  echo -n '   '
map fi --normalize unit --dim_reduction 250; evaluate fi --retrieval invsoftmax --inv_temperature  4;  echo -n '   '
map es --normalize unit --dim_reduction 251; evaluate es --retrieval invsoftmax --inv_temperature 14;  echo
echo    '-----------------------------------------------------------------------'
echo -n ' Proposed (nearest neighbor)          '
map it --normalize unit center --whiten --src_dewhiten src --trg_dewhiten trg --trg_reweight --dim_reduction 242; evaluate it;  echo -n '   '
map de --normalize unit center --whiten --src_dewhiten src --trg_dewhiten trg --trg_reweight --dim_reduction 272; evaluate de;  echo -n '   '
map fi --normalize unit center --whiten --src_dewhiten src --trg_dewhiten trg --trg_reweight --dim_reduction 268; evaluate fi;  echo -n '   '
map es --normalize unit center --whiten --src_dewhiten src --trg_dewhiten trg --trg_reweight --dim_reduction 216; evaluate es;  echo
echo -n ' Proposed (inverted softmax)          '
map it --normalize unit center --whiten --src_dewhiten src --trg_dewhiten trg --trg_reweight --dim_reduction 242; evaluate it --retrieval invsoftmax --inv_temperature 12;  echo -n '   '
map de --normalize unit center --whiten --src_dewhiten src --trg_dewhiten trg --trg_reweight --dim_reduction 272; evaluate de --retrieval invsoftmax --inv_temperature  1;  echo -n '   '
map fi --normalize unit center --whiten --src_dewhiten src --trg_dewhiten trg --trg_reweight --dim_reduction 268; evaluate fi --retrieval invsoftmax --inv_temperature  1;  echo -n '   '
map es --normalize unit center --whiten --src_dewhiten src --trg_dewhiten trg --trg_reweight --dim_reduction 216; evaluate es --retrieval invsoftmax --inv_temperature  1;  echo
echo    '-----------------------------------------------------------------------'

rm -rf $TMP
