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

import argparse
import sys
from vecmap import VecMap


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Map word embeddings in two languages into a shared space')
    parser.add_argument('src_input', help='the input source embeddings')
    parser.add_argument('trg_input', help='the input target embeddings')
    parser.add_argument('src_output', help='the output source embeddings')
    parser.add_argument('trg_output', help='the output target embeddings')
    parser.add_argument('--encoding', default='utf-8', help='the character encoding for input/output (defaults to utf-8)')
    parser.add_argument('--precision', choices=['float16', 'float32', 'float64'], default='float32', help='the floating-point precision (defaults to float32)')
    parser.add_argument('--cuda', action='store_true', help='use cuda (requires cupy)')
    parser.add_argument('--device_id', default=None, type=int, help='device ID used for the --cuda option (defaults to 0)')
    parser.add_argument('--pca', action='store_true', help='use pca first')
    parser.add_argument('--batch_size', default=10000, type=int, help='batch size (defaults to 10000); does not affect results, larger is usually faster but uses more memory')
    parser.add_argument('--seed', type=int, default=0, help='the random seed (defaults to 0)')

    recommended_group = parser.add_argument_group('recommended settings', 'Recommended settings for different scenarios')
    recommended_type = recommended_group.add_mutually_exclusive_group()
    recommended_type.add_argument('--supervised', metavar='DICTIONARY', help='recommended if you have a large training dictionary')
    recommended_type.add_argument('--semi_supervised', metavar='DICTIONARY', help='recommended if you have a small seed dictionary')
    recommended_type.add_argument('--identical', action='store_true', help='recommended if you have no seed dictionary but can rely on identical words')
    recommended_type.add_argument('--unsupervised', action='store_true', help='recommended if you have no seed dictionary and do not want to rely on identical words')
    recommended_type.add_argument('--acl2018', action='store_true', help='reproduce our ACL 2018 system')
    recommended_type.add_argument('--aaai2018', metavar='DICTIONARY', help='reproduce our AAAI 2018 system')
    recommended_type.add_argument('--acl2017', action='store_true', help='reproduce our ACL 2017 system with numeral initialization')
    recommended_type.add_argument('--acl2017_seed', metavar='DICTIONARY', help='reproduce our ACL 2017 system with a seed dictionary')
    recommended_type.add_argument('--emnlp2016', metavar='DICTIONARY', help='reproduce our EMNLP 2016 system')

    init_group = parser.add_argument_group('advanced initialization arguments', 'Advanced initialization arguments')
    init_type = init_group.add_mutually_exclusive_group()
    init_type.add_argument('-d', '--init_dictionary', default=sys.stdin.fileno(), metavar='DICTIONARY', help='the training dictionary file (defaults to stdin)')
    init_type.add_argument('--init_identical', action='store_true', help='use identical words as the seed dictionary')
    init_type.add_argument('--init_numerals', action='store_true', help='use latin numerals (i.e. words matching [0-9]+) as the seed dictionary')
    init_type.add_argument('--init_unsupervised', action='store_true', help='use unsupervised initialization')
    init_group.add_argument('--unsupervised_vocab', type=int, default=None, help='restrict the vocabulary to the top k entries for unsupervised initialization')

    mapping_group = parser.add_argument_group('advanced mapping arguments', 'Advanced embedding mapping arguments')
    mapping_group.add_argument('--normalize', choices=['unit', 'center', 'unitdim', 'centeremb', 'none'], nargs='*', default=[], help='the normalization actions to perform in order')
    mapping_group.add_argument('--whiten', action='store_true', help='whiten the embeddings')
    mapping_group.add_argument('--src_reweight', type=float, default=0, nargs='?', const=1, help='re-weight the source language embeddings')
    mapping_group.add_argument('--trg_reweight', type=float, default=0, nargs='?', const=1, help='re-weight the target language embeddings')
    mapping_group.add_argument('--src_dewhiten', choices=['src', 'trg'], help='de-whiten the source language embeddings')
    mapping_group.add_argument('--trg_dewhiten', choices=['src', 'trg'], help='de-whiten the target language embeddings')
    mapping_group.add_argument('--dim_reduction', type=int, default=None, help='apply dimensionality reduction')
    mapping_type = mapping_group.add_mutually_exclusive_group()
    mapping_type.add_argument('-c', '--orthogonal', action='store_true', help='use orthogonal constrained mapping')
    mapping_type.add_argument('-u', '--unconstrained', action='store_true', help='use unconstrained mapping')

    self_learning_group = parser.add_argument_group('advanced self-learning arguments', 'Advanced arguments for self-learning')
    self_learning_group.add_argument('--self_learning', action='store_true', help='enable self-learning')
    self_learning_group.add_argument('--vocabulary_cutoff', type=int, default=None, help='restrict the vocabulary to the top k entries')
    self_learning_group.add_argument('--direction', choices=['forward', 'backward', 'union'], default='union', help='the direction for dictionary induction (defaults to union)')
    self_learning_group.add_argument('--csls', type=int, nargs='?', default=0, const=10, metavar='NEIGHBORHOOD_SIZE', dest='csls_neighborhood', help='use CSLS for dictionary induction')
    self_learning_group.add_argument('--threshold', default=0.000001, type=float, help='the convergence threshold (defaults to 0.000001)')
    self_learning_group.add_argument('--validation', default=None, metavar='DICTIONARY', help='a dictionary file for validation at each iteration')
    self_learning_group.add_argument('--stochastic_initial', default=0.1, type=float, help='initial keep probability stochastic dictionary induction (defaults to 0.1)')
    self_learning_group.add_argument('--stochastic_multiplier', default=2.0, type=float, help='stochastic dictionary induction multiplier (defaults to 2.0)')
    self_learning_group.add_argument('--stochastic_interval', default=50, type=int, help='stochastic dictionary induction interval (defaults to 50)')
    args = parser.parse_args()

    if args.supervised is not None:
        parser.set_defaults(init_dictionary=args.supervised, normalize=['unit', 'center', 'unit'], whiten=True, src_reweight=0.5, trg_reweight=0.5, src_dewhiten='src', trg_dewhiten='trg', batch_size=1000)
    if args.semi_supervised is not None:
        parser.set_defaults(init_dictionary=args.semi_supervised, normalize=['unit', 'center', 'unit'], whiten=True, src_reweight=0.5, trg_reweight=0.5, src_dewhiten='src', trg_dewhiten='trg', self_learning=True, vocabulary_cutoff=20000, csls_neighborhood=10)
    if args.identical:
        parser.set_defaults(init_identical=True, normalize=['unit', 'center', 'unit'], whiten=True, src_reweight=0.5, trg_reweight=0.5, src_dewhiten='src', trg_dewhiten='trg', self_learning=True, vocabulary_cutoff=20000, csls_neighborhood=10)
    if args.unsupervised or args.acl2018:
        parser.set_defaults(init_unsupervised=True, unsupervised_vocab=4000, normalize=['unit', 'center', 'unit'], whiten=True, src_reweight=0.5, trg_reweight=0.5, src_dewhiten='src', trg_dewhiten='trg', self_learning=True, vocabulary_cutoff=20000, csls_neighborhood=10)
    if args.aaai2018:
        parser.set_defaults(init_dictionary=args.aaai2018, normalize=['unit', 'center'], whiten=True, trg_reweight=1, src_dewhiten='src', trg_dewhiten='trg', batch_size=1000)
    if args.acl2017:
        parser.set_defaults(init_numerals=True, orthogonal=True, normalize=['unit', 'center'], self_learning=True, direction='forward', stochastic_initial=1.0, stochastic_interval=1, batch_size=1000)
    if args.acl2017_seed:
        parser.set_defaults(init_dictionary=args.acl2017_seed, orthogonal=True, normalize=['unit', 'center'], self_learning=True, direction='forward', stochastic_initial=1.0, stochastic_interval=1, batch_size=1000)
    if args.emnlp2016:
        parser.set_defaults(init_dictionary=args.emnlp2016, orthogonal=True, normalize=['unit', 'center'], batch_size=1000)
    args = parser.parse_args()


    training_mode = 'orthogonal' if args.orthogonal else \
                    'unconstrained' if args.unconstrained else \
                    'advanced'
    init_dictionary_mode =  'unsupervised' if args.init_unsupervised else \
                            'identical' if args.init_identical else \
                            'numerals' if args.init_numerals else \
                            'seed'
    vecmap = VecMap(
        training_mode = training_mode,
        whiten = args.whiten,
        src_reweight = args.src_reweight,
        trg_reweight = args.trg_reweight,
        src_dewhiten = args.src_dewhiten,
        trg_dewhiten = args.trg_dewhiten,
        dim_reduction = args.dim_reduction,
        init_dictionary_mode = init_dictionary_mode,
        dictionary_induction_direction = args.direction,
        unsupervised_dictionary_size = args.unsupervised_vocab,
        vocabulary_cutoff = args.vocabulary_cutoff,
        normalization_actions = args.normalize,
        file_encodings = args.encoding,
        csls = args.csls_neighborhood,
        dtype = args.precision,
    )
    if args.cuda:
        vecmap.to_cuda(args.device_id)
    vecmap.set_seed(args.seed)
    vecmap.set_train_data(
        src_input = args.src_input,
        trg_input = args.trg_input,
        seed_dictionary = args.init_dictionary,
        pca = args.pca,
    )
    if args.validation is not None:
        vecmap.set_validation_dictionary(args.validation)
    if args.self_learning:
        vecmap.self_learning_train(
            dict_update_batch_size = args.batch_size,
            objective_threshold = args.threshold,
            stochastic_interval = args.stochastic_interval,
            stochastic_initial = args.stochastic_initial,
        )
    else:
        vecmap.train()
    vecmap.save_embeddings(
        src_output = args.src_output,
        trg_output = args.trg_output,
    )

if __name__ == '__main__':
    main()
