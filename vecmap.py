import numpy as np
from cupy_utils import supports_cupy, get_cupy, get_array_module, asnumpy
import embeddings
import re
import time
import collections


def topk_mean(m, k, inplace=False):  # TODO Assuming that axis is 1
    xp = get_array_module(m)
    n = m.shape[0]
    ans = xp.zeros(n, dtype=m.dtype)
    if k <= 0:
        return ans
    if not inplace:
        m = xp.array(m)
    ind0 = xp.arange(n)
    ind1 = xp.empty(n, dtype=int)
    minimum = m.min()
    for i in range(k):
        m.argmax(axis=1, out=ind1)
        ans += m[ind0, ind1]
        m[ind0, ind1] = minimum
    return ans / k


def dropout(m, p):
    if p <= 0.0:
        return m
    else:
        xp = get_array_module(m)
        mask = xp.random.rand(*m.shape) >= p
        return m*mask


class VecMap:
    def __init__(
        self,
        training_mode : str = 'advanced', # 'orthogonal' or 'unconstrained' or 'advanced'
        whiten : bool = False,
        src_reweight : float = 0.,
        trg_reweight : float = 0.,
        src_dewhiten : str = None, # 'src' or 'trg'
        trg_dewhiten : str = None, # 'src' or 'trg'
        dim_reduction : int = None,
        init_dictionary_mode : str = 'unsupervised', # 'unsupervised', 'numerals', 'identical', 'seed'
        dictionary_induction_direction : str = 'union', # 'forward', 'backward', 'union'
        unsupervised_dictionary_size : int = None,
        vocabulary_cutoff : int = None,
        normalization_actions : list = [], # ['unit', 'center', 'unitdim', 'centeremb', 'none']
        file_encodings : str = 'utf-8',
        csls : int = 0,
        dtype : str = 'float32',
    ) -> None:
        VecMap._check_training_mode_validity(training_mode)
        VecMap._check_dewhiten_mode_validity(src_dewhiten)
        VecMap._check_dewhiten_mode_validity(trg_dewhiten)
        VecMap._check_whitening_and_dewhitening_concordancy(whiten, src_dewhiten, trg_dewhiten)
        VecMap._check_init_dictionary_mode(init_dictionary_mode)
        VecMap._check_normalization_actions(normalization_actions)

        self._training_mode = training_mode
        self._whiten_is_active = whiten
        self._src_dewhiten = src_dewhiten
        self._trg_dewhiten = trg_dewhiten
        self.x_rw = src_reweight
        self.z_rw = trg_reweight
        self.dim_reduction = dim_reduction
        self._dictionary_induction_direction = dictionary_induction_direction
        self._init_dictionary_mode = init_dictionary_mode
        self._unsupervised_dictionary_size = np.inf if unsupervised_dictionary_size is None else unsupervised_dictionary_size
        self._cutoff = np.inf if vocabulary_cutoff is None else vocabulary_cutoff
        self.normalization_actions = normalization_actions
        self.encoding = file_encodings
        self.csls = csls
        self.dtype = dtype
        
        self.dictionary = None
        self.validation_set = None
        self.xp = np


    @staticmethod
    def _check_dewhiten_mode_validity(mode):
        if mode not in ['src', 'trg']:
            raise Exception(f"De-whitening mode can be either 'src' or 'trg', not {mode}")
    @staticmethod
    def _check_whitening_and_dewhitening_concordancy(whiten, src_dewhiten, trg_dewhiten):
        if (src_dewhiten is not None or trg_dewhiten is not None) and not whiten:
            raise Exception('ERROR: De-whitening requires whitening first')
    @staticmethod
    def _check_training_mode_validity(mode):
        if mode not in ['orthogonal', 'unconstrained', 'advanced']:
            raise Exception(f"Training mode can be either 'orthogonal', 'unconstrained', or 'advanced', not {mode}")
    @staticmethod
    def _check_dictionary_induction_direction(direction):
        if direction not in ['forward', 'backward', 'union']:
            raise Exception(f"Dictionary induction direction can be either 'forward', 'backward', or 'union', not {direction}")
    @staticmethod
    def _check_init_dictionary_mode(mode):
        if mode not in ['unsupervised', 'numerals', 'identical', 'seed']:
            raise Exception(f"Init ictionary mode can be either 'unsupervised', 'numerals', 'identical', or 'seed', not {mode}")
    @staticmethod
    def _check_normalization_actions(actions):
        for action in actions:
            if action not in ['unit', 'center', 'unitdim', 'centeremb', 'none']:
                raise Exception("Invalid normalization actions.")
    @staticmethod
    def _check_seed_dictionary(mode, path):
        if mode=='seed' and path is None:
            raise Exception("Init seed dictionary path is missing. Set the `seed_dictionary` argument.")

    
    def to_cuda(self, device_id=None):
        if not supports_cupy():
            raise Exception('ERROR: Install CuPy for CUDA support')
        self.xp = get_cupy()
        if device_id is not None:
            self.xp.cuda.Device(device_id).use()

    def set_seed(self, seed):
        self.xp.random.seed(seed)


    def _whitening_transformation(self, m):
        u, s, vt = self.xp.linalg.svd(m, full_matrices=False)
        return vt.T.dot(self.xp.diag(1/s)).dot(vt)
    

    def _actual_whiten(self, a, indices):
        w1 = self._whitening_transformation(a[indices])
        return a.dot(w1), w1
    

    def _whiten(self, a, indices):
        if self._whiten_is_active:
            return self._actual_whiten(a, indices)
        else:
            return a, None
    
    
    def _bidirectional_orthogonal_map(self, x, ix, z, iz):
        wx, s, wz_t = self.xp.linalg.svd(x[ix].T.dot(z[iz]))
        wz = wz_t.T
        return x.dot(wx), z.dot(wz), wx, wz, s
    

    def _reweight(self, a, s, rw=0):
        return a * s**rw
    

    def _actual_dewhiten(self, a, w1, w2):
        return a.dot(w2.T.dot(self.xp.linalg.inv(w1)).dot(w2))
    
    
    def _dewhiten(self, a, wx1, wz1, wx2, wz2, mode):
        if mode is None:
            return a
        if mode=='src':
            w1, w2 = wx1, wx2
        if mode=='trg':
            w1, w2 = wz1, wz2
        return self._actual_dewhiten(a, w1, w2)
    

    def _dim_reduction(self, a):
        return a[:, :self.dim_reduction]
    

    def _advanced_map(self, x, ix, z, iz):
        xw = x.copy()
        zw = z.copy()

        xw, wx1 = self._whiten(xw, ix)
        zw, wz1 = self._whiten(zw, iz)

        xw, zw, wx2, wz2, s = self._bidirectional_orthogonal_map(xw, ix, zw, iz)

        xw = self._reweight(xw, s, self.x_rw)
        zw = self._reweight(zw, s, self.z_rw)

        xw = self._dewhiten(xw, wx1, wz1, wx2, wz2, self._src_dewhiten)
        zw = self._dewhiten(zw, wx1, wz1, wx2, wz2, self._trg_dewhiten)

        xw = self._dim_reduction(xw)
        zw = self._dim_reduction(zw)

        return xw, zw
    

    def _orthogonal_map(self, x, ix, z, iz):
        u, s, v_t = self.xp.linalg.svd(z[iz].T.dot(x[ix]))
        xw = x.dot(v_t.T.dot(u.T))
        zw = z.copy()
        return xw, zw
    

    def _unconstrained_map(self, x, ix, z, iz):
        w = self.xp.linalg.inv(x[ix].T.dot(x[ix])).dot(x[ix].T).dot(z[iz])
        xw = x.dot(w)
        zw = z.copy()
        return xw, zw
    

    def _map(self, x, ix, z, iz, is_last_iteration):
        if self._training_mode=='advanced':
            if is_last_iteration:
                return self._advanced_map(x, ix, z, iz)
            else:
                return self._orthogonal_map(x, ix, z, iz)
        if self._training_mode=='orthogonal':
            return self._orthogonal_map(x, ix, z, iz)
        if self._training_mode=='unconstrained':
            return self._unconstrained_map(x, ix, z, iz)
        

    def _get_indices(self, sim):
        if self._dictionary_induction_direction=='forward':
            return self.xp.arange(sim.shape[1]), sim.argmax(axis=1)
        if self._dictionary_induction_direction=='backward':
            return sim.argmax(axis=0), self.xp.arange(sim.shape[0])
        if self._dictionary_induction_direction=='union':
            return self.xp.concatenate((self.xp.arange(sim.shape[1]), sim.argmax(axis=0))), \
                   self.xp.concatenate((sim.argmax(axis=1), self.xp.arange(sim.shape[0])))
        

    def _build_unsupervised_seed_dictionary(self):
        size = min(self.x.shape[0], self.z.shape[0], self._unsupervised_dictionary_size)
        u, s, vt = self.xp.linalg.svd(self.x[:size], full_matrices=False)
        xsim = (u*s).dot(u.T)
        u, s, vt = self.xp.linalg.svd(self.z[:size], full_matrices=False)
        zsim = (u*s).dot(u.T)
        del u, s, vt
        xsim.sort(axis=1)
        zsim.sort(axis=1)
        embeddings.normalize(xsim, self.normalization_actions)
        embeddings.normalize(zsim, self.normalization_actions)
        sim = xsim.dot(zsim.T)
        if self.csls > 0:
            knn_sim_fwd = topk_mean(sim, k=self.csls)
            knn_sim_bwd = topk_mean(sim.T, k=self.csls)
            sim -= knn_sim_fwd[:, self.xp.newaxis]/2 + knn_sim_bwd/2
        return self._get_indices(sim)
    

    def _build_identical_seed_dictionary(self, x_words, z_words):
        identicals = set(x_words).intersection(set(z_words))
        xi, zi = [], []
        for word in identicals:
            xi.append(self.src_word2ind[word])
            zi.append(self.trg_word2ind[word])
        return xi, zi
    

    def _build_numerals_seed_dictionary(self):
        numeral_regex = re.compile('^[0-9]+$')
        x_numerals = {word for word in self.x_words if numeral_regex.match(word) is not None}
        z_numerals = {word for word in self.z_words if numeral_regex.match(word) is not None}
        return self._build_identical_seed_dictionary(x_numerals, z_numerals)


    def _read_dictionary(self, path):
        xi, zi = [], []
        oov = []
        with open(path, encoding=self.encoding, errors='surrogateescape') as f:
            for line in f:
                x_word, z_word = line.split()
                try:
                    x, z = self.src_word2ind[x_word], self.trg_word2ind[z_word]
                    xi.append(x)
                    zi.append(z)
                except KeyError:
                    oov.append((x_word, z_word))
        return xi, zi, oov


    def _build_seed_dictionary(self, seed_dictionary_path):
        if self._init_dictionary_mode=='unsupervised':
            return self._build_unsupervised_seed_dictionary()
        if self._init_dictionary_mode=='numerals':
            return self._build_numerals_seed_dictionary()
        if self._init_dictionary_mode=='identical':
            return self._build_identical_seed_dictionary(self.x_words, self.z_words)
        if self._init_dictionary_mode=='seed':
            xi, zi, _ = self._read_dictionary(seed_dictionary_path)
            return xi, zi
        
    
    def _select_indices_for_dictionary(self, x, x_size, z, z_size, keep_prob, batch_size):
        knn_sim = self.xp.zeros(z_size, dtype=self.dtype)
        if self.csls > 0:
            for b in range(0, z_size, batch_size):
                sim = z[b:b+batch_size].dot(x[:x_size].T)
                knn_sim[b:b+batch_size] = topk_mean(sim, k=self.csls, inplace=True)
        best_sim = self.xp.full(x_size, -100, dtype=self.dtype)
        indices = self.xp.zeros(x_size, dtype=int)
        for b in range(0, x_size, batch_size):
            sim = x[b:b+batch_size].dot(z[:z_size].T)
            best_sim[b:b+batch_size] = sim.max(axis=1)
            sim -= knn_sim/2
            indices[b:b+batch_size] = dropout(sim, 1 - keep_prob).argmax(axis=1)
        return indices, self.xp.mean(best_sim)
    

    def _rebuild_dictionary(self, x, z, keep_prob, batch_size):
        x_size = min(x.shape[0], self._cutoff)
        z_size = min(z.shape[0], self._cutoff)
        if self._dictionary_induction_direction=='forward':
            xi = self.xp.arange(x_size)
            zi, objective = self._select_indices_for_dictionary(x, x_size, z, z_size, keep_prob, batch_size)
        if self._dictionary_induction_direction=='backward':
            xi, objective = self._select_indices_for_dictionary(z, z_size, x, x_size, keep_prob, batch_size)
            zi = self.xp.arange(z_size)
        if self._dictionary_induction_direction=='union':
            xi, objective1 = self._select_indices_for_dictionary(z, z_size, x, x_size, keep_prob, batch_size)
            zi, objective2 = self._select_indices_for_dictionary(x, x_size, z, z_size, keep_prob, batch_size)
            objective = (objective1 + objective2) / 2
            xi = self.xp.concatenate((self.xp.arange(x_size), xi))
            zi = self.xp.concatenate((zi, self.xp.arange(z_size)))
        return (xi, zi), objective


    def set_validation_dictionary(self, path):
        xi, zi, _ = self._read_dictionary(path)
        self.validation_set = collections.defaultdict(set)
        for x, z in zip(xi, zi):
            self.validation_set[x].add(z)
        print(f"Validation set length: {len(self.validation_set)}")


    def validate(self, x, z):
        src = list(self.validation_set.keys())
        simval = x[src].dot(z.T)
        nn = asnumpy(simval.argmax(axis=1))
        accuracy = np.mean([(nn[i] in self.validation_set[src[i]]) for i in range(len(src))])
        similarity = np.mean([max([simval[i, j].tolist() for j in self.validation_set[src[i]]]) for i in range(len(src))])
        return accuracy, similarity


    def _log(self, itr, duration, objective, keep_prob, similarity=None, accuracy=None):
        print(f'ITERATION {itr} - DURATION {duration}s:')
        print(f'\t- Objective: {objective}')
        print(f'\t- Drop probability: {1-keep_prob}')
        if accuracy is not None:
            print(f'\t- Val. similarity:  {similarity}')
            print(f'\t- Val. accuracy:    {accuracy}')


    def set_train_data(
        self,
        src_input : str,
        trg_input : str,
        seed_dictionary : str = None,
    ) -> None:
        VecMap._check_seed_dictionary(self._init_dictionary_mode, seed_dictionary)
        with open(src_input, encoding=self.encoding, errors='surrogateescape') as srcfile:
            self.x_words, x = embeddings.read(srcfile, dtype=self.dtype)
        with open(trg_input, encoding=self.encoding, errors='surrogateescape') as trgfile:
            self.z_words, z = embeddings.read(trgfile, dtype=self.dtype)
        self.x = self.xp.asarray(x)
        self.z = self.xp.asarray(z)
        embeddings.normalize(self.x, self.normalization_actions)
        embeddings.normalize(self.z, self.normalization_actions)
        self.src_word2ind = {word: i for i, word in enumerate(self.x_words)}
        self.trg_word2ind = {word: i for i, word in enumerate(self.z_words)}
        self.ix, self.iz = self._build_seed_dictionary(seed_dictionary)


    def train(
        self,
    ) -> None:
        self.xw, self.zw = self._map(self.x, self.ix, self.z, self.iz, True)


    def self_learning_train(
        self,
        dict_update_batch_size : int = 10000,
        stochastic_initial : float = 0.1,
        objective_threshold : float = 0.000001,
        stochastic_interval : int = 50,
        stochastic_multiplier : float = 2.,
        log : bool = True,
    ) -> None:
        xw, zw = self.x.copy(), self.z.copy()

        finish = False
        best_objective = objective = -100
        last_improvement = -1
        t0 = None
        keep_prob = stochastic_initial
        
        itr = 0
        while not finish:
            if itr:
                (self.ix, self.iz), objective = self._rebuild_dictionary(xw, zw, keep_prob, dict_update_batch_size)
            itr += 1

            if objective - best_objective >= objective_threshold:
                last_improvement = itr
                best_objective = objective

            if log:
                accuracy, similarity = None, None
                if self.validation_set:
                    accuracy, similarity = self.validate(xw, zw)

                duration = time.time() - t0 if t0 is not None else 0
                self._log(itr, duration, objective, keep_prob, similarity, accuracy)
                
            if keep_prob >= 1.0:
                finish = True
            if itr - last_improvement > stochastic_interval:
                keep_prob = min(1.0, stochastic_multiplier*keep_prob)
                last_improvement = itr

            t0 = time.time()
            xw, zw = self._map(self.x, self.ix, self.z, self.iz, finish)

        self.xw, self.zw = xw, zw
        

    def save_embeddings(
        self,
        src_output : str,
        trg_output : str,
    ) -> None:
        for words, emb, file_name in [[self.x_words, self.xw, src_output],
                                             [self.z_words, self.zw, trg_output]]:
            with open(file_name, mode='w', encoding=self.encoding, errors='surrogateescape') as f:
                embeddings.write(words, emb, f)

