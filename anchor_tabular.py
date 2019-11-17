import anchor_base
import lime
import lime.lime_tabular
import collections
import sklearn
import numpy as np
import compute_explanations
import math
import copy
import itertools
import math
import setcoverinc

class AnchorTabularExplainer(object):
    """
    bla
    """

    def __init__(self, class_names, feature_names, data=None,
                 categorical_names=None, ordinal_features=[]):
        self.encoder = collections.namedtuple('random_name',
                                              ['transform'])(lambda x: x)
        self.categorical_features = []
        if categorical_names:
            # TODO: Check if this n_values is correct!!
            cat_names = sorted(categorical_names.keys())
            n_values = [len(categorical_names[i]) for i in cat_names]
            self.encoder = sklearn.preprocessing.OneHotEncoder(
                categorical_features=cat_names,
                n_values=n_values)
            self.encoder.fit(data)
            self.categorical_features = self.encoder.categorical_features
        self.ordinal_features = ordinal_features
        self.feature_names = feature_names
        self.class_names = class_names
        self.categorical_names = categorical_names

    def fit(self, train_data, train_labels, validation_data,
            validation_labels, test_data, test_labels):
        """
        bla
        """
        self.min = {}
        self.max = {}
        self.std = {}
        self.train = train_data
        self.train_labels = train_labels
        self.validation = validation_data
        self.validation_labels = validation_labels
        self.test = test_data
        self.test_labels = test_labels
        all_data = np.concatenate((train_data, validation_data, test_data))
        rg, maxs, mins = compute_explanations.get_range(all_data)
        print rg
        self.b_validation = compute_explanations.binary_transform(validation_data, maxs, mins)
        self.b_test = compute_explanations.binary_transform(self.test, maxs, mins)
        self.m = len(self.b_validation)
        print self.m
        self.scaler = sklearn.preprocessing.StandardScaler()
        self.scaler.fit(train_data)
        # self.discretizer = lime.lime_tabular.DecileDiscretizer(
        #     train_data, self.categorical_features, self.feature_names)
        for f in range(train_data.shape[1]):
            if f in self.categorical_features:
                continue
            self.min[f] = np.min(train_data[:, f])
            self.max[f] = np.max(train_data[:, f])
            self.std[f] = np.std(train_data[:, f])

        self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            train_data, feature_names=self.feature_names,
            categorical_features=self.categorical_features,
            categorical_names=self.categorical_names, verbose=False,
            class_names=self.class_names)

    def compute_cx(self, c):
        self.predict_validation_labels = c.predict(self.encoder.transform(self.validation))
        self.predict_test_labels = c.predict(self.encoder.transform(self.test))
        # self.predict_validation_labels=[]
        # for row in self.validation:
        #     self.predict_validation_labels.append(c.predict(row.reshape(1,-1)))
        # self.cx=np.logical_xor(self.predict_validation_labels,self.validation_labels).astype(int)
        self.cx = []
        for i in range(len(self.predict_validation_labels)):
            self.cx.append(1 if self.predict_validation_labels[i] == self.validation_labels[i] else 0)
        self.cx_test = []
        for i in range(len(self.predict_test_labels)):
            self.cx_test.append(1 if self.predict_test_labels[i] == self.test_labels[i] else 0)
            # print('cx')
            # print(self.cx)

    def generate_sets(self, data_row, k, rg):
        self.k = k
        self.sets_feature = []
        self.r_sets = []
        self.b_sets = []
        rg_prefix_sum = [0]
        for i in range(1, len(rg)):
            rg_prefix_sum.append(rg_prefix_sum[i - 1] + rg[i - 1])
        if (k == 1):
            for i in range(len(data_row)):
                # literal (i,1/0) means feature i, 0 or 1 means is it negate
                feature = (i, 0)
                n_feature = (i, 1)
                self.sets_feature.append(feature)
                self.sets_feature.append(n_feature)
                # generate elements for each set
                set1 = set([])
                n_set = set([])
                r_set = set([])
                r_n_set = set([])
                for j, d in enumerate(self.b_validation):
                    if (self.satisfy(feature, d)):
                        set1.add(j)
                        if (self.cx[j] == 0):
                            r_set.add(j)
                    else:
                        n_set.add(j)
                        if (self.cx[j] == 0):
                            r_n_set.add(j)
                self.b_sets.append(set1)
                self.b_sets.append(n_set)
                self.r_sets.append(r_set)
                self.r_sets.append(r_n_set)

        if (k == 2):
            iterable = []
            for i in range(len(rg)):
                iterable.append(i)
            comb = list(itertools.combinations(iterable, 2))
            for (i, j) in comb:
                # print i,j
                for a in range(rg[i]):
                    for b in range(rg[j]):
                        f1 = (rg_prefix_sum[i] + a, 0, rg_prefix_sum[j] + b, 0)
                        f2 = (rg_prefix_sum[i] + a, 0, rg_prefix_sum[j] + b, 1)
                        f3 = (rg_prefix_sum[i] + a, 1, rg_prefix_sum[j] + b, 0)
                        f4 = (rg_prefix_sum[i] + a, 1, rg_prefix_sum[j] + b, 1)
                        set1 = set([])
                        set2 = set([])
                        set3 = set([])
                        set4 = set([])
                        set1_r = set([])
                        set2_r = set([])
                        set3_r = set([])
                        set4_r = set([])
                        for idx, z in enumerate(self.b_validation):
                            if (self.satisfy(f1, z)):
                                set1.add(idx)
                                if (self.cx[idx] == 0):
                                    set1_r.add(idx)
                            elif (self.satisfy(f2, z)):
                                set2.add(idx)
                                if (self.cx[idx] == 0):
                                    set2_r.add(idx)
                            elif (self.satisfy(f3, z)):
                                set3.add(idx)
                                if (self.cx[idx] == 0):
                                    set3_r.add(idx)
                            elif (self.satisfy(f4, z)):
                                set4.add(idx)
                                if (self.cx[idx] == 0):
                                    set4_r.add(idx)
                        self.sets_feature.append(f1)
                        self.sets_feature.append(f2)
                        self.sets_feature.append(f3)
                        self.sets_feature.append(f4)
                        self.b_sets.append(set1)
                        self.b_sets.append(set2)
                        self.b_sets.append(set3)
                        self.b_sets.append(set4)
                        self.r_sets.append(set1_r)
                        self.r_sets.append(set2_r)
                        self.r_sets.append(set3_r)
                        self.r_sets.append(set4_r)

        if (k == 3):
            iterable = []
            for i in range(len(rg)):
                iterable.append(i)
            comb = list(itertools.combinations(iterable, 3))
            for (i, j, y) in comb:
                # print i,j
                for a in range(rg[i]):
                    for b in range(rg[j]):
                        for c in range(rg[y]):
                            f1 = (rg_prefix_sum[i] + a, 0, rg_prefix_sum[j] + b, 0, rg_prefix_sum[y] + c, 0)
                            f2 = (rg_prefix_sum[i] + a, 0, rg_prefix_sum[j] + b, 1, rg_prefix_sum[y] + c, 0)
                            f3 = (rg_prefix_sum[i] + a, 1, rg_prefix_sum[j] + b, 0, rg_prefix_sum[y] + c, 0)
                            f4 = (rg_prefix_sum[i] + a, 1, rg_prefix_sum[j] + b, 1, rg_prefix_sum[y] + c, 0)
                            f5 = (rg_prefix_sum[i] + a, 0, rg_prefix_sum[j] + b, 0, rg_prefix_sum[y] + c, 1)
                            f6 = (rg_prefix_sum[i] + a, 0, rg_prefix_sum[j] + b, 1, rg_prefix_sum[y] + c, 1)
                            f7 = (rg_prefix_sum[i] + a, 1, rg_prefix_sum[j] + b, 0, rg_prefix_sum[y] + c, 1)
                            f8 = (rg_prefix_sum[i] + a, 1, rg_prefix_sum[j] + b, 1, rg_prefix_sum[y] + c, 1)
                            self.sets_feature.append(f1)
                            self.sets_feature.append(f2)
                            self.sets_feature.append(f3)
                            self.sets_feature.append(f4)
                            self.sets_feature.append(f5)
                            self.sets_feature.append(f6)
                            self.sets_feature.append(f7)
                            self.sets_feature.append(f8)
                            set1 = set([])
                            set2 = set([])
                            set3 = set([])
                            set4 = set([])
                            set5 = set([])
                            set6 = set([])
                            set7 = set([])
                            set8 = set([])
                            set1_r = set([])
                            set2_r = set([])
                            set3_r = set([])
                            set4_r = set([])
                            set5_r = set([])
                            set6_r = set([])
                            set7_r = set([])
                            set8_r = set([])
                            for idx, z in enumerate(self.b_validation):
                                if (self.satisfy(f1, z)):
                                    set1.add(idx)
                                    if (self.cx[idx] == 0):
                                        set1_r.add(idx)
                                elif (self.satisfy(f2, z)):
                                    set2.add(idx)
                                    if (self.cx[idx] == 0):
                                        set2_r.add(idx)
                                elif (self.satisfy(f3, z)):
                                    set3.add(idx)
                                    if (self.cx[idx] == 0):
                                        set3_r.add(idx)
                                elif (self.satisfy(f4, z)):
                                    set4.add(idx)
                                    if (self.cx[idx] == 0):
                                        set4_r.add(idx)
                                elif (self.satisfy(f5, z)):
                                    set5.add(idx)
                                    if (self.cx[idx] == 0):
                                        set5_r.add(idx)
                                elif (self.satisfy(f6, z)):
                                    set6.add(idx)
                                    if (self.cx[idx] == 0):
                                        set6_r.add(idx)
                                elif (self.satisfy(f7, z)):
                                    set7.add(idx)
                                    if (self.cx[idx] == 0):
                                        set7_r.add(idx)
                                elif (self.satisfy(f8, z)):
                                    set8.add(idx)
                                    if (self.cx[idx] == 0):
                                        set8_r.add(idx)
                            self.b_sets.append(set1)
                            self.b_sets.append(set2)
                            self.b_sets.append(set3)
                            self.b_sets.append(set4)
                            self.b_sets.append(set5)
                            self.b_sets.append(set6)
                            self.b_sets.append(set7)
                            self.b_sets.append(set8)
                            self.r_sets.append(set1_r)
                            self.r_sets.append(set2_r)
                            self.r_sets.append(set3_r)
                            self.r_sets.append(set4_r)
                            self.r_sets.append(set5_r)
                            self.r_sets.append(set6_r)
                            self.r_sets.append(set7_r)
                            self.r_sets.append(set8_r)

                            # debug
                            # for i,set in enumerate(self.b_sets):
                            #     print(self.sets_feature[i])
                            #     print(set)

    def sample_from_train(self, conditions_eq, conditions_neq, conditions_geq,
                          conditions_leq, num_samples, validation=False):
        """
        bla
        """
        train = self.train if not validation else self.validation
        idx = np.random.choice(range(train.shape[0]), num_samples,
                               replace=True)
        sample = train[idx]
        for f in conditions_eq:
            sample[:, f] = np.repeat(conditions_eq[f], num_samples)
        for f in conditions_geq:
            idx = sample[:, f] <= conditions_geq[f]
            if f in conditions_leq:
                idx = (idx + (sample[:, f] > conditions_leq[f])).astype(bool)
            if idx.sum() == 0:
                continue
            options = train[:, f] > conditions_geq[f]
            if f in conditions_leq:
                options = options * (train[:, f] <= conditions_leq[f])
            if options.sum() == 0:
                min_ = conditions_geq.get(f, self.min[f])
                max_ = conditions_leq.get(f, self.max[f])
                to_rep = np.random.uniform(min_, max_, idx.sum())
            else:
                to_rep = np.random.choice(train[options, f], idx.sum(),
                                          replace=True)
            sample[idx, f] = to_rep
        for f in conditions_leq:
            if f in conditions_geq:
                continue
            idx = sample[:, f] > conditions_leq[f]
            if idx.sum() == 0:
                continue
            options = train[:, f] <= conditions_leq[f]
            if options.sum() == 0:
                min_ = conditions_geq.get(f, self.min[f])
                max_ = conditions_leq.get(f, self.max[f])
                to_rep = np.random.uniform(min_, max_, idx.sum())
            else:
                to_rep = np.random.choice(train[options, f], idx.sum(),
                                          replace=True)
            sample[idx, f] = to_rep
        return sample

    def explain_lime(self, data_row, predict_proba_fn, num_features=5,
                     with_intercept=False, use_same_dist=False):
        """predict_proba_fn is original function"""

        def predict_fn(x):
            return predict_proba_fn(self.encoder.transform(x))

        def clean_fnames(aslist):
            import re
            ret = []
            for x in aslist:
                strz = x[0]
                fname = strz[:strz.find('=')]
                fvalue = strz[strz.find('=') + 1:]
                if fname in fvalue:
                    strz = fvalue
                ret.append((strz, x[1]))
            return ret

        if use_same_dist:
            # Doing lime here with different distribution
            kernel_width = np.sqrt(data_row.shape[0]) * .75

            def kernel(d):
                return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

            base = lime.lime_base.LimeBase(kernel, verbose=False)
            sample_fn, mapping = self.get_sample_fn(data_row, predict_proba_fn,
                                                    sample_whole_instances=True)
            raw, data, _ = sample_fn([], 5000, False)
            distances = sklearn.metrics.pairwise_distances(
                data,
                np.ones(data[0].reshape(1, -1).shape),
                metric='euclidean').ravel()
            pred_one = predict_fn(raw[0].reshape(1, -1))[0].argmax()
            labels = predict_fn(raw)
            if len(self.class_names) == 2:
                label = 1
            else:
                label = pred_one
            intercept, local_exp, score, _ = base.explain_instance_with_data(
                data, labels, distances, label, num_features)
            ret_exp = {}
            ret_exp['as_map'] = dict(local_exp)

            values = []
            for f, w in local_exp:
                v = data_row[f]
                fname = '%s = ' % self.feature_names[f]
                v = int(v)
                if ('<' in self.categorical_names[f][v]
                    or '>' in self.categorical_names[f][v]):
                    fname = ''
                fname = '%s%s' % (fname, self.categorical_names[f][v])
                values.append((fname, w))
            ret_exp['linear_model'] = values
            ret_exp['intercept'] = intercept
            return ret_exp

        ret_exp = {}
        features_to_use = num_features + 1 if with_intercept else num_features
        if len(self.class_names) == 2:
            # ret_exp['label_names'] = list(self.class_names)
            exp = self.lime_explainer.explain_instance(
                data_row, predict_fn, num_features=features_to_use)
            label = 1
            # baseline = ('Baseline', exp.intercept
            # ret_exp['data'] = exp.as_list(1)
        else:
            exp = self.lime_explainer.explain_instance(
                data_row, predict_fn, num_features=features_to_use,
                top_labels=1)
            label = exp.as_map().keys()[0]
            # label_name = self.class_names[label]
            # ret_exp['label_names'] = ['NOT %s' % label_name, label_name]
            # ret_exp['data'] = exp.as_list(label)
        intercept = exp.intercept[label] - 0.5
        linear_model = clean_fnames(exp.as_list(label))
        ret_exp['as_map'] = dict(exp.as_map()[label])
        if np.abs(intercept) > np.abs(linear_model[-1][1]) and with_intercept:
            linear_model = [('Baseline', intercept)] + linear_model[:-1]
            ret_exp['as_map'] = dict(sorted(ret_exp['as_map'].items(),
                                            key=lambda x: np.abs(x[1]))[1:])
        ret_exp['linear_model'] = linear_model
        ret_exp['intercept'] = exp.intercept[label]
        return ret_exp

    def get_sample_fn(self, data_row, classifier_fn,
                      sample_whole_instances=True, desired_label=None):
        def predict_fn(x):
            return classifier_fn(self.encoder.transform(x))

        true_label = desired_label
        if true_label is None:
            true_label = predict_fn(data_row.reshape(1, -1))[0]
        # must map present here to include categorical features (for conditions_eq), and numerical features for geq and leq
        mapping = {}
        for f in self.categorical_features:
            if f in self.ordinal_features:
                for v in range(len(self.categorical_names[f])):
                    idx = len(mapping)
                    if data_row[f] <= v:
                        mapping[idx] = (f, 'leq', v)
                        # names[idx] = '%s <= %s' % (self.feature_names[f], v)
                    elif data_row[f] > v:
                        mapping[idx] = (f, 'geq', v)
                        # names[idx] = '%s > %s' % (self.feature_names[f], v)
            else:
                idx = len(mapping)
                mapping[idx] = (f, 'eq', data_row[f])
                # names[idx] = '%s = %s' % (
                #     self.feature_names[f],
                #     self.categorical_names[f][int(data_row[f])])

        def sample_fn(present, num_samples, compute_labels=True):
            conditions_eq = {}
            conditions_leq = {}
            conditions_geq = {}
            for x in present:
                f, op, v = mapping[x]
                if op == 'eq':
                    conditions_eq[f] = v
                if op == 'leq':
                    if f not in conditions_leq:
                        conditions_leq[f] = v
                    conditions_leq[f] = min(conditions_leq[f], v)
                if op == 'geq':
                    if f not in conditions_geq:
                        conditions_geq[f] = v
                    conditions_geq[f] = max(conditions_geq[f], v)
            # conditions_eq = dict([(x, data_row[x]) for x in present])
            raw_data = self.sample_from_train(
                conditions_eq, {}, conditions_geq, conditions_leq, num_samples,
                validation=sample_whole_instances)
            data = np.zeros((num_samples, len(mapping)), int)
            for i in mapping:
                f, op, v = mapping[i]
                if op == 'eq':
                    data[:, i] = (raw_data[:, f] == data_row[f]).astype(int)
                if op == 'leq':
                    data[:, i] = (raw_data[:, f] <= v).astype(int)
                if op == 'geq':
                    data[:, i] = (raw_data[:, f] > v).astype(int)
            # data = (raw_data == data_row).astype(int)
            labels = []
            if compute_labels:
                labels = (predict_fn(raw_data) == true_label).astype(int)
            return raw_data, data, labels

        return sample_fn, mapping

    def explain_lucb_beam(self, data_row, classifier_fn, threshold=1,
                          delta=0.05, tau=0.1, batch_size=10,
                          max_anchor_size=None,
                          desired_label=None,
                          sample_whole_instances=True, **kwargs):
        # It's possible to pass in max_anchor_size
        sample_fn, mapping = self.get_sample_fn(
            data_row, classifier_fn, sample_whole_instances,
            desired_label=desired_label)
        # print(sample_fn)
        print('mapping')
        print(mapping)
        exp = anchor_base.AnchorBaseBeam.anchor_beam(
            sample_fn, delta=delta, epsilon=tau, batch_size=batch_size,
            desired_confidence=threshold, max_anchor_size=max_anchor_size,
            **kwargs)
        self.add_names_to_exp(data_row, exp, mapping)
        return exp

    def add_names_to_exp(self, data_row, hoeffding_exp, mapping):
        # TODO: precision recall is all wrong, coverage functions wont work
        # anymore due to ranges
        idxs = hoeffding_exp['feature']
        hoeffding_exp['names'] = []
        hoeffding_exp['feature'] = [mapping[idx][0] for idx in idxs]
        ordinal_ranges = {}
        for idx in idxs:
            f, op, v = mapping[idx]
            if op == 'geq' or op == 'leq':
                if f not in ordinal_ranges:
                    ordinal_ranges[f] = [float('-inf'), float('inf')]
            if op == 'geq':
                ordinal_ranges[f][0] = max(ordinal_ranges[f][0], v)
            if op == 'leq':
                ordinal_ranges[f][1] = min(ordinal_ranges[f][1], v)
        handled = set()
        for idx in idxs:
            f, op, v = mapping[idx]
            # v = data_row[f]
            if op == 'eq':
                fname = '%s = ' % self.feature_names[f]
                if f in self.categorical_names:
                    v = int(v)
                    if ('<' in self.categorical_names[f][v]
                        or '>' in self.categorical_names[f][v]):
                        fname = ''
                    fname = '%s%s' % (fname, self.categorical_names[f][v])
                else:
                    fname = '%s%.2f' % (fname, v)
            else:
                if f in handled:
                    continue
                geq, leq = ordinal_ranges[f]
                fname = ''
                geq_val = ''
                leq_val = ''
                if geq > float('-inf'):
                    name = self.categorical_names[f][geq + 1]
                    if '<' in name:
                        geq_val = name.split()[0]
                    elif '>' in name:
                        geq_val = name.split()[-1]
                if leq < float('inf'):
                    name = self.categorical_names[f][leq]
                    if leq == 0:
                        leq_val = name.split()[-1]
                    elif '<' in name:
                        leq_val = name.split()[-1]
                if leq_val and geq_val:
                    fname = '%s < %s <= %s' % (geq_val, self.feature_names[f],
                                               leq_val)
                elif leq_val:
                    fname = '%s <= %s' % (self.feature_names[f], leq_val)
                elif geq_val:
                    fname = '%s > %s' % (self.feature_names[f], geq_val)
                handled.add(f)
            hoeffding_exp['names'].append(fname)

    def explain_setcover(self, xstar, classifier_fn, rg, miu, gamma
                         ):
        lowest_err = 1
        low_err_kdnf = None
        R = len(self.cx) - np.array(self.cx).sum()  # of red elements
        true_miu=(1-float(gamma)/2)*miu
        print('R=')
        print(R)
        X=1
        while(X<=R):
            print('X=')
            print(X)
            err, kdnf = setcoverinc.low_deg_partial(X, xstar, true_miu,self.b_sets,self.r_sets,self.k,self.m,self.sets_feature)
            if (err < lowest_err):
                low_err_kdnf = kdnf
                lowest_err=err
            X=math.ceil(X*1.1)
        return lowest_err, low_err_kdnf

    # whether a data point satisfy a term
    def satisfy(self, dnf, data_row):
        if (self.k == 1):
            (i, j) = dnf
            if (j == 0):
                return data_row[i].astype(bool)
            else:
                return not data_row[i].astype(bool)
        elif (self.k == 2):
            (i1, j1, i2, j2) = dnf
            if (j1 == 0):
                if (j2 == 0):
                    return data_row[i1].astype(bool) and data_row[i2].astype(bool)
                else:
                    return data_row[i1].astype(bool) and not data_row[i2].astype(bool)
            else:
                if (j2 == 0):
                    return not data_row[i1].astype(bool) and data_row[i2].astype(bool)
                else:
                    return not data_row[i1].astype(bool) and not data_row[i2].astype(bool)
        elif (self.k == 3):
            (i1, j1, i2, j2, i3, j3) = dnf
            if (j1 == 0):
                if (j2 == 0):
                    if (j3 == 0):
                        return data_row[i1].astype(bool) and data_row[i2].astype(bool) and data_row[i2].astype(bool)
                    else:
                        return data_row[i1].astype(bool) and data_row[i2].astype(bool) and not data_row[i2].astype(
                            bool)
                else:
                    if (j3 == 0):
                        return data_row[i1].astype(bool) and not data_row[i2].astype(bool) and data_row[i2].astype(
                            bool)
                    else:
                        return data_row[i1].astype(bool) and not data_row[i2].astype(bool) and not data_row[
                            i2].astype(
                            bool)
            else:
                if (j2 == 0):
                    if (j3 == 0):
                        return not data_row[i1].astype(bool) and data_row[i2].astype(bool) and data_row[i2].astype(
                            bool)
                    else:
                        return not data_row[i1].astype(bool) and data_row[i2].astype(bool) and not data_row[
                            i2].astype(
                            bool)
                else:
                    if (j3 == 0):
                        return not data_row[i1].astype(bool) and not data_row[i2].astype(bool) and data_row[
                            i2].astype(
                            bool)
                    else:
                        return not data_row[i1].astype(bool) and not data_row[i2].astype(bool) and not data_row[
                            i2].astype(
                            bool)
        else:
            return True

    def compute_precision_coverage(self, kdnf):
        sat = []
        n_sat = 0
        n_sat_cx = 0
        for i, d in enumerate(self.b_test):
            flag = False
            for term in kdnf:
                if (self.satisfy(term, d)):
                    flag = True
                    n_sat += 1
                    if (self.cx_test[i] == 1):
                        n_sat_cx += 1
                    break
            sat.append(flag)
        coverage = n_sat * 1.0 / len(self.b_test)
        precision = n_sat_cx * 1.0 / n_sat
        return precision, coverage

