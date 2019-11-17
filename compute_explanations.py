from __future__ import print_function
import argparse
import pickle
import xgboost
import sklearn
import sklearn.neural_network
import utils
import anchor_tabular
import numpy as np

# get the range of number that represents categories:
def get_range(dataset):
    maxs = []
    mins = []
    rg = []
    for column in dataset.T:
        maxs.append(max(column))
        mins.append(min(column))
        rg.append(int(max(column) - min(column) + 1))
    return rg, maxs, mins


# transform dataset into binary features
def binary_transform(dataset, maxs, mins):
    l = len(maxs)
    # print(maxs)
    b_validation = []
    for row in dataset:
        newrow = []
        for i in range(l):
            newfeature = np.zeros(int(maxs[i] - mins[i] + 1), dtype=int)
            newfeature[int(row[i])] = 1
            # print(row[i])
            newrow = np.concatenate((newrow, newfeature))
        b_validation.append(newrow)
    return np.array(b_validation).astype(int)


def main():
    # sys.stdout = open('log.txt', 'w')
    parser = argparse.ArgumentParser(description='Compute some explanations.')
    parser.add_argument('-d', dest='dataset', required=True,
                        choices=['adult', 'recidivism', 'lending'],
                        help='dataset to use')
    parser.add_argument('-e', dest='explainer', required=True,
                        choices=['lime', 'anchor', 'setcover'],
                        help='explainer, either anchor or lime or setcover')
    parser.add_argument('-m', dest='model', required=True,
                        choices=['xgboost', 'logistic', 'nn'],
                        help='model: xgboost, logistic or nn')
    parser.add_argument('-c', dest='checkpoint', required=False,
                        default=5, type=int,
                        help='checkpoint after this many explanations')
    parser.add_argument('-o', dest='output', required=True)
    #parser.add_argument('-p',dest='n_cores',required=True,type=int)
    parser.add_argument('-n',dest='n_process',required=True,type=int)

    args = parser.parse_args()
    print(args.output)
    dataset = utils.load_dataset(args.dataset, balance=True)
    ret = {}
    ret['dataset'] = args.dataset
    for x in ['train_idx', 'test_idx', 'validation_idx']:
        ret[x] = getattr(dataset, x)

    explainer = anchor_tabular.AnchorTabularExplainer(
        dataset.class_names, dataset.feature_names,
        dataset.data, dataset.categorical_names)
    explainer.fit(dataset.train, dataset.labels_train,
                  dataset.validation, dataset.labels_validation, dataset.test, dataset.labels_test)

    if args.model == 'xgboost':
        c = xgboost.XGBClassifier(n_estimators=400, nthread=10, seed=1)
        c.fit(explainer.encoder.transform(dataset.train), dataset.labels_train)
    if args.model == 'logistic':
        c = sklearn.linear_model.LogisticRegression()
        c.fit(explainer.encoder.transform(dataset.train), dataset.labels_train)
    if args.model == 'nn':
        c = sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(50, 50))
        c.fit(explainer.encoder.transform(dataset.train), dataset.labels_train)

    explainer.compute_cx(c)

    # print(dataset.validation)
    # print(dataset.labels_train)

    ret['encoder'] = explainer.encoder
    ret['model'] = c
    ret['model_name'] = args.model

    def predict_fn(x):
        return c.predict(explainer.encoder.transform(x))

    def predict_proba_fn(x):
        return c.predict_proba(explainer.encoder.transform(x))

    print('Train', sklearn.metrics.accuracy_score(dataset.labels_train,
                                                  predict_fn(dataset.train)))
    print('Validation', sklearn.metrics.accuracy_score(dataset.labels_validation,
                                                       predict_fn(dataset.validation)))
    print('Test', sklearn.metrics.accuracy_score(dataset.labels_test,
                                                 predict_fn(dataset.test)))
    threshold = 0.95
    tau = 0.1
    delta = 0.05
    epsilon_stop = 0.05
    batch_size = 100
    if args.explainer == 'anchor':
        explain_fn = utils.get_reduced_explain_fn(
            explainer.explain_lucb_beam, c.predict, threshold=threshold,
            delta=delta, tau=tau, batch_size=batch_size / 2,
            sample_whole_instances=True,
            beam_size=10, epsilon_stop=epsilon_stop)
    elif args.explainer == 'lime':
        explain_fn = utils.get_reduced_explain_fn(
            explainer.explain_lime, c.predict_proba, num_features=5,
            use_same_dist=True)
    elif args.explainer == 'setcover':
        # set miu and gamma
        miu = 0.1
        gamma = 0
        if(args.dataset=='adult'):
            if(args.model=='nn'):
                miu=0.076
            if (args.model == 'xgboost'):
                miu = 0.097
            if (args.model == 'logistic'):
                miu = 0.107
        if (args.dataset == 'lending'):
            if (args.model == 'nn'):
                miu = 0.166
            if (args.model == 'xgboost'):
                miu = 0.284
            if (args.model == 'logistic'):
                miu = 0.286
        if (args.dataset == 'recidivism'):
            if (args.model == 'nn'):
                miu = 0.011
            if (args.model == 'xgboost'):
                miu = 0.048
            if (args.model == 'logistic'):
                miu = 0.068
        rg, maxs, mins = get_range(dataset.validation)
        explain_fn = utils.get_reduced_explain_fn(
            explainer.explain_setcover, c.predict, rg=rg, miu=miu, gamma=gamma
        )

    if args.explainer == 'setcover':
        explainer.generate_sets(explainer.b_validation[0], k=3, rg=rg)  # change k for k-dnf here
        ret['exps'] = []
        #n_points=len(explainer.validation)/args.n_cores
        # start=args.n_process*n_points
        # end=(args.n_process+1)*n_points if args.n_process+1<args.n_cores else len(explainer.validation)
        f=open(args.output,'w',0)
        i=args.n_process
        print('generate ref class for')
        print(i)
        err,exp = explain_fn(i)
        print('exp')
        print(exp)
        # ret['exps'].append(exp)
        precision,coverage=explainer.compute_precision_coverage(exp)
        print (precision,coverage)
        f.write(str(exp))
        f.write('\n')
        f.write(str(precision))
        f.write('\n')
        f.write(str(coverage))
        f.write('\n')
        f.close()

        #pickle.dump(ret, open(args.output, 'w'))
    else:
        ret['exps'] = []
        for i, d in enumerate(dataset.validation, start=1):
            print('generate anchor/lime for')
            print(i)
            print(d)
            if i % 100 == 0:
                print(i)
            if i % args.checkpoint == 0:
                print('Checkpointing')
                pickle.dump(ret, open(args.output + '.checkpoint', 'w'))
            exp = explain_fn(d)
            print('exp')
            print(exp)
            ret['exps'].append(exp)

        pickle.dump(ret, open(args.output, 'w'))


if __name__ == '__main__':
    main()
