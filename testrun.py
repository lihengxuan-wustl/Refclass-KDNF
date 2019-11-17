import subprocess
import multiprocessing

datasets = ['lending']
explainers = ['setcover']
models = ['logistic']
out = 'out_pickles'
n_cores=multiprocessing.cpu_count()
print n_cores

for dataset in datasets:
    for explainer in explainers:
        for model in models:
            for p in range(0,n_cores):
                outfile = 'tmp/%s-%s-%s-%s.log' % (dataset, explainer, model,p)
                print 'Outfile:', outfile
                outfile = open(outfile, 'w+',0)
                cmd = 'python compute_explanations.py -d %s -e %s -m %s -o %s -p %s -n %s' % (
                    dataset, explainer, model,
                    '%s/%s-%s-%s-%s.txt' % (out, dataset, explainer, model,p),n_cores,p)
                print cmd
                subprocess.Popen(cmd.split(), stdout=outfile)
