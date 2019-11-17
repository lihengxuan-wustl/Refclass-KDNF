import subprocess
import multiprocessing
import argparse



datasets = ['recidivism']
explainers = ['setcover']
models = ['xgboost','nn','logistic']
out = 'out_pickles'

parser = argparse.ArgumentParser(description='ArrayID')
parser.add_argument('-i', dest='p', required=True)
args=parser.parse_args()

print args.p

#n_cores=multiprocessing.cpu_count()
#print n_cores

list=[]

for dataset in datasets:
    for explainer in explainers:
        for model in models:
            #for p in range(0,n_cores):
                outfile = 'tmp/%s-%s-%s-%s.log' % (dataset, explainer, model,args.p)
                print 'Outfile:', outfile
                outf = open(outfile, 'w+',0)
                cmd = 'python compute_explanations.py -d %s -e %s -m %s -o %s -n %s' % (
                    dataset, explainer, model,
                    '%s/%s-%s-%s-%s.txt' % (out, dataset, explainer, model,args.p),args.p)
                print cmd
                list.append(subprocess.Popen(cmd.split(), stdout=outf))

for item in list:
    item.wait()
