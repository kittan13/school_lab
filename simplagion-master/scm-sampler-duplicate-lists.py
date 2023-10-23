import subprocess as sp
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns  # for aesthetic
#%matplotlib inline
sns.set_style('ticks')
import tqdm 
from tqdm import tqdm_notebook

from simplex_utils import *

def facet_list_to_graph(facet_list):
    """Convert a facet list to a bipartite graph"""
    g = nx.Graph()
    for f, facet in enumerate(facet_list):
        for v in facet:
            g.add_edge('v' + str(v), 'f' + str(f))  # differentiate node types
    return g

def seq_to_file(nums,f):
    with open(f, mode='wb', encoding='utf-8') as myfile:
        myfile.write(' '.join(map(str,nums)));
        

def facets_to_file(cliques,f):
    with open(f, mode='w', encoding='utf-8') as myfile:
        for clique in cliques:
            myfile.write(' '.join(str(el) for el in clique))
            myfile.write('\n')       


def sanitize_sequence(sequence):
    count = 0;
    relabel = {}
    new_cliques = []
    for clique in sequence:
        for el in clique:
            if el not in relabel:
                relabel[el] = count
                count+=1;
        new_cliques.append(list(map(lambda x: relabel[x], clique)));   
    return new_cliques;

"""
def sanitize_sequence(sequence):
    count = 0
    relabel = {}
    new_cliques = []
    for clique in sequence:
        new_clique = []  # タプルに変換する前の新しいクリークリスト
        for el in clique:
            if el not in relabel:
                relabel[el] = count
                count += 1
            new_clique.append(relabel[el])
        new_cliques.append(tuple(new_clique))  # タプルに変換して追加
    return new_cliques
"""



def rejection_sampling(command, seed=0):
    # Call sampler with subprocess
    proc = sp.run(command, stdout=sp.PIPE)
    # Read output as a facet list 
    facet_list = []
    for line in proc.stdout.decode().split("\n")[1:-1]:
        if line.find("#") == 0:
            yield facet_list
            facet_list = []
        else:
            facet_list.append([int(x) for x in line.strip().split()])
    yield facet_list
    


def multiply_cliques(cliques, n):
    s = []
    sc = sanitize_sequence(cliques)
    for el in sc:
        s.extend(el)
    m = np.max(list(set(s)))
    new_cliques = []
    new_cliques.extend(sc)
    for nn in range(1,n):
        new_cliques.extend(map(lambda x: list(np.array(x)+nn*m), sc))
    return new_cliques



datasetlist = ['InVS13','InVS15','LH10','SFHH','LyonSchool'] #['Thiers13'] 
nms = [5,10]#,15,30]
thrs = [0.8] # [0.95, 0.9, 0.85, 0.8]
#thr_dir = 'thr_data/'
thr_dir = "C:/kittan/lab/simplagion-master/Data/Sociopatterns/thr_data/"
output_dir = 'thr_data_duplicate/'
iterations = 10
n = 5

from random import seed, randint
seed(int(time()))

for dataset in tqdm_notebook(datasetlist):
    for nm in tqdm_notebook(nms):
        for thr in thrs:
            filename = thr_dir+'aggr_'+str(nm)+'_'+str(thr)+'min_cliques_'+dataset+'.json'
            #filename = r'C:\kittan\lab\simplagion-master\Data\Sociopatterns\thr_data_random\random_5_0.8min_cliques_InVS13.json'
            cliques = json.load(open(filename));
            sc = sanitize_sequence(cliques);
            sc = multiply_cliques(sc, n);
            facets_to_file(sc, './tmp/facet_list.txt')
            command = ['scm/bin/mcmc_sampler', './tmp/facet_list.txt', '-t', str(iterations),
                    '-c', '-f', '10000'] #'--seed='+str(randint(0,10000000)),'-b', '500', '-f', '500'
            jd = open(output_dir+'random_'+str(nm)+'_'+str(thr)+'min_cliques_'+dataset+'.json','w')
            ls = []
            for facet_list in rejection_sampling(command):
                ls.append(facet_list)
            json.dump(ls,jd)
            jd.close()


deh = json.load(open('./thr_data_duplicate/random_15_0.8min_cliques_Thiers13.json'))

deh[0][:5], deh[1][:5]
