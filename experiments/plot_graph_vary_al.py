import os
import sys
import pickle as pkl
import traceback

import numpy as np
from scipy.special import softmax
import matplotlib.pyplot as plt
import tqdm

import matplotlib.pyplot as plt

GRAPH_ROOT_DIR = 'graphs/large'


exp_cases = sys.argv[1].split(',')

print(f'exp_cases = {exp_cases}')
print()

FIG_DIM = 5

plt.rcParams['figure.figsize'] = (5, 5)
plt.rcParams['figure.dpi'] = 300

plt.rcParams.update({
    'font.size': 20,
    'text.usetex': False,
    'font.family': 'Helvetica'
})
title_fontsize = 22

graph_cases = {
    'theory': {
        'random': (True, 'black', 'x-'),
        'ntkgp-frgp-ev': (True, 'blue', 'o-'),
        'ntkgp-frgp-ev-emp': (False, 'purple', 'p--'),
        'kmeans-pp': (True, 'orange', 'D-'),
        'badge': (False, 'red', 'v-'),
        'mlmoc-batched-randinit': (False, 'green', 's-'),
    },
    
    'emp': {
        'random': (True, 'black', 'x-'),
        'ntkgp-frgp-ev-emp': (False, 'blue', 'o-'),
        'kmeans-pp': (True, 'orange', 'D-'),
        'badge': (False, 'red', 'v-'),
        'mlmoc-batched-randinit': (False, 'green', 's-'),
    },
    
    'emp-alt': {
        'ntkgp-frgp-ev-emp': (False, 'blue', 'o-'),
        'badge': (False, 'red', 'v-'),
        'mlmoc-batched-randinit': (False, 'green', 's-'),
    },
}

fnames = {'test_labels', 'test_losses', 'test_preds'}

CASES = {
    2:  ('mnist_mlp-b', ((25, 100), (26, 200), (27, 400)), 'Batch size', False, 100, 400, 'theory,emp,emp-alt'),
    13: ('emnist_mlp-b', ((135, 100), (136, 200), (137, 400), (138, 800)), 'Batch size', False, 100, 800, 'theory,emp,emp-alt'),
    3:  ('emnist_cnn-b', ((35, 100), (36, 200), (37, 400), (38, 800)), 'Batch size', False, 100, 800, 'emp,emp-alt'),
    18: ('cifar10_wrn-b', ((180, 100), (181, 200), (182, 400), (183, 800)), 'Batch size', False, 100, 800, 'emp,emp-alt'),
    8:  ('svhn_wrn-b', ((85, 100), (86, 200), (87, 400), (88, 800)), 'Batch size', False, 100, 800, 'emp,emp-alt'),
}


LABELS = {
    'random': 'Random',
    'kmeans-pp': 'K-Means++',
    'badge': 'BADGE',
    'mlmoc-randinit': 'MLMOC',
    'mlmoc-batched-randinit': 'MLMOC',
    
    'ntkgp-frgp-ev': 'EV-GP',
    'ntkgp-frgp-mi': 'MI-GP',
    'ntkgp-frgp-90v': '90V-GP',
    'ntkgp-frgp-mv': '100V-GP',
    
    'ntkgp-sgp-ev': 'EV-sGP',
    'ntkgp-sgp-mi': 'MI-sGP',
    'ntkgp-sgp-mv': '100V-sGP',
    
    'ntkgp-frgp-ev-emp': 'EV-GP-Emp',
    'ntkgp-frgp-mi-emp': 'MI-GP-Emp',
    'ntkgp-frgp-90v-emp': '90V-GP-Emp',
    'ntkgp-frgp-mv-emp': '100V-GP-Emp',
}


def entropy(d):
    nans = np.isnan(d['test_preds']).any(axis=(1, 2))
    preds = d['test_preds'][~nans, :]
    logits = np.array([(np.argmax(preds, axis=-1) == i).sum(axis=0) for i in range(preds.shape[-1])]).T / preds.shape[0]
    ent = np.sum(np.nan_to_num(- logits * np.log(logits, where=(logits > 0))), axis=1)
    return ent


def filter_nan(arr):
    return arr[~np.isnan(arr).any(axis=(1, 2))]


for e_id in exp_cases:
    
    prefix, exp_code_ints, xlabel, is_regr, epoch_use, max_sz, graph_cases_check = CASES[int(e_id)]
    subfolders = ['training'] if is_regr else ['training', 'training-mse']
    
    graph_cases_check = graph_cases_check.split(',')
    alg_to_read = {y for x in graph_cases_check for y in graph_cases[x].keys()}
                    
    for subfolder in subfolders:
    
        processed_data = dict()
        
        for exp_code_int, x_val in exp_code_ints:
        
            exp_code = f'exp{exp_code_int}'
            base_folder = f'./results_main/{exp_code}'
                            
            print(f'############# {exp_code} {subfolder} #############')
            training_data = dict()
        
            try:
                
                for case_name in sorted(list(os.listdir(base_folder)))[::-1]:
                    folder = f'{base_folder}/{case_name}'
                
                    if os.path.isdir(folder):
                        
                        alg = case_name.split('_')[1]
                        if alg not in alg_to_read:
                            continue
                        
                        print(folder)
                        data = dict()
                        for cc, _, fs in os.walk(f'{folder}/{subfolder}'):
                            if f'epoch={epoch_use}' not in cc:
                                continue
                            c = cc.split(folder)[1][1:]
                            ck = c.split(subfolder)[1][1:]
                            it, sz, ep = ck.split('/')
                            it = int(it.split('=')[1])
                            sz = int(sz.split('=')[1])
                            ep = int(ep.split('=')[1])
                            ck = (it, sz, ep)
                            
                            if sz != max_sz:
                                continue
                            
                            if ck not in data.keys():
                                data[ck] = dict()
                            
                            for f in fs:
                                if f.endswith('.pickle'):
                                    f = f.split('.pickle')[0]
                                else:
                                    continue
                                    
                                if f in fnames:
                                    x = np.load(f'{folder}/{c}/{f}.pickle', allow_pickle=True)
                                    x = x[np.argsort(x[:,0]), 1:]
                                    data[ck][f] = x
                        
                        if alg not in training_data:
                            training_data[alg] = []
                        training_data[alg].append(data)

                iters = set()
                epochs = set()

                for alg in training_data.keys():
                    for d in training_data[alg]:
                        for it, sz, ep in d.keys():
                            iters.add(it)
                            epochs.add(ep)

                iters = sorted(list(iters))
                epochs = sorted(list(epochs))

                if subfolder == 'training':
                    true_label = lambda d: d['test_labels'].squeeze(-1)
                else:
                    true_label = lambda d: np.argmax(d['test_labels'], axis=2)

                for alg in training_data.keys():
                    proc_d = []
                    for d in training_data[alg]:
                        tot = 0
                        try:
                            proc_d_dict = dict()
                            for f in fnames:
                                proc_d_dict[f] = np.array([d[i, max_sz, epochs[-1]][f] for i in iters])
                                
                            if is_regr:
                                proc_d_dict['test_predvar'] = np.var(filter_nan(proc_d_dict['test_preds']), axis=(1, 2))
                            else:
                                proc_d_dict['test_ent'] = entropy(proc_d_dict)
                                proc_d_dict['test_acc'] = (np.argmax(filter_nan(proc_d_dict['test_preds']), axis=2) == 
                                                        true_label(proc_d_dict)[0][None, :])
                            proc_d.append(proc_d_dict)
                            tot += 1
                        except KeyError:
                            pass
                        
                    if alg not in processed_data.keys():
                        processed_data[alg] = dict()
                    processed_data[alg][x_val] = proc_d
                    
            except Exception as e:
                print(str(type(e)), ':', str(e))
                continue
            
        xvals = sorted([s[1] for s in exp_code_ints])
        print('xvals =', xvals)

        for g in graph_cases_check:
            
            algs_list = graph_cases[g]
            graph_out = f'{GRAPH_ROOT_DIR}/{g}' 
            
            print(f'========== {graph_out} {prefix} {subfolder} ==========')
            
            try:
                
                algs = [a for a in algs_list.keys() if a in processed_data.keys()]
                print('algs =', algs)

                def plot_data(indiv_fn, errbar=None, xlabel=None, ylabel=None, legend=True, fname=None, legend_path=None):
                    
                    fig, ax = plt.subplots()
                    for alg in algs:
                        try:
                            use_prev_val, colour, ls = algs_list[alg]
                            
                            ys, ys_std = [], []
                            for sz in xvals:
                                if use_prev_val and sz > xvals[0]:
                                    ys.append(ys[0])
                                    ys_std.append(ys_std[0])
                                    print(alg, sz, '- reuse values')
                                    continue
                                if sz not in processed_data[alg].keys() or len(processed_data[alg]) == 0:
                                    ys.append(np.nan)
                                    ys_std.append(np.nan)
                                    print(alg, sz, '- no key')
                                    continue
                                    
                                rec1 = [indiv_fn(d) for d in processed_data[alg][sz]]
                                if len(rec1) == 0:
                                    ys.append(np.nan)
                                    ys_std.append(np.nan)
                                    print(alg, sz, '- no records')
                                    continue
                                assert not isinstance(rec1[0], np.ndarray), rec1[0]
                                rec = [x for x in rec1 if not np.isnan(x).any()]
                                if (len(rec1) > len(rec)) or (len(rec) < 5):
                                    print(alg, sz, f'- records={len(rec1)}, usable={len(rec)}')
                                rec = rec[:5]  # limit to 5 trials
                                ys.append(np.mean(rec))
                                if callable(errbar):
                                    rec_err1 = [errbar(d) for d in processed_data[alg][sz]]
                                    assert not isinstance(rec1[0], np.ndarray), rec1[0]
                                    rec_err = [x for x in rec_err1 if not np.isnan(x).any() and np.abs(x) < 1e9]
                                    ys_std.append(np.mean(rec_err))
                                else:
                                    ys_std.append(np.std(rec))
                        except Exception as e:
                            print(traceback.format_exc())
                            pass
                        
                        zorder = 2.5 if ('ntkgp' in alg) else 2
                        ax.errorbar(
                            xvals,
                            ys,
                            yerr=ys_std if errbar else None,
                            fmt=ls,
                            color=colour,
                            label=LABELS[alg],
                            ms=10,
                            alpha=0.8,
                            capsize=3,
                            zorder=zorder
                        )
                        
                    ax.set_xlabel(xlabel)
                    ax.set_ylabel(ylabel)
                    # ax.grid(True)
                    fig.tight_layout()
                    if fname:
                        os.makedirs(os.path.dirname(fname), exist_ok=True)
                        fig.savefig(fname)
                    if legend_path is not None:
                        # ax.legend().set_zorder(4.)
                        fig_leg = plt.figure()
                        ax_leg = fig_leg.add_subplot(111)
                        # add the legend from the previous axes
                        ax_leg.legend(*ax.get_legend_handles_labels(), loc='center')
                        # hide the axes frame and the x/y labels
                        ax_leg.axis('off')
                        fig_leg.savefig(f'{legend_path}/legend.png')
                        plt.close(fig_leg)
                    plt.close(fig)

                    return fig
                
                graph_out_prefix = f'{graph_out}/{prefix}/{subfolder}'

                plot_data(
                    indiv_fn=lambda d: np.nanmean(d['test_losses']), 
                    errbar=True,
                    xlabel=xlabel,
                    ylabel='Mean test loss',
                    fname=f'{graph_out_prefix}/test-loss-mean',
                    legend_path=graph_out_prefix
                )


                plot_data(
                    indiv_fn=lambda d: np.nanmean(d['test_losses'][d['test_losses'] < 1e6]), 
                    errbar=lambda d: np.nanstd(np.nanmean(d['test_losses'][(d['test_losses'] < 1e6).all(axis=(1, 2))], axis=(1, 2))),
                    xlabel=xlabel,
                    ylabel='Mean test loss',
                    fname=f'{graph_out_prefix}/test-loss-mean-w-stderr'
                )


                plot_data(
                    indiv_fn=lambda d: np.nanmean(d['test_losses'][d['test_losses'] < 1e6]), 
                    errbar=lambda d: np.nanstd(np.nanmean(d['test_losses'][(d['test_losses'] < 1e6).all(axis=(1, 2))], axis=(1, 2))),
                    xlabel=xlabel,
                    ylabel='Mean test loss',
                    fname=f'{graph_out_prefix}/test-loss-mean-w-stderr'
                )


                plot_data(
                    indiv_fn=lambda d: np.nanstd(np.nanmean(d['test_losses'][(d['test_losses'] < 1e6).all(axis=(1, 2))], axis=(1, 2))),
                    errbar=True,
                    xlabel=xlabel,
                    ylabel='Std. dev. test loss',
                    fname=f'{graph_out_prefix}/test-loss-sd',
                )
                
                
                if is_regr:
                    plot_data(
                        indiv_fn=lambda d: np.mean(d['test_predvar']),
                        errbar=True,
                        xlabel=xlabel,
                        ylabel='Mean output variance',
                        fname=f'{graph_out_prefix}/pred-var-mean',
                    )


                    plot_data(
                        indiv_fn=lambda d: np.percentile(d['test_predvar'], 90), 
                        errbar=True,
                        xlabel=xlabel,
                        ylabel='90th %ile output variance',
                        fname=f'{graph_out_prefix}/pred-var-90p',
                    )


                    plot_data(
                        indiv_fn=lambda d: np.percentile(d['test_predvar'], 99), 
                        errbar=True,
                        xlabel=xlabel,
                        ylabel='99th %ile output variance',
                        fname=f'{graph_out_prefix}/pred-var-99p',
                    )
                
                else:
                    
                    plot_data(
                        indiv_fn=lambda d: np.mean(d['test_ent']),
                        errbar=True,
                        xlabel=xlabel,
                        ylabel='Mean output entropy',
                        fname=f'{graph_out_prefix}/pred-ent-mean',
                    )


                    plot_data(
                        indiv_fn=lambda d: np.percentile(d['test_ent'], 90), 
                        errbar=True,
                        xlabel=xlabel,
                        ylabel='90th %ile output entropy',
                        fname=f'{graph_out_prefix}/pred-ent-90p',
                    )


                    plot_data(
                        indiv_fn=lambda d: np.percentile(d['test_ent'], 99), 
                        errbar=True,
                        xlabel=xlabel,
                        ylabel='99th %ile output entropy',
                        fname=f'{graph_out_prefix}/pred-ent-99p',
                    )


                    plot_data(
                        indiv_fn=lambda d: np.mean(d['test_acc']),
                        errbar=True,
                        xlabel=xlabel,
                        ylabel='Mean test accuracy',
                        fname=f'{graph_out_prefix}/acc-mean',
                    )


                    plot_data(
                        lambda d: np.std(np.nanmean(d['test_acc'], axis=1)),
                        errbar=False,
                        xlabel=xlabel,
                        ylabel='Std. dev. test accuracy',
                        fname=f'{graph_out_prefix}/acc-sd',
                    )
                    
            except Exception as e:
                print(traceback.format_exc())
                pass
            
    print('--------------- DONE ---------------\n')