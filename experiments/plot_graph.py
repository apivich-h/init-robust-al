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

FIG_DIM = 5

plt.rcParams['figure.figsize'] = (5, 5)
plt.rcParams['figure.dpi'] = 300

plt.rcParams.update({
    'font.size': 20,
    'text.usetex': False,
    'font.family': 'Helvetica'
})
title_fontsize = 22

exp_cases = sys.argv[1].split(',')

print(f'exp_cases = {exp_cases}')
print()

fnames = {'test_labels', 'test_losses', 'test_preds'}

CASES = {
    'exp1' :  (True,  1000,  10,  50, 'uci_boston_10',          'regr,regr-bb,approx,crits,crits-no100'), 
    'exp3' :  (True,  1000,  20, 100, 'uci_kin8nm_20',          'regr,regr-bb,approx,crits,crits-no100'), 
    'exp4' :  (False, 1000,  20, 100, 'uci_optdigits_20',       'regr,regr-bb,approx,crits,crits-no100'), 
    'exp7' :  (True,  1000,  20, 100, 'uci_protein_20',         'regr,regr-bb,approx,crits,crits-no100'), 
    'exp11':  (True,  1000,  20, 100, 'uci_optdigits_regr_20',  'regr,regr-bb,approx,crits,crits-no100'), 
    'exp12':  (True,  1000,  20, 100, 'uci_naval_20',           'regr,regr-bb,approx,crits,crits-no100'), 
    
    'exp25':  (False, 100,  100,  400, 'mnist_mlp_100',         'theory,theory-bb,emp,emp-bb,bbt,approx,crits,randinit'), 
    'exp26':  (False, 100,  200,  400, 'mnist_mlp_200',         'theory,emp,approx,crits'), 
    
    'exp35':  (False, 100,  100,  600, 'emnist_cnn_100',        'emp,emp-bb,bbe,randinit'), 
    'exp36':  (False, 100,  200,  600, 'emnist_cnn_200',        'emp'),
     
    'exp135': (False, 100,  100,  600, 'emnist_mlp_100',        'theory,theory-bb,emp,emp-bb,randinit'), 
    'exp136': (False, 100,  200,  600, 'emnist_mlp_200',        'theory,emp'), 
    
    'exp85':  (False, 100,  100,  800, 'svhn_wrn_100',          'emp,randinit'),
    'exp86':  (False, 100,  200,  800, 'svhn_wrn_200',          'emp'),
    'exp185': (False, 100,  500, 2000, 'svhn_wrn_500',          'emp-l'),
    'exp285': (False, 100,  100,  800, 'svhn_wrn-a_100',        'emp,randinit'),
    'exp289': (False, 100,  500, 2000, 'svhn_wrn-a_200',        'emp-l'),
    'exp386': (False, 1000, 200, 1000, 'svhn_rn18_200',         'emp-l'),
    
    'exp391': (False, 1000, 200, 1000, 'cifar10_rn18_200',      'emp-l'),
    'exp392': (False, 1000, 500, 2000, 'cifar10_rn18_500',      'emp-l'),
    
    'exp91':  (False, 100,  200,  800, 'cifar100_wrn_200',      'emp,emp-l'),
    'exp95':  (False, 100,  500, 2000, 'cifar100_wrn_500',      'emp-l'),
    'exp291': (False, 100,  200,  800, 'cifar100_wrn-a_200',    'emp,emp-l'),
    'exp295': (False, 100,  500, 2000, 'cifar100_wrn-a_500',    'emp-l'),
    'exp395': (False, 1000, 200, 1000, 'cifar100_rn18_200',     'emp-l'),
    'exp396': (False, 1000, 500, 2000, 'cifar100_rn18_500',     'emp-l'),
}


TO_PLOT = {
    'approx': {
        'random': ('Random', 'black', 'x:'),
        'ntkgp-frgp-ev': ('EV-GP', 'blue', 'o-'),
        'ntkgp-sgp-ev': ('EV-sGP', 'red', 's--'),
        'ntkgp-sgpla-ev': ('EV-sGP+LA', 'orange', 'p--'),
        'ntkgp-frgp-ev-emp': ('EV-GP-Emp', 'green', 'v-'),
    },
                
    'crits': {
        'random': ('Random', 'black', 'x-'),
        'ntkgp-frgp-ev': ('EV-GP', 'blue', 'o-'),
        'ntkgp-frgp-mi': ('MI-GP', 'purple', 's-'),
        'ntkgp-frgp-90v': ('90V-GP', 'orange', 'p-'),
        'ntkgp-frgp-mv': ('100V-GP', 'red', 'v-'),
    },
    
    'crits-no100': {
        'random': ('Random', 'black', 'x-'),
        'ntkgp-frgp-ev': ('EV-GP', 'blue', 'o-'),
        'ntkgp-frgp-mi': ('MI-GP', 'purple', 's-'),
        'ntkgp-frgp-90v': ('90V-GP', 'orange', 'p-'),
    },
                
    'regr': {
        'random': ('Random', 'black', 'x-'),
        'ntkgp-frgp-ev': ('EV-GP (ours)', 'blue', 'o-'),
        'kmeans-pp': ('K-Means++','orange', 'D-'),
    },
    
    'regr-bb': {
        'random': ('Random', 'black', 'x-'),
        'ntkgp-frgp-ev': ('EV-GP (ours)', 'blue', 'o-'),
        'kmeans-pp': ('K-Means++','orange', 'D-'),
        'maxent': ('Max-Ent','cyan', 'p-'),
    },
                
    'theory': {
        'random': ('Random', 'black', 'x-'),
        'ntkgp-frgp-ev': ('EV-GP (ours)', 'blue', 'o-'),
        'kmeans-pp': ('K-Means++','orange', 'D-'),
        'badge': ('BADGE', 'red', 'v-'),
        'mlmoc-batched-randinit': ('MLMOC', 'green', 's-'),
    },
    
    'theory-bb': {
        'random': ('Random', 'black', 'x-'),
        'ntkgp-frgp-ev': ('EV-GP (ours)', 'blue', 'o-'),
        'kmeans-pp': ('K-Means++','orange', 'D-'),
        'badge': ('BADGE', 'red', 'v-'),
        'mlmoc-batched-randinit': ('MLMOC', 'green', 's-'),
        'batchbald': ('BatchBALD', 'pink', '^-'),
        'maxent': ('Max-Ent','cyan', 'p-'),
    },
                
    'emp': {
        'random': ('Random', 'black', 'x-'),
        'ntkgp-frgp-ev-emp': ('EV-GP-Emp (ours)', 'blue', 'o-'),
        'kmeans-pp': ('K-Means++','orange', 'D-'),
        'badge': ('BADGE', 'red', 'v-'),
        'mlmoc-batched-randinit': ('MLMOC', 'green', 's-'),
    },
    
    'emp-bb': {
        'random': ('Random', 'black', 'x-'),
        'ntkgp-frgp-ev-emp': ('EV-GP-Emp (ours)', 'blue', 'o-'),
        'kmeans-pp': ('K-Means++','orange', 'D-'),
        'badge': ('BADGE', 'red', 'v-'),
        'mlmoc-batched-randinit': ('MLMOC', 'green', 's-'),
        'batchbald': ('BatchBALD', 'pink', '^-'),
    },
    
    'bbt': {
        'ntkgp-frgp-ev': ('EV-GP (ours)', 'blue', 'o-'),
        'batchbald': ('BatchBALD', 'pink', '^-'),
    },
    
    'bbe': {
        'ntkgp-frgp-ev-emp': ('EV-GP-Emp (ours)', 'blue', 'o-'),
        'batchbald': ('BatchBALD', 'pink', '^-'),
    },
                
    'emp-noinit': {
        'random': ('Random', 'black', 'x-'),
        'ntkgp-frgp-ev-emp': ('EV-GP-Emp (ours)', 'blue', 'o-'),
        'kmeans-pp': ('K-Means++','orange', 'D-'),
        'badge': ('BADGE', 'red', 'v-'),
        'mlmoc-batched': ('MLMOC', 'green', 's-'),
    },
                
    'emp-l': {
        'random': ('Random', 'black', 'x-'),
        'ntkgp-frgp-ev-emp': ('EV-GP-Emp (ours)', 'blue', 'o-'),
        'ntkgp-frgp-ev-emp-small': ('EV-GP-Emp (ours)', 'blue', 'o-'),
        'kmeans-pp': ('K-Means++','orange', 'D-'),
        'badge': ('BADGE', 'red', 'v-'),
        'mlmoc-batched-randinit-l': ('MLMOC', 'green', 's-'),
    },
                
    'randinit': {
        'random': ('Random', 'black', 'x--'),
        'mlmoc-batched': ('MLMOC', 'purple', 's-'),
        'mlmoc-batched-randinit': ('MLMOC (Rand. Init.)', 'cyan', 'p-'),
        'badge': ('BADGE', 'red', 'v-'),
        'badge-randinit': ('BADGE (Rand. Init.)', 'orange', '^-'),
    },
                
    'ms': {
        'random': ('Random', 'black', 'x--'),
        'ntkgp-frgp-ev': ('EV-GP', 'blue', 'o-'),
        'ntkgp-ensemble-ev': ('EV-GP + Model Sel.', 'red', 's-'),
    },
}


for k in TO_PLOT.keys():
    
    fig, ax = plt.subplots()
    
    for alg in TO_PLOT[k]:
        label, colour, ls = TO_PLOT[k][alg]
        zorder = 2.5 if ('ntkgp' in alg) else 2
        ax.errorbar(
            [0, 1],
            [0, 1],
            fmt=ls,
            color=colour,
            ms=10,
            label=label,
            alpha=0.6,
            capsize=3,
            zorder=zorder
        )
        
    n_col = (len(TO_PLOT[k]) + 1) // 2 if len(TO_PLOT[k]) > 3 else len(TO_PLOT[k])
    factor = 1.5 if len(TO_PLOT[k]) > 3 else 1
    
    ax.legend().set_zorder(4.)
    fig_leg = plt.figure(figsize=(n_col*2, factor))
    ax_leg = fig_leg.add_subplot(111)
    # add the legend from the previous axes
    ax_leg.legend(*ax.get_legend_handles_labels(), loc='center', ncol=n_col)
    # hide the axes frame and the x/y labels
    ax_leg.axis('off')
    fig_leg.tight_layout()
    fig_leg.savefig(f'./graphs/legends/legend_{k}.png', bbox_inches = 'tight', pad_inches = 0)
    plt.close(fig_leg)
    plt.close(fig)


def entropy(d):
    nans = np.isnan(d['test_preds']).any(axis=(1, 2))
    preds = d['test_preds'][~nans, :]
    logits = np.array([(np.argmax(preds, axis=-1) == i).sum(axis=0) for i in range(preds.shape[-1])]).T / preds.shape[0]
    ent = np.sum(np.nan_to_num(- logits * np.log(logits, where=(logits > 0))), axis=1)
    return ent


def filter_nan(arr):
    return arr[~np.isnan(arr).any(axis=(1, 2))]


for exp_code_int in exp_cases:
    
    exp_code = f'exp{exp_code_int}'
    is_regr, epoch_use, sel_batch_sz, sel_budget, exp_name, graph_cases = CASES[exp_code]
    
    subfolders = ['training']
    
    base_folder = f'./results_main/{exp_code}'
    
    crits_to_load = {k for g in graph_cases.split(',') for k in TO_PLOT[g].keys()}
        
    for subfolder in subfolders:
        
        print(f'############# {exp_code} {subfolder} #############')
        training_data = dict()
    
        try:
            
            for case_name in sorted(list(os.listdir(base_folder)))[::-1]:
                
                folder = f'{base_folder}/{case_name}'
                
                if not os.path.isdir(folder):
                    continue
                
                alg = case_name.split('_')[1]
                if alg not in crits_to_load:
                    continue
            
                if os.path.isdir(folder):
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
                        
                        if (sz % sel_batch_sz != 0) or (sz > sel_budget):
                            continue
                        
                        ck = (it, sz, ep)
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
                                
                    alg = case_name.split('_')[1]
                    if alg not in training_data:
                        training_data[alg] = []
                    training_data[alg].append(data)

            iters = set()
            sizes = set()
            epochs = set()

            for alg in training_data.keys():
                for d in training_data[alg]:
                    for it, sz, ep in d.keys():
                        iters.add(it)
                        sizes.add(sz)
                        epochs.add(ep)

            iters = sorted(list(iters))
            sizes = sorted(list(sizes))
            epochs = sorted(list(epochs))
            
            print('iters =', iters)
            print('sizes =', sizes)
            print('epoch =', epochs)

            if subfolder == 'training':
                true_label = lambda d: d['test_labels'].squeeze(-1)
            else:
                true_label = lambda d: np.argmax(d['test_labels'], axis=2)

            processed_data = dict()

            for alg in training_data.keys():
                proc_d = []
                for d in training_data[alg]:
                    tot = 0
                    try:
                        proc_d_dict = dict()
                        max_sz = max(sizes)
                        for sz in sizes:
                            sz_dict = dict()
                            for f in fnames:
                                sz_dict[f] = np.array([d[i, sz, epochs[-1]][f] for i in iters])
                                
                            if is_regr:
                                sz_dict['test_predvar'] = np.var(filter_nan(sz_dict['test_preds']), axis=(1, 2))
                            else:
                                sz_dict['test_ent'] = entropy(sz_dict)
                                sz_dict['test_acc'] = (np.argmax(filter_nan(sz_dict['test_preds']), axis=2) == 
                                                       true_label(sz_dict)[0][None, :])
                                if 'cifar100' in exp_name:
                                    sz_dict['test_top5'] = np.argsort(sz_dict['test_preds'], axis=2)[:, :, -5:]
                                    sz_dict['test_top5_acc'] = (true_label(sz_dict)[0][None, :, None] == sz_dict['test_top5']).any(axis=2)
                            proc_d_dict[sz] = sz_dict
                        proc_d.append(proc_d_dict)
                        tot += 1
                    except KeyError:
                        pass
                processed_data[alg] = proc_d
                
        except Exception as e:
            print(traceback.format_exc())
            continue

        for g in graph_cases.split(','):
            
            algs_list = TO_PLOT[g]
            
            graph_out = f'{GRAPH_ROOT_DIR}/{g}' 
            
            print(f'========== {graph_out} {exp_code} {subfolder} ==========')
            
            try:
                
                algs = [a for a in algs_list.keys() if a in training_data.keys()]

                def plot_data(indiv_fn, errbar=None, xlabel=None, ylabel=None, legend=True, fname=None, legend_path=None):
                    
                    fig, ax = plt.subplots()
                    for alg in algs:
                        ys, ys_std = [], []
                        for sz in sizes:
                            rec1 = [indiv_fn(d[sz]) for d in processed_data[alg]]
                            if len(rec1) == 0:
                                ys.append(np.nan)
                                ys_std.append(np.nan)
                                print(alg, sz, '- no records')
                                continue
                            assert not isinstance(rec1[0], np.ndarray), rec1[0]
                            rec = [x for x in rec1 if not np.isnan(x).any() and np.abs(x) < 100.]
                            if (len(rec1) > len(rec)) or (len(rec) < 5):
                                print(alg, sz, f'- records={len(rec1)}, usable={len(rec)}')
                            rec = rec[:5]  # limit to 5 trials
                            ys.append(np.mean(rec))
                            if callable(errbar):
                                rec_err1 = [errbar(d[sz]) for d in processed_data[alg]]
                                assert not isinstance(rec1[0], np.ndarray), rec1[0]
                                rec_err = [x for x in rec_err1 if not np.isnan(x).any() and np.abs(x) < 1e9]
                                ys_std.append(np.mean(rec_err))
                            else:
                                ys_std.append(np.std(rec))
                        
                        if not np.isnan(ys).all():
                            label, colour, ls = algs_list[alg]
                            zorder = 2.5 if ('ntkgp' in alg) else 2
                            ax.errorbar(
                                sizes,
                                ys,
                                yerr=ys_std if errbar else None,
                                fmt=ls,
                                color=colour,
                                ms=10,
                                label=label,
                                alpha=0.6,
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
                        ax.legend().set_zorder(4.)
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
                
                prefix = f'{graph_out}/{exp_name}/{subfolder}'

                plot_data(
                    indiv_fn=lambda d: np.nanmean(d['test_losses']), 
                    errbar=True,
                    xlabel='Selection points',
                    ylabel='Test MSE' if is_regr else 'Mean test loss',
                    fname=f'{prefix}/test-loss-mean',
                    legend_path=prefix
                )


                plot_data(
                    indiv_fn=lambda d: np.nanmean(d['test_losses'][d['test_losses'] < 1e6]), 
                    errbar=lambda d: np.nanstd(np.nanmean(d['test_losses'][(d['test_losses'] < 1e6).all(axis=(1, 2))], axis=(1, 2))),
                    xlabel='Selection points',
                    ylabel='Test MSE' if is_regr else 'Mean test loss',
                    fname=f'{prefix}/test-loss-mean-w-stderr'
                )


                plot_data(
                    indiv_fn=lambda d: np.nanstd(np.nanmean(d['test_losses'][(d['test_losses'] < 1e6).all(axis=(1, 2))], axis=(1, 2))),
                    errbar=True,
                    xlabel='Selection points',
                    ylabel='Test MSE std. dev.' if is_regr else 'Std. dev. test loss',
                    fname=f'{prefix}/test-loss-sd',
                )
                
                
                if is_regr:
                    plot_data(
                        indiv_fn=lambda d: np.mean(d['test_predvar']),
                        errbar=True,
                        xlabel='Selection points',
                        ylabel='Mean output variance',
                        fname=f'{prefix}/pred-var-mean',
                    )


                    plot_data(
                        indiv_fn=lambda d: np.percentile(d['test_predvar'], 90), 
                        errbar=True,
                        xlabel='Selection points',
                        ylabel='90th %ile output variance',
                        fname=f'{prefix}/pred-var-90p',
                    )


                    # plot_data(
                    #     indiv_fn=lambda d: np.percentile(d['test_predvar'], 99), 
                    #     errbar=True,
                    #     xlabel='Selection points',
                    #     ylabel='99th percentile output variance',
                    #     fname=f'{prefix}/pred-var-99p',
                    # )
                
                
                else:
                    
                    
                    plot_data(
                        indiv_fn=lambda d: np.mean(d['test_ent']),
                        errbar=True,
                        xlabel='Selection points',
                        ylabel='Mean output entropy',
                        fname=f'{prefix}/pred-ent-mean',
                    )


                    plot_data(
                        indiv_fn=lambda d: np.percentile(d['test_ent'], 90), 
                        errbar=True,
                        xlabel='Selection points',
                        ylabel='90th %ile output entropy',
                        fname=f'{prefix}/pred-ent-90p',
                    )


                    plot_data(
                        indiv_fn=lambda d: np.percentile(d['test_ent'], 99), 
                        errbar=True,
                        xlabel='Selection points',
                        ylabel='99th %ile output entropy',
                        fname=f'{prefix}/pred-ent-99p',
                    )


                    plot_data(
                        indiv_fn=lambda d: np.mean(d['test_acc']),
                        errbar=True,
                        xlabel='Selection points',
                        ylabel='Mean test accuracy',
                        fname=f'{prefix}/acc-mean',
                    )


                    plot_data(
                        lambda d: np.std(np.nanmean(d['test_acc'], axis=1)),
                        errbar=False,
                        xlabel='Selection points',
                        ylabel='Std. dev. test accuracy',
                        fname=f'{prefix}/acc-sd',
                    )
                    
                    
                    plot_data(
                        indiv_fn=lambda d: np.nanmean(d['test_acc']), 
                        errbar=lambda d: np.nanstd(np.nanmean(d['test_acc'], axis=1)),
                        xlabel='Selection points',
                        ylabel='Mean test accuracy',
                        fname=f'{prefix}/acc-mean-w-stderr'
                    )
                    
                    if 'cifar100' in exp_name:
                        
                        plot_data(
                            indiv_fn=lambda d: np.mean(d['test_top5_acc']),
                            errbar=True,
                            xlabel='Selection points',
                            ylabel='Mean test top-5 accuracy',
                            fname=f'{prefix}/acc-top5-mean',
                        )
                        
                        
                        plot_data(
                            indiv_fn=lambda d: np.nanmean(d['test_top5_acc']), 
                            errbar=lambda d: np.nanstd(np.nanmean(d['test_top5_acc'], axis=1)),
                            xlabel='Selection points',
                            ylabel='Mean test top-5 accuracy',
                            fname=f'{prefix}/acc-top5-mean-w-stderr'
                        )
                    
                    
            except Exception as e:
                print(traceback.format_exc())
                pass
            
    print('--------------- DONE ---------------\n')
