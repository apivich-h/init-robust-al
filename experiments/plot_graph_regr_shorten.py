import os
import pickle as pkl

import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import matplotlib.cm as cm
from scipy import stats

GRAPH_ROOT_DIR = './graphs/small'

FIG_DIM = 5

plt.rcParams['figure.figsize'] = (FIG_DIM, FIG_DIM)
plt.rcParams['figure.dpi'] = 300

plt.rcParams.update({
    'font.size': 20,
    'text.usetex': False,
    'font.family': 'Helvetica'
})
title_fontsize = 22


plt.rcParams['figure.dpi'] = 300

PLOT_STUFF = True

ROOT_FOLDER = './results_regr'


# CASES = sorted([int(x.split('_')[1]) for x in os.listdir(ROOT_FOLDER) 
#                 if (x.startswith('TESTCASE') and os.path.isdir(f'{ROOT_FOLDER}/{x}'))])
CASES = [1, 3, 9, 11, 15, 27, 35, 37, 41, 43]
print(f'cases = {CASES}')

crits = {
    'ev': 'EV',
    'mi': 'MI',
    '90v': '90V',
    'mv': '100V',
}
gps = {
    'frgp': 'GP',
    'sgp': 'sGP',
    'sgpla': 'sGP+LA'
}

future_vals = {
    'mean_loss': lambda a, i: np.mean(a[f'r{i}_test_mse_loss'] if len(a[f'r{i}_test_mse_loss'].shape) == 1 
                                      else a[f'r{i}_test_mse_loss'][:, -1]),
    'pred_var': lambda a, i: np.mean(np.var(a[f'r{i}_test_preds'], axis=0), axis=1),
    'ev_crit': lambda a, i: np.mean(a[f'r{i}_frgp_criterion_ev_tr']),
    'mi_crit': lambda a, i: np.mean(a[f'r{i}_frgp_criterion_mi_te']),
}


extra_vals_to_track = []


for case_idx in tqdm.tqdm(CASES):

    prefix = f'{ROOT_FOLDER}/TESTCASE_{case_idx}'

    # GETTING ALL THE DATA #

    try:

        arrs = dict()

        for root, subdirs, files in os.walk(prefix):
            root_sub = root[len(prefix) + 1:]
            sub_arrs = dict()
            for f in files:
                if f.endswith('.npy'):
                    sub_arrs[f.split('.')[0]] = np.load(f'{root}/{f}')
            arrs[root_sub] = sub_arrs

        params = dict()
        with open(f'{prefix}/record', 'r') as f:
            for ln in f.readlines():
                y = [x.strip() for x in ln.split('=')]
                if len(y) == 2:
                    try:
                        params[y[0]] = eval(y[1])
                    except Exception as e:
                        params[y[0]] = y[1]

    except Exception as e:

        print(case_idx, str(e))
        continue

    # SAVE EXTRA DATA FOR PLOT #

    n_ = params['next_batch_selection_sz']
    for k in sorted(arrs.keys()):
        if k.startswith('batch'):
            a_ = arrs[k]
            for i in range(0, n_ + 1):
                # remove nans
                a_[f'r{i}_test_preds'] = a_[f'r{i}_test_preds'][~np.isnan(a_[f'r{i}_test_preds']).any(axis=(1, 2))]
                a_[f'r{i}_test_mse_loss'] = a_[f'r{i}_test_mse_loss'][~np.isnan(a_[f'r{i}_test_mse_loss']).any(axis=1)]
                # # remove big numbers
                a_[f'r{i}_test_preds'] = a_[f'r{i}_test_preds'][
                    (np.abs(a_[f'r{i}_test_preds']) < 1e3).all(axis=(1, 2))]
                a_[f'r{i}_test_mse_loss'] = a_[f'r{i}_test_mse_loss'][
                    (np.abs(a_[f'r{i}_test_mse_loss']) < 1e3).all(axis=1)]

                a_[f'r{i}_test_pred_var'] = np.mean(np.var(a_[f'r{i}_test_preds'], axis=0), axis=1)
                a_[f'r{i}_test_pred_var_mean'] = np.mean(a_[f'r{i}_test_pred_var'])
                a_[f'r{i}_test_mse_loss_final'] = (a_[f'r{i}_test_mse_loss'] 
                                                    if len(a_[f'r{i}_test_mse_loss'].shape) == 1 
                                                    else a_[f'r{i}_test_mse_loss'][:, -1])
                a_[f'r{i}_test_mse_loss_final_mean'] = np.mean(a_[f'r{i}_test_mse_loss_final'])
                a_[f'r{i}_best_init'] = np.argmin(a_[f'r{i}_test_mse_loss_final'])
                a_[f'r{i}_mean_pred'] = np.mean(a_[f'r{i}_test_preds'], axis=0)
                a_[f'r{i}_mean_pred_diff'] = (a_[f'r{i}_mean_pred'] - a_[f'r{i}_test_preds'][a_[f'r{i}_best_init']])**2
                a_[f'r{i}_best_pred_diff_mean'] = np.mean(a_[f'r{i}_mean_pred_diff'])
                for q in [50, 90, 100]:
                    a_[f'r{i}_test_preds_var_q{q}'] = np.percentile(a_[f'r{i}_test_pred_var'], q=q)
                    a_[f'r{i}_best_pred_diff_q{q}'] = np.percentile(a_[f'r{i}_mean_pred_diff'], q=q)

    # # GET ACCURACY INFO FOR CLASSIFICATION #

    if params['prob_type'] == 'classification':
        for k in sorted(arrs.keys()):
            if k.startswith('batch'):
                a_ = arrs[k]
                for i in range(params['next_batch_selection_sz'] + 1):
                    if f'r{i}_test_preds' in a_:
                        a_[f'r{i}_test_accuracy'] = np.mean(
                            np.argmax(a_[f'r{i}_test_preds'], axis=2) == arrs['data']['actual_ys_test'].flatten()[None, :],
                            axis=1)
    elif params['prob_type'] == 'binary_classification':
        for k in sorted(arrs.keys()):
            if k.startswith('batch'):
                a_ = arrs[k]
                for i in range(params['next_batch_selection_sz'] + 1):
                    if f'r{i}_test_preds' in a_:
                        a_[f'r{i}_test_accuracy'] = np.mean(
                            (np.sign(a_[f'r{i}_test_preds']) == 1.) == arrs['data']['actual_ys_test'],
                            axis=(1, 2))

    random_dict = dict()
    for i in range(1, n_ + 1):
        for v in ['test_pred_var_mean', 'test_mse_loss_final_mean', 'test_accuracy',
                  'test_preds_var_q50', 'test_preds_var_q90', 'test_preds_var_q100',
                  'best_pred_diff_q50', 'best_pred_diff_q90', 'best_pred_diff_q100']:
            random_dict[f'r{i}_{v}'] = np.mean([arrs[k][f'r{i}_{v}'] for k in arrs.keys()
                                                if 'batch_random' in k and f'r{i}_{v}' in arrs[k].keys()])
    arrs['batch_rand_all'] = random_dict

    # DATA FOR FUTURE PLOTS #

    extra_vals_data = dict()
    n_ = params['next_batch_selection_sz']
    for k in sorted(arrs.keys()):
        if k.startswith('batch'):
            do_plot = False
            try:
                a_ = arrs[k]
                rec = dict()
                for track_val in future_vals.keys():
                    rec[track_val] = np.array([future_vals[track_val](a_, i) for i in range(1, n_)])
                extra_vals_data[k] = rec
            except:
                pass
    extra_vals_data['params'] = params
    extra_vals_to_track.append(extra_vals_data)

    # SET UP GRAPHING FUNCTIONS #

    suffix = f'{params["dset_name"]}_i{params["init_select_sz"]:03d}_s{params["next_batch_selection_sz"]:02d}'

    # dataset_name = {
    #     'mock_reg_high_noise': 'Random Data (Noisy)',
    #     'mock_reg_low_noise': 'Random Data (Low noise)',
    #     'mock_reg_unbal_high_noise': 'Random Data (Noisy, Mismatched)',
    #     'mock_reg_unbal_low_noise': 'Random Data (Low noise, Mismatched)',

    #     'boston': 'Boston',
    #     'breast_cancer_wisconsin': 'Breast Cancer',
    #     'breast_cancer_wisconsin_mismatched': 'Breast Cancer (Mismatched)',
    #     'concrete': 'Concrete',
    #     'energy': 'Energy',
    #     'image_segmentation': 'Image Segmentation',
    #     'image_segmentation_mismatched': 'Image Segmentation (Mismatched)',
    #     'kin8nm': 'Robot Kinematics',
    #     'naval': 'Naval',
    #     'optdigits': 'Handwritten Digits',
    #     'power': 'Power',
    #     'protein': 'Protein',
    #     'spambase': 'Spambase',
    #     'spambase_mismatched': 'Spambase (Mismatched)',
    #     'wine': 'Wine',
    #     'yeast': 'Yeast',
    #     'yacht': 'Yacht'
    # }[params["dset_name"]]


    def plot_val(xs_fn, ys_fn, xs_label=None, ys_label=None, title=None, fname_prefix=None,
                 track_crit=True, crit=None, gp=None, multicolour=True, get_score=False, alpha=None, add_legend=True,
                 extra_plot_slope=None, rnds=None, xticks_rounds=False, ax=None, ms=None, marker='.'):

        n = params['next_batch_selection_sz']

        if gp is not None:
            name = {
                f'batch_ev-{gp}': ('o-', 'blue'),
                f'batch_mi-{gp}': ('s--', 'purple'),
                f'batch_90v-{gp}': ('p-', 'cyan'),
                f'batch_mv-{gp}': ('^-', 'red'),
                'batch_rand_all': ('x:', 'black')
            }
            if crit is not None:
                crit = {crit} if isinstance(crit, str) else crit
                for k in list(name.keys()):
                    if 'rand' in k:
                        continue
                    _has = False
                    for c in crit:
                        if c in k:
                            _has = True
                            break
                    if not _has:
                        name.pop(k)
        elif crit is not None:
            name = {
                f'batch_{crit}-frgp': ('o-', 'blue'),
                f'batch_{crit}-sgp': ('o--', 'green'),
                f'batch_{crit}-sgpla': ('o-.', 'red'),
                'batch_rand_all': ('x:', 'black'),
            }
        else:
            name = dict()

        if rnds is None:
            rnds = range(1, n + 1)
            count = n
        else:
            count = len(list(rnds))

        xs_all = []
        ys_all = []

        if ax is None:
            fig = plt.figure(figsize=(FIG_DIM, FIG_DIM))
            ax = fig.add_subplot()
        else:
            fig = None

        has_labels = False

        arrs_keys = [k for k in name.keys() if k in arrs.keys()]
        arrs_keys.extend([k for k in arrs.keys() if k.startswith('batch') and 'random' in k])

        for k in arrs_keys:
            if k.startswith('batch'):
                do_plot = False
                try:
                    a = arrs[k]
                    if track_crit:
                        alpha_ = 1. if (k in name.keys()) else (alpha if alpha is not None else 0.1)
                        if k in name.keys():
                            has_labels = True
                            do_plot = True
                            if k == 'batch_rand_all':
                                label = 'Random'
                                xs = np.concatenate([xs_fn(a, i) for i in rnds], axis=None)
                                ys = np.concatenate([ys_fn(a, i) for i in rnds], axis=None)
                                ys_err = np.array([np.std([ys_fn(arrs[k], i) for k in arrs.keys() if 'random' in k]) for i in rnds])
                                ax.errorbar(xs, ys, ys_err, fmt=name[k][0], color=name[k][1], alpha=0.6, label=label, capsize=3)
                            else:
                                crit, approx = k.split('_')[1].split('-')
                                crit = crits[crit]
                                approx = gps[approx]
                                if gp == 'frgp':
                                    label = crit
                                else:
                                    label = f"{crit}-{approx}"
                                xs = np.concatenate([xs_fn(a, i) for i in rnds], axis=None)
                                ys = np.concatenate([ys_fn(a, i) for i in rnds], axis=None)
                                ax.plot(xs, ys, name[k][0], color=name[k][1], alpha=0.6, label=label, zorder=2.2, ms=10)
                    else:
                        alpha_ = alpha if alpha is not None else 0.1
                    if ('random' in k) and get_score:
                        xs = np.concatenate([xs_fn(a, i) for i in rnds], axis=None)
                        ys = np.concatenate([ys_fn(a, i) for i in rnds], axis=None)
                        if isinstance(multicolour, str):
                            col = multicolour
                        elif isinstance(multicolour, bool) and multicolour and (len(xs) == len(ys) == count):
                            col = cm.rainbow(np.linspace(0, 1, count)) 
                        else:
                            col = 'gray'
                        m = marker if ('random' in k) else name[k][0][0]
                        ax.scatter(xs, ys, s=ms, marker=m, color=col, alpha=alpha_)
                        xs_all.extend(xs)
                        ys_all.extend(ys)
                except KeyError as e:
                    if k != 'batch_rand_all':
                        print('KeyError', k, e)
                except Exception as e:
                    print(k, e)

        if get_score and len(xs_all) > 0:
            try:
                sp = stats.spearmanr(xs_all, ys_all).correlation
                pe = stats.pearsonr(xs_all, ys_all)[0]
                if not (np.isnan(sp) or np.isnan(pe)):
                    txt = f'SP={sp:.3f}\nPE={pe:.2f}'
                    # yal = 0.05 if (sp > 0.) else 0.9
                    # ax.text(0.75, yal, txt, transform=ax.transAxes, ha='left', va='bottom',
                    #         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                    
                    loc = 'lower right' if (sp > 0.) else 'upper right'
                    text_box = AnchoredText(txt, frameon=True, loc=loc, pad=0.1)
                    plt.setp(text_box.patch, boxstyle='round', facecolor='wheat', alpha=0.5)
                    ax.add_artist(text_box)
            except:
                pass

        if (extra_plot_slope is not None) and len(xs_all) > 0:
            xmin = min(0., min(xs_all))
            xmax = max(xs_all)
            for ii, m in enumerate(extra_plot_slope):
                ax.plot([xmin, xmax], [m * xmin, m * xmax], color=f'C{ii + 1}', alpha=0.7,
                        label="y = x" if m == 1. else f"y = {m}x")
                has_labels = True

        if add_legend and has_labels:
            ax.legend()

        ax.set_xlabel(xs_label)
        ax.set_ylabel(ys_label)
        if xticks_rounds:
            ax.set_xticks(range(0, n+1, 5))
        if title is not None:
            ax.set_title(title, fontsize=title_fontsize)

        if fig is not None:
            if fname_prefix is not None and len(xs_all) > 0:
                os.makedirs(f'{GRAPH_ROOT_DIR}/{fname_prefix}', exist_ok=True)
                fig.savefig(f'{GRAPH_ROOT_DIR}/{fname_prefix}/{suffix}.jpg', transparent=False)

            plt.close(fig)
            return fig
        else:
            return None


    def plot_val_multirow(xs_fns, ys_fn, xs_labels=None, ys_label=None, sharey=True, 
                          suptitle=None, titles=None, fname_prefix=None,
                          track_crit_arr=False, crit_arr=None, gp_arr=None, multicolour=True, 
                          get_score_arr=None, alpha=None,
                          xticks_rounds_arr=None, ms=None, marker=None, add_legend_lst=None, extra_plot_slopes=None):
        n_columns = len(xs_fns)
        fig = plt.figure(figsize=(FIG_DIM * n_columns, FIG_DIM))
        axs = fig.subplots(nrows=1, ncols=n_columns, sharey=sharey)
        for i in range(n_columns):
            plot_val(
                xs_fn=xs_fns[i],
                ys_fn=ys_fn if sharey else ys_fn[i],
                xs_label=xs_labels[i] if xs_labels else None,
                ys_label=None if sharey else ys_label[i],
                title=titles[i] if titles else None,
                fname_prefix=None,
                track_crit=track_crit_arr[i] if track_crit_arr else False,
                crit=crit_arr[i] if crits else None,
                gp=gp_arr[i] if gps else None,
                multicolour=multicolour,
                get_score=get_score_arr[i] if get_score_arr else False,
                alpha=alpha[i] if alpha is not None else None,
                xticks_rounds=xticks_rounds_arr[i] if xticks_rounds_arr else None,
                ax=axs[i] if n_columns > 1 else axs,
                ms=ms,
                marker=marker[i] if marker is not None else None,
                add_legend=add_legend_lst[i] if add_legend_lst else False,
                extra_plot_slope=extra_plot_slopes[i] if extra_plot_slopes else None
            )

        # if params["init_select_sz"] > 0:
        #     fig.suptitle(f'{dataset_name} - {params["init_select_sz"]} init. pts.' if suptitle is None else suptitle,
        #                 fontsize=title_fontsize)
        # else:
        #     fig.suptitle(f'{dataset_name}' if suptitle is None else suptitle, fontsize=title_fontsize)
            
        if sharey:
            if n_columns > 1:
                axs[0].set_ylabel(ys_label)
            else:
                axs.set_ylabel(ys_label)

        fig.tight_layout()

        if fname_prefix is not None:
            os.makedirs(f'{GRAPH_ROOT_DIR}/{fname_prefix}', exist_ok=True)
            fig.savefig(f'{GRAPH_ROOT_DIR}/{fname_prefix}/{suffix}.jpg', transparent=False)

        plt.close(fig)

    if PLOT_STUFF:
        
        plot_val_multirow(
            fname_prefix='mse_mean-round-afrgp',
            xs_fns=[
                lambda a, i: i
            ],
            ys_fn=lambda a, i: a[f'r{i}_test_mse_loss_final_mean'],
            xs_labels=[
                'Selection points',
            ],
            ys_label='Test MSE',
            track_crit_arr=[True, True],
            gp_arr=['frgp'],
            crit_arr=[{'ev'}],
            multicolour=False,
            get_score_arr=[False],
            add_legend_lst=[True],
            alpha=[0.2],
            xticks_rounds_arr=[True],
            marker='x'
        )

        plot_val_multirow(
            fname_prefix='mse_mean-criterion_round-afrgp',
            xs_fns=[
                lambda a, i: np.mean(a[f'r{i}_frgp_criterion_ev_tr']),
                lambda a, i: i
            ],
            ys_fn=lambda a, i: a[f'r{i}_test_mse_loss_final_mean'],
            xs_labels=[
                f'EV criterion',
                'Selection points',
            ],
            ys_label='Test MSE',
            track_crit_arr=[True, True],
            gp_arr=['frgp', 'frgp'],
            crit_arr=['ev', {'ev'}],
            multicolour=False,
            get_score_arr=[True, False],
            add_legend_lst=[False, True],
            alpha=[0.5, 0.2],
            xticks_rounds_arr=[False, True],
            marker='.x'
        )
        
        plot_val_multirow(
            fname_prefix='mse_mean-round-afrgp_crit',
            xs_fns=[
                lambda a, i: i
            ],
            ys_fn=lambda a, i: a[f'r{i}_test_mse_loss_final_mean'],
            xs_labels=[
                'Selection points',
            ],
            ys_label='Test MSE',
            track_crit_arr=[True],
            gp_arr=['frgp'],
            crit_arr=[{'ev', '90v', 'mv', 'mi'}],
            multicolour=False,
            get_score_arr=[False],
            add_legend_lst=[True],
            alpha=[0.2],
            xticks_rounds_arr=[True],
            marker='x'
        )
        
        plot_val_multirow(
            fname_prefix='mse_mean-criterion_round-afrgp_crit',
            xs_fns=[
                lambda a, i: np.mean(a[f'r{i}_frgp_criterion_90v_te']),
                lambda a, i: np.mean(a[f'r{i}_frgp_criterion_mv_te']),
                lambda a, i: np.mean(a[f'r{i}_frgp_criterion_mi_te']),
                lambda a, i: i
            ],
            ys_fn=lambda a, i: a[f'r{i}_test_mse_loss_final_mean'],
            xs_labels=[
                f'90V criterion',
                f'100V criterion',
                f'MI criterion',
                'Selection points',
            ],
            ys_label='Test MSE',
            track_crit_arr=[True, True, True, True],
            gp_arr=['frgp', 'frgp', 'frgp', 'frgp'],
            crit_arr=['90v', 'mv', 'mi', {'ev', '90v', 'mv', 'mi'}],
            multicolour=False,
            get_score_arr=[True, True, True, False],
            add_legend_lst=[False, False, False, True],
            alpha=[0.5, 0.5, 0.5, 0.2],
            xticks_rounds_arr=[False, False, False, True],
            marker='...x'
        )
        
        plot_val_multirow(
            fname_prefix='mse_mean-predvar_mean-afrgp',
            xs_fns=[
                lambda a, i: a[f'r{i}_test_pred_var_mean']
            ],
            ys_fn=lambda a, i: a[f'r{i}_test_mse_loss_final_mean'],
            xs_labels=[
                'Mean output variance',
            ],
            ys_label='Test MSE',
            track_crit_arr=[False],
            gp_arr=[None],
            crit_arr=[None],
            multicolour=False,
            get_score_arr=[True],
            add_legend_lst=[True],
            alpha=[0.5],
            xticks_rounds_arr=[False],
            marker='.'
        )

        plot_val_multirow(
            fname_prefix=f'predvar_mean-round-cev',
            xs_fns=[
                lambda a, i: i
            ],
            ys_fn=lambda a, i: a[f'r{i}_test_pred_var_mean'],
            xs_labels=[
                'Selection points',
            ],
            ys_label='Mean output variance',
            track_crit_arr=[True],
            gp_arr=[None],
            crit_arr=['ev'],
            multicolour=False,
            get_score_arr=[False],
            add_legend_lst=[True],
            alpha=[0.2],
            xticks_rounds_arr=[True],
            marker='x'
        )

        for (q, ys_label) in [
            # (50, 'Median'),
            (90, '90th %ile'),
            # (100, 'Maximum')
        ]:
            plot_val_multirow(
                fname_prefix=f'predvar_q{q}-criterion_round-afrgp',
                xs_fns=[
                    lambda a, i: np.mean(a[f'r{i}_frgp_criterion_ev_te']),
                    lambda a, i: i
                ],
                ys_fn=lambda a, i: a[f'r{i}_test_preds_var_q{q}'],
                xs_labels=[
                    f'EV criterion',
                    'Selection points',
                ],
                ys_label=f'{ys_label} output variance',
                track_crit_arr=[True, True],
                gp_arr=['frgp', 'frgp'],
                crit_arr=['ev', {'ev'}],
                multicolour=False,
                get_score_arr=[True, False],
                add_legend_lst=[False, True],
                alpha=[0.5, 0.2],
                xticks_rounds_arr=[False, True],
            marker='.x'
            )
            
            plot_val_multirow(
                fname_prefix=f'predvar_q{q}-criterion_round-afrgp_crits',
                xs_fns=[
                    lambda a, i: np.mean(a[f'r{i}_frgp_criterion_90v_te']),
                    lambda a, i: np.mean(a[f'r{i}_frgp_criterion_mv_te']),
                    lambda a, i: np.mean(a[f'r{i}_frgp_criterion_mi_te']),
                    lambda a, i: i
                ],
                ys_fn=lambda a, i: a[f'r{i}_test_preds_var_q{q}'],
                xs_labels=[
                    f'90V criterion',
                    f'100V criterion',
                    f'MI criterion',
                    'Selection points',
                ],
                ys_label=f'{ys_label} output variance',
                track_crit_arr=[True, True, True, True],
                gp_arr=['frgp', 'frgp', 'frgp', 'frgp'],
                crit_arr=['90v', 'mv', 'mi', {'ev', '90v', 'mv', 'mi'}],
                multicolour=False,
                get_score_arr=[True, True, True, False],
                add_legend_lst=[False, False, False, True],
                alpha=[0.5, 0.5, 0.5, 0.2],
                xticks_rounds_arr=[False, False, False, True],
            marker='...x'
            )

            plot_val_multirow(
                fname_prefix=f'predvar_q{q}-criterion_round-cev',
                xs_fns=[
                    lambda a, i: np.mean(a[f'r{i}_frgp_criterion_ev_te']),
                    lambda a, i: i
                ],
                ys_fn=lambda a, i: a[f'r{i}_test_preds_var_q{q}'],
                xs_labels=[
                    f'EV criterion',
                    'Selection points',
                ],
                ys_label=f'{ys_label} output variance',
                track_crit_arr=[True, True],
                gp_arr=[None, None],
                crit_arr=['ev', 'ev'],
                multicolour=False,
                get_score_arr=[True, False],
                add_legend_lst=[False, True],
                alpha=[0.5, 0.2],
                xticks_rounds_arr=[False, True],
                marker='.x'
            )
            
            plot_val_multirow(
                fname_prefix=f'predvar_q{q}-round-cev',
                xs_fns=[
                    lambda a, i: i
                ],
                ys_fn=lambda a, i: a[f'r{i}_test_preds_var_q{q}'],
                xs_labels=[
                    'Selection points',
                ],
                ys_label=f'{ys_label} output variance',
                track_crit_arr=[True],
                gp_arr=[None],
                crit_arr=['ev'],
                multicolour=False,
                get_score_arr=[False],
                add_legend_lst=[True],
                alpha=[0.2],
                xticks_rounds_arr=[True],
                marker='x'
            )

        plot_val_multirow(
            fname_prefix=f'predvar-ntkgp_var-frgp-sgp',
            xs_fns=[
                lambda a, i: a[f'r{i}_frgp_cov_te'],
                lambda a, i: a[f'r{i}_sgp_cov_te'],
            ],
            ys_fn=lambda a, i: a[f'r{i}_test_pred_var'],
            xs_labels=[
                'NTKGP variance',
                'sNTKGP variance',
            ],
            ys_label='Empirical output variance',
            track_crit_arr=[False, False],
            gp_arr=[None, None],
            crit_arr=[None, None],
            multicolour='blue',
            get_score_arr=[True, True],
            alpha=[0.01, 0.01],
            xticks_rounds_arr=[False, False],
            ms=5.,
            marker='..'
        )
        
        plot_val_multirow(
            fname_prefix=f'predvar-ntkgp_var',
            xs_fns=[
                lambda a, i: a[f'r{i}_frgp_cov_te'],
            ],
            ys_fn=lambda a, i: a[f'r{i}_test_pred_var'],
            xs_labels=[
                'NTKGP variance',
            ],
            ys_label='Empirical output variance',
            track_crit_arr=[False],
            gp_arr=[None],
            crit_arr=[None],
            multicolour='blue',
            get_score_arr=[True],
            alpha=[0.01],
            xticks_rounds_arr=[False],
            ms=5.,
            marker='..'
        )

        if 'classification' in params['prob_type']:
            
            plot_val_multirow(
                fname_prefix=f'acc_mean-round-afrgp',
                xs_fns=[
                    lambda a, i: i
                ],
                ys_fn=lambda a, i: np.mean(a[f'r{i}_test_accuracy']),
                xs_labels=[
                    'Selection points',
                ],
                ys_label='Mean test accuracy',
                track_crit_arr=[True],
                gp_arr=['frgp'],
                crit_arr=[{'ev'}],
                multicolour=False,
                get_score_arr=[False],
                add_legend_lst=[True],
                alpha=[0.2],
                xticks_rounds_arr=[True],
                marker='x'
            )
            
            plot_val_multirow(
                fname_prefix=f'acc_mean-round-afrgp_crits',
                xs_fns=[
                    lambda a, i: i
                ],
                ys_fn=lambda a, i: np.mean(a[f'r{i}_test_accuracy']),
                xs_labels=[
                    'Selection points',
                ],
                ys_label='Mean test accuracy',
                track_crit_arr=[True],
                gp_arr=['frgp'],
                crit_arr=[{'ev', '90v', 'mv', 'mi'}],
                multicolour=False,
                get_score_arr=[False],
                add_legend_lst=[True],
                alpha=[0.2],
                xticks_rounds_arr=[True],
                marker='x'
            )

with open('extracted_data.pkl', 'wb+') as f:
    pkl.dump(extra_vals_to_track, f)

