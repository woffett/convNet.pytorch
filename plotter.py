from smallfry.plotter import gather_results,clean_results,plot_driver
from matplotlib import pyplot as plt

def plot_distillation_tune_hp_results():
    path_regex = '/proj/distill/results/2020-01-22-tuneDistillHP/*/*results.json'
    all_results = clean_results(gather_results(path_regex))
    lrs = [0.01, 0.03, 0.1, 0.3, 1] # 5
    # alphas = [0, 0.1, 0.5, 0.9, 1] # 5
    kldiv_temps = [0.1, 1, 10, 100] # 4
    # mse_temps = [0] # 1
    # losses = ['mse','kldiv'] # 2

    # PLOT KLDIV RESULTS
    for temp in kldiv_temps:
        info_per_line = {}
        for lr in lrs:
            # for each learning rate, have a different line (we are plotting prec@1 vs. alpha)
            info_per_line['lr={}'.format(lr)] = {
                'lr':[lr],
                'distill_loss':['kldiv'],
                'temperature':[temp]
            }
        plot_driver(all_results, None, info_per_line, 'alpha', 'best_prec1')
        plt.savefig('/proj/distill/results/2020-01-22-tuneDistillHP/figures/kldiv_temp,{}_tune_hp_results.pdf'.format(temp))

    # PLOT MSE RESULTS
    info_per_line = {}
    for lr in lrs:
            # for each learning rate, have a different line (we are plotting prec@1 vs. alpha)
        info_per_line['lr={}'.format(lr)] = {
            'lr':[lr],
            'distill_loss':['mse']
        }
    plot_driver(all_results, None, info_per_line, 'alpha', 'best_prec1')
    plt.savefig('/proj/distill/results/2020-01-22-tuneDistillHP/figures/mse_tune_hp_results.pdf')