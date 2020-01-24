from smallfry.plotter import gather_results,clean_results,plot_driver
from matplotlib import pyplot as plt

def plot_distillation_tune_hp_results():
    path_regex = '/proj/distill/results/2020-01-22-tuneDistillHP/*/*results.json'
    all_results = clean_results(gather_results(path_regex))
    var_info = ['distributed', [False]]
    ylims = [68,72]
    lrs = [0.01, 0.03, 0.1, 0.3, 1] # 5
    # alphas = [0, 0.1, 0.5, 0.9, 1] # 5
    kldiv_temps = [0.1, 1, 10, 100] # 4
    # mse_temps = [0] # 1
    # losses = ['mse','kldiv'] # 2

    # PLOT KLDIV RESULTS
    i=0
    for temp in kldiv_temps:
        info_per_line = {}
        for lr in lrs:
            # for each learning rate, have a different line (we are plotting prec@1 vs. alpha)
            info_per_line['lr={}'.format(lr)] = {
                'lr':[lr],
                'distill_loss':['kldiv'],
                'temperature':[temp]
            }
        plt.figure(i)
        i += 1
        plot_driver(all_results, {}, info_per_line, 'alpha', 'best_prec1', var_info=var_info)
        plt.ylim(ylims)
        plt.savefig('/proj/distill/results/2020-01-22-tuneDistillHP/figures/kldiv_temp,{}_tune_hp_results.pdf'.format(temp))
        plt.close()

    # PLOT MSE RESULTS
    info_per_line = {}
    for lr in lrs:
            # for each learning rate, have a different line (we are plotting prec@1 vs. alpha)
        info_per_line['lr={}'.format(lr)] = {
            'lr':[lr],
            'distill_loss':['mse']
        }
    plt.figure(i)
    plot_driver(all_results, {}, info_per_line, 'alpha', 'best_prec1', var_info=var_info)
    plt.ylim(ylims)
    plt.savefig('/proj/distill/results/2020-01-22-tuneDistillHP/figures/mse_tune_hp_results.pdf')
    plt.close()

if __name__ == '__main__':
    plot_distillation_tune_hp_results()
