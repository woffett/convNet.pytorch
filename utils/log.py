import shutil
import os
from itertools import cycle
import torch
import logging
import logging.config
import datetime
import json
import csv
import subprocess
import pathlib

import pandas as pd
from bokeh.io import output_file, save, show
from bokeh.plotting import figure
from bokeh.layouts import column
from bokeh.models import Div

try:
    import hyperdash
    HYPERDASH_AVAILABLE = True
except ImportError:
    HYPERDASH_AVAILABLE = False

def get_date_str():
    return '{:%Y-%m-%d}'.format(datetime.date.today())

def non_default_args(parser, config):
    non_default = []
    for action in parser._actions:
        default = action.default
        key = action.dest
        if key == 'help':
            continue
        val = config[key]
        if val != default:
            non_default.append((key,val))
    assert len(non_default) > 0, 'There must be a non-default arg'
    return non_default
            

def get_runname(parser, config, full=False):
    runname = ''
    to_skip = (
        'config_file',
        'results_dir',
        'rungroup_name',
        'datasets_dir',
        'input_size',
        'model_config',
        'teacher_path',
        'teacher_model_config',
        'dtype',
        'device',
        'device_ids',
        'world-size',
        'local_rank',
        'dist_init',
        'dist_backend',
        'workers',
        'epochs',
        'Start_epoch',
        'drop_optim_state',
        'save_all',
        'label_smoothing',
        'mixup',
        'cutmix',
        'duplicates',
        'chunk_batch',
        'cutout',
        'autoaugment',
        'print_freq',
        'adapt_grad_norm',
        'resume',
        'evaluate',
        'tensorwatch',
        'tensorwatch_port',
        'profile',
        'results_filename'
    )
    required = (
        'distill_loss',
        'alpha',
        'beta',
        #'temperature'
    )
    for key in required:
        val = config[key]
        format_str = '{},{:.3g}_' if type(val) is float else '{},{}_'
        runname += format_str.format(key, val)
    for key, val in non_default_args(parser, config):
        if key not in (to_skip + required):
            format_str = '{},{:.3g}_' if type(val) is float else '{},{}_'
            runname += format_str.format(key, val)
    # remove the final '_' from runname
    if full:
        runname = 'rungroup,' + config['rungroup_name'] + '_' + runname
    return runname[:-1] if runname[-1] == '_' else runname

def get_savepath(parser, args):
    basedir = args.results_dir
    rungroup = '{}-{}'.format(get_date_str(), args.rungroup_name)
    runname = get_runname(parser, dict(args._get_kwargs()))
    return os.path.join(basedir, rungroup, runname)

def export_args_namespace(args, filename):
    """
    args: argparse.Namespace
        arguments to save
    filename: string
        filename to save at
    """
    with open(filename, 'w') as fp:
        json.dump(dict(args._get_kwargs()), fp, sort_keys=True, indent=4)


def setup_logging(log_file='log.txt', resume=False, dummy=False):
    """
    Setup logging configuration
    """
    if dummy:
        logging.getLogger('dummy')
    else:
        if os.path.isfile(log_file) and resume:
            file_mode = 'a'
        else:
            file_mode = 'w'

        root_logger = logging.getLogger()
        if root_logger.handlers:
            root_logger.removeHandler(root_logger.handlers[0])
        logging.basicConfig(level=logging.DEBUG,
                            format="%(asctime)s - %(levelname)s - %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S",
                            filename=log_file,
                            filemode=file_mode)
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)


def plot_figure(data, x, y, title=None, xlabel=None, ylabel=None, legend=None,
                x_axis_type='linear', y_axis_type='linear',
                width=800, height=400, line_width=2,
                colors=['red', 'green', 'blue', 'orange',
                        'black', 'purple', 'brown'],
                tools='pan,box_zoom,wheel_zoom,box_select,hover,reset,save',
                append_figure=None):
    """
    creates a new plot figures
    example:
        plot_figure(x='epoch', y=['train_loss', 'val_loss'],
                        'title='Loss', 'ylabel'='loss')
    """
    if not isinstance(y, list):
        y = [y]
    xlabel = xlabel or x
    legend = legend or y
    assert len(legend) == len(y)
    if append_figure is not None:
        f = append_figure
    else:
        f = figure(title=title, tools=tools,
                   width=width, height=height,
                   x_axis_label=xlabel or x,
                   y_axis_label=ylabel or '',
                   x_axis_type=x_axis_type,
                   y_axis_type=y_axis_type)
    colors = cycle(colors)
    for i, yi in enumerate(y):
        f.line(data[x], data[yi],
               line_width=line_width,
               line_color=next(colors), legend=legend[i])
    f.legend.click_policy = "hide"
    return f


class ResultsLog(object):

    supported_data_formats = ['csv', 'json']

    def __init__(self, path='', title='', params=None, resume=False, data_format='csv'):
        """
        Parameters
        ----------
        path: string
            path to directory to save data files
        plot_path: string
            path to directory to save plot files
        title: string
            title of HTML file
        params: Namespace
            optionally save parameters for results
        resume: bool
            resume previous logging
        data_format: str('csv'|'json')
            which file format to use to save the data
        """
        if data_format not in ResultsLog.supported_data_formats:
            raise ValueError('data_format must of the following: ' +
                             '|'.join(['{}'.format(k) for k in ResultsLog.supported_data_formats]))

        if data_format == 'json':
            self.data_path = '{}.json'.format(path)
        else:
            self.data_path = '{}.csv'.format(path)
        if params is not None:
            export_args_namespace(params, '{}.json'.format(path))
        self.plot_path = '{}.html'.format(path)
        self.results = None
        self.clear()
        self.first_save = True
        if os.path.isfile(self.data_path):
            if resume:
                self.load(self.data_path)
                self.first_save = False
            else:
                os.remove(self.data_path)
                self.results = pd.DataFrame()
        else:
            self.results = pd.DataFrame()

        self.title = title
        self.data_format = data_format

        if HYPERDASH_AVAILABLE:
            name = self.title if title != '' else path
            self.hd_experiment = hyperdash.Experiment(name)
            if params is not None:
                for k, v in params._get_kwargs():
                    self.hd_experiment.param(k, v, log=False)

    def clear(self):
        self.figures = []

    def add(self, **kwargs):
        """Add a new row to the dataframe
        example:
            resultsLog.add(epoch=epoch_num, train_loss=loss,
                           test_loss=test_loss)
        """
        df = pd.DataFrame([kwargs.values()], columns=kwargs.keys())
        self.results = self.results.append(df, ignore_index=True)
        if hasattr(self, 'hd_experiment'):
            for k, v in kwargs.items():
                self.hd_experiment.metric(k, v, log=False)

    def smooth(self, column_name, window):
        """Select an entry to smooth over time"""
        # TODO: smooth only new data
        smoothed_column = self.results[column_name].rolling(
            window=window, center=False).mean()
        self.results[column_name + '_smoothed'] = smoothed_column

    def save(self, title=None):
        """save the json file.
        Parameters
        ----------
        title: string
            title of the HTML file
        """
        title = title or self.title
        if len(self.figures) > 0:
            if os.path.isfile(self.plot_path):
                os.remove(self.plot_path)
            if self.first_save:
                self.first_save = False
                logging.info('Plot file saved at: {}'.format(
                    os.path.abspath(self.plot_path)))

            output_file(self.plot_path, title=title)
            plot = column(
                Div(text='<h1 align="center">{}</h1>'.format(title)), *self.figures)
            save(plot)
            self.clear()

        if self.data_format == 'json':
            self.results.to_json(self.data_path, orient='records', lines=True)
        else:
            self.results.to_csv(self.data_path, index=False, index_label=False)

    def load(self, path=None):
        """load the data file
        Parameters
        ----------
        path:
            path to load the json|csv file from
        """
        path = path or self.data_path
        if os.path.isfile(path):
            if self.data_format == 'json':
                self.results.read_json(path)
            else:
                self.results.read_csv(path)
        else:
            raise ValueError('{} isn''t a file'.format(path))

    def show(self, title=None):
        title = title or self.title
        if len(self.figures) > 0:
            plot = column(
                Div(text='<h1 align="center">{}</h1>'.format(title)), *self.figures)
            show(plot)

    def plot(self, *kargs, **kwargs):
        """
        add a new plot to the HTML file
        example:
            results.plot(x='epoch', y=['train_loss', 'val_loss'],
                         'title='Loss', 'ylabel'='loss')
        """
        f = plot_figure(self.results, *kargs, **kwargs)
        self.figures.append(f)

    def image(self, *kargs, **kwargs):
        fig = figure()
        fig.image(*kargs, **kwargs)
        self.figures.append(fig)

    def end(self):
        if hasattr(self, 'hd_experiment'):
            self.hd_experiment.end()


def save_checkpoint(state, is_best, path='.', filename='checkpoint.pth.tar',
                    runname='', save_all=False):
    filename = os.path.join(path, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(path,
                                               runname + 'model_best.pth.tar'))
    if save_all:
        shutil.copyfile(filename, os.path.join(
            path, runname + 'checkpoint_epoch_%s.pth.tar' % state['epoch']))

def is_windows():
    """Determine if running on windows OS."""
    return os.name == 'nt'
    
def get_git_hash_and_diff(git_repo_dir=None, log=True, debug=False):
    git_hash = None
    git_diff = None
    if not is_windows():
        try:
            wd = os.getcwd()
            if git_repo_dir is None:
                git_repo_dir = pathlib.Path(__file__).parent.absolute()
            os.chdir(git_repo_dir)
            git_hash = str(subprocess.check_output(
                ['git','rev-parse','--short','HEAD']).strip())[2:9]
            git_diff = str(subprocess.check_output(['git','diff']).strip())[3:]
            if not debug:
                # if not in debug mode, local repo changes are not allowed.
                assert git_diff == '', 'Cannot have any local changes'
            os.chdir(wd)
            if log:
                logging.info('Git hash {}'.format(git_hash))
                logging.info('Git diff {}'.format(git_diff))
        except FileNotFoundError:
            if log:
                logging.info('Unable to get git hash.')
    return git_hash, git_diff


def gen_results_json(args, save_path, best_prec1, best_prec5, runname):
    master_dict = dict(args._get_kwargs())
    train_prec1 = []
    train_prec5 = []
    train_loss = []    
    train_eos = []
    val_prec1 = []
    val_prec5 = []
    val_loss = []
    val_eos = []

    results_filename = os.path.join(save_path, runname + '_results.csv')
    with open(results_filename) as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            train_prec1.append(float(row['training prec1']))
            train_prec5.append(float(row['training prec5']))
            train_loss.append(float(row['training loss']))
            train_eos.append(float(row['training eos']))
            val_prec1.append(float(row['validation prec1']))
            val_prec5.append(float(row['validation prec5']))
            val_loss.append(float(row['validation loss']))
            val_eos.append(float(row['validation eos']))

    results_dict = dict()
    results_dict['train_prec1'] = train_prec1
    results_dict['train_prec5'] = train_prec5
    results_dict['train_loss'] = train_loss
    results_dict['train_eos'] = train_eos
    results_dict['val_prec1'] = val_prec1
    results_dict['val_prec5'] = val_prec5
    results_dict['val_loss'] = val_loss
    results_dict['val_eos'] = val_eos

    # save final/best prec results
    results_dict['final_prec1'] = float(row['validation prec1'])
    results_dict['final_prec5'] = float(row['validation prec5'])
    results_dict['best_prec1'] = best_prec1
    results_dict['best_prec5'] = best_prec5

    # save final/best eos results
    results_dict['final_train_eos'] = train_eos[-1]
    results_dict['final_val_eos'] = val_eos[-1]
    results_dict['best_train_eos'] = max(train_eos)
    results_dict['best_val_eos'] = max(val_eos)

    master_dict['results'] = results_dict

    master_dict['githash'], master_dict['gitdiff'] = get_git_hash_and_diff(debug=args.debug)

    with open(os.path.join(save_path, runname + '_results.json'), 'w') as f:
        json.dump(master_dict, f, sort_keys=True, indent=4)
