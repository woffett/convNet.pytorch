import argparse
import models
from data import DataRegime, SampledDataRegime

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))
parser = argparse.ArgumentParser(description='Calculating eigenspace overlap')

parser.add_argument('--path', type=str, required=True,
                    help='path to model weights')
parser.add_argument('--model', '-a', metavar='MODEL', default='resnet',
                    choices=model_names,
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: resnet)')
parser.add_argument('--model-config', default='',
                    help='additional architecture configuration')
parser.add_argument('--input-size', type=int, default=None,
                    help='image input size')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--dataset', metavar='DATASET', default='cifar100',
                    help='dataset name or folder')
parser.add_argument('--datasets-dir', metavar='DATASETS_DIR', default='~/Datasets',
                    help='datasets dir')

LOGIT_DIMS = {
    'resnet18': 512,
    'mobilenet': 1024,
}

def get_embedding(model, model_name, loader, batch_size, k):
    '''
    we can think of the loader as a n x d design matrix
    we want to output an n x k matrix, k being the dimension of the 
    last layer before the linear layer and softmax
    '''
    n = batch_size * len(loader)
    k = LOGIT_DIMS['model']
    embedding = torch.tensor((n, k))
    model = model.to('cuda')
    for i, (inp, target) in enumerate(loader):
        inp = inp.to('cuda')
        outp = model.features(inp)
        if model_name == 'mobilenet':
            outp = model.avg_pool(x)
            outp = outp.view(outp.size(0), -1)
        embedding[(i-1)*batch_size:i*batch_size, :] = outp

    return embedding
    

def main(args):
    model = models.__dict__[args.model]
    model_config = {'dataset': args.dataset} \
        if args.model not in ['efficientnet', 'mobilenet'] \
           else dict()
    if args.model_config is not '':
        model_config = dict(model_config, **literal_eval(args.model_config))
        
    model = model(**model_config)
    data = DataRegime(getattr(model, 'data_eval_regime', None),
                      defaults={'datasets_path': args.datasets_dir,
                                'name': args.dataset,
                                'split': None,
                                'augment': False,
                                'input_size': args.input_size,
                                'batch_size': args.eval_batch_size,
                                'shuffle': False,
                                'num_workers': args.workers,
                                'pin_memory': True,
                                'drop_last': False})
    loader = data.get_loader()
    outp_matrix = get_outputs(model, args.model, loader, args.batch_size)

if __name__ == '__main__':
    args = parser.parse_rags()
    main(args)
