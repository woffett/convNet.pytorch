import pathlib
import helpers

def get_cmdfile_path(filename):
    return str(pathlib.PurePath(helpers.get_base_dir(), 'scripts', filename))

def cmds_1_16_20_train_base_models():
    filename = get_cmdfile_path('1_16_20_train_base_models_cmds')
    cmd_format_str = ('qsub -V -b y -wd /proj/distill/wd -v '
                          'cmd="python /proj/distill/git/convNet.pytorch/main.py '
                          '--results-dir {} --datasets-dir {} --dataset {} --save {} --model {} '
                          '--model-config {} --input-size {} --dtype {} --autoaugment --device cuda '
                          '--epochs {} --batch-size {} --lr {} --weight-decay {} --print-freq 10 --seed {}" '
                          '/proj/distill/git/convNet.pytorch/ml_env.sh\n')
    results_dir = '/proj/distill/results'
    data_dir = '/proj/distill/data'
    dataset = 'cifar100'
    # training resnet18 and mobilenet
    run_names = ['resnet18_1_16_20', 'mobilenet_1_16_20']
    models = ['resnet','mobilenet']
    model_configs = ['\'{\\"groups\\": [1,1,1,1], \\"depth\\": 18, \\"width\\": [64, 128, 256, 512]}\'',
                    '\'{\\"num_classes\\": 100, \\"cifar\\": True}\'']
    input_size = 32
    dtype = 'float'
    epochs = 170
    batch_size = 128
    lr = 0.1
    weight_decay = 5e-4
    seed = 1

    with open(filename,'w') as f:
        for i,model in enumerate(models):
            model_config = model_configs[i]
            run_name = run_names[i]
            f.write(cmd_format_str.format(results_dir, data_dir, dataset, run_name, model, model_config,
                                          input_size, dtype, epochs, batch_size, lr, weight_decay, seed))

    # These are the commands that Jonathan used.
    # $PYTHON $MAIN \
    # 	--results-dir $OUTPUT --datasets-dir $DATA --save rn18_imgnet_channels --dataset cifar100
    # 	--model resnet --model-config  '{"groups": [1,1,1,1], "depth": 18, "width": [64, 128, 256, 512]}'
    # 	--input-size 32	--dtype float --autoaugment	--device cuda --epochs 170 --batch-size $BATCH --lr $LR
    # 	--weight-decay $DECAY --print-freq 10 --seed $SEED

    # $PYTHON $MAIN \
    # 	--results-dir $OUTPUT --datasets-dir $DATA --save mobilenet_cifar100 --dataset cifar100 
    #     --model mobilenet --model-config '{"num_classes": 100, "cifar": True}' 
    # 	--input-size 32 --dtype float --autoaugment --device cuda --epochs 170 --batch-size $BATCH --lr $LR \
    # 	--weight-decay $DECAY --print-freq 10 --seed $SEED

if __name__ == '__main__':
    cmds_1_16_20_train_base_models()