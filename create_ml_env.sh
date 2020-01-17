. /proj/anaconda3/etc/profile.d/conda.sh
conda env create -f distill_env.sh
conda activate ml
conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=9.0 -c pytorch
