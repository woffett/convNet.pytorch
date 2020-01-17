. /proj/anaconda3/etc/profile.d/conda.sh
conda env create -f distill_env.sh
conda activate ml
conda install pytorch=1.0.0 torchvision cudatoolkit=9.0 -c pytorch