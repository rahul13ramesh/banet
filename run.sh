export PATH="/usr/local/nvidia/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/nvidia/lib:$LD_LIBRARY_PATH"

# Tools config for CUDA, Anaconda installed in the common /tools directory
source /tools/config.sh
# Activate your environment
source  /scratch/scratch1/shweta/virtualEnv/shweta_env/bin/activate

cd /scratch/scratch1/rahul/videoCaption/banet/

python -u video.py > logs/out_video.log
python -u caption.py > logs/out_caption.log
python -u  train.py > logs/out_train.log
python -u  evaluate.py > logs/out_eval.log
