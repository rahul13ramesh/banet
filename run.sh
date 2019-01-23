export PATH="/usr/local/nvidia/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/nvidia/lib:$LD_LIBRARY_PATH"

# Tools config for CUDA, Anaconda installed in the common /tools directory
source /tools/config.sh
# Activate your environment
source  /scratch/scratch1/shweta/virtualEnv/shweta_env/bin/activate

cd /scratch/scratch1/rahul/videoCaption/banet/
#/scratch/scratch1/shweta/video-captioning/virtualEnv/videocaption/bin/python2.7 video.py > out.txt
python -u video.py > out.txt
#python -u caption.py > out_caption_nominus.txt
#python -u  train.py > out_train_nominus
# python -u  evaluate.py > out_eval
