export OPENBLAS_NUM_THREADS=4  
export WANDB_API_KEY=local-04561b3685afca039aca56f554efe6a008118c01
export WANDB_BASE_URL=http://www.zangzelin.fun:4080

export GIT_SSH_COMMAND="ssh -i /root/ssh/zelin -o IdentitiesOnly=yes -p 8084" 
export GIT_SSH_COMMAND="ssh -i /zangzelin/ssh/zelin -o IdentitiesOnly=yes -p 8084" 


python main.py fit -c conf/step1/G_xc_1gpu_jkyliu.yaml