export RANK=0
export WORLD_SIZE=1
export LOCAL_RANK=0
export MASTER_ADDR=localhost
export MASTER_PORT=12355

python evaluate.py --cfg-path lavis/projects/gpt4point/eval/captioning3d_cap3d_opt2.7b_eval.yaml