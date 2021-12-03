path="/data/babymind/WPBERT_models/RES_INTEGER/WPBERT_REP_WP/checkpoint-4/pytorch_model.bin"
gpu=3
model_name="VQ3VQ2_submission_test"

for hidden in 7
do
    for repr in "seq" 
    do
        python3 WPBERT_semantic.py --pretrain_path $path --model_name $model_name --hidden $hidden --repr $repr --gpu $gpu &
    done
done