path="/data/babymind/WPBERT_models/WPBERT_LM_phonefirst/checkpoint-28/pytorch_model.bin"
model="WPBERT_LM_phonefirst"
metric="lexical"

python3 WPBERT_lexical_syntactic.py --pretrain_path $path --model_name $model --metric $metric --number 0 --gpu 0 &
python3 WPBERT_lexical_syntactic.py --pretrain_path $path --model_name $model --metric $metric --number 1 --gpu 0 &
python3 WPBERT_lexical_syntactic.py --pretrain_path $path --model_name $model --metric $metric --number 2 --gpu 1 &
python3 WPBERT_lexical_syntactic.py --pretrain_path $path --model_name $model --metric $metric --number 3 --gpu 1 &