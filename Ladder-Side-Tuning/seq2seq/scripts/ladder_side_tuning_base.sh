# This scripts trains Adapters method.
# For smaller datasets of GLUE (mrpc, cola, and stsb), we set the `num_train_epochs` to 20,
# for other larger datasets in GLUE we used `num_train_epochs` of 3. For all datasets we tried
# with the adapter's bottleneck size of `task_reduction_factor`=[32, 16, 8], and report the 
# results on the test set for the model performing the best on the validation set.

folder_name=all_output_logs/
if [ ! -d ${folder_name} ] ; then
    mkdir -p ${folder_name}
fi

source scripts/env.sh

config_name=side_transformers
other_name=$3
output_name=${config_name}_${other_name}

r=$5
lr=3e-3
weight_decay=0.0
lr_factor=1.0

encoder_side_layers="[1,2,3,5,6,7,9,10,11]"

rm -r outputs/${output_name}

python scripts/update_scripts_for_given_input.py configs/${config_name}.json model_name_or_path str t5-base
python scripts/update_scripts_for_given_input.py configs/${config_name}.json tokenizer_name str t5-base

python scripts/update_scripts_for_given_input.py configs/${config_name}.json seed int $4
python scripts/update_scripts_for_given_input.py configs/${config_name}.json task_name str $2
python scripts/update_scripts_for_given_input.py configs/${config_name}.json eval_dataset_name str $2
python scripts/update_scripts_for_given_input.py configs/${config_name}.json test_dataset_name str $2
python scripts/update_scripts_for_given_input.py configs/${config_name}.json output_dir str outputs/${output_name}
python scripts/update_scripts_for_given_input.py configs/${config_name}.json use_gate str "learnable"
python scripts/update_scripts_for_given_input.py configs/${config_name}.json task_reduction_factor int ${r}
python scripts/update_scripts_for_given_input.py configs/${config_name}.json load_side_pretrained_weights str fisher-v2
python scripts/update_scripts_for_given_input.py configs/${config_name}.json learning_rate float ${lr}
python scripts/update_scripts_for_given_input.py configs/${config_name}.json num_train_epochs int ${num_epochs[$2]}
python scripts/update_scripts_for_given_input.py configs/${config_name}.json weight_decay float ${weight_decay}
python scripts/update_scripts_for_given_input.py configs/${config_name}.json lr_factor float ${lr_factor}
python scripts/update_scripts_for_given_input.py configs/${config_name}.json add_bias_sampling str2bool True
python scripts/update_scripts_for_given_input.py configs/${config_name}.json create_side_lm str2bool False
python scripts/update_scripts_for_given_input.py configs/${config_name}.json freeze_side_lm str2bool False
python scripts/update_scripts_for_given_input.py configs/${config_name}.json add_residual_after str2bool False
python scripts/update_scripts_for_given_input.py configs/${config_name}.json encoder_side_layers eval ${encoder_side_layers}
python scripts/update_scripts_for_given_input.py configs/${config_name}.json decoder_side_layers eval ${encoder_side_layers}


CUDA_VISIBLE_DEVICES=$1 python run_seq2seq.py  configs/${config_name}.json

cp outputs/${output_name}/all_results.json  all_output_logs/t5-base:${output_name}_bias_r${r}_lr${lr}_$2@$4.json