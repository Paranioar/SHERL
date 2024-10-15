for task in "cola" "rte" "stsb" "mrpc" "qnli" "sst2" "mnli" "qqp"
do  
    for seed in 0 1 2 3
    do
        bash scripts/ladder_side_tuning_base.sh '0' $task "sherl" $seed
    done
done
