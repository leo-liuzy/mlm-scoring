pretrain_model_dir=pre-trained_language_models
target_dir="checkpoint_52_500000"
output_dir="output"

for file in data/* ; do
    echo "Scoring pairs in ${file}..."
    for model_path in ${pretrain_model_dir}/roberta.base.faststatsync.me_fp16.cmpltsents.mp0.2.roberta_base.tps512.adam.fp16adam.b2_0.98.eps1e-06.cl0.0.lr0.0006.wu24000.dr0.1.atdr0.1.wd0.01.ms32.uf2.mu500000.s1.ngpu128 ; do
        # echo $model
        echo "Scoring with ${model_path}..."
        if [[ ! -d "$model_path" ]]
        then
            continue
        fi
        model_name=${model_path#"$pretrain_model_dir/"}
        mkdir -p ${output_dir}/${model_name}/
        mlm score \
            --mode ref \
            --model ${model_path}/${target_dir} \
            --gpus 0 \
            --split-size 500 \
            ${file} \
            > ${output_dir}/${model_name}/"$(basename ${file} .txt).lm.json"
        echo 
    done
done