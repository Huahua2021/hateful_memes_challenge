training_cuda_num=2,3,4,5
predict_cuda_num=2
annotations_path=/root/.cache/torch/mmf/data/datasets/hateful_memes/defaults/annotations/

if [ -d "${annotations_path}kfold" ]; then
rm -r ${annotations_path}kfold
fi

python generate_dataset.py

cp ${annotations_path}train.jsonl ${annotations_path}train.jsonl.bak
cp ${annotations_path}dev_unseen.jsonl ${annotations_path}dev_unseen.jsonl.bak

if [ -d "./csv" ]; then
rm -r ./csv
fi

if [ -d "./treated_csv" ]; then
rm -r ./treated_csv
fi

for ((i=0; i<=4; i++))
do
cp ${annotations_path}kfold/train${i}.jsonl ${annotations_path}train.jsonl
cp ${annotations_path}kfold/dev${i}.jsonl ${annotations_path}dev_unseen.jsonl

CUDA_VISIBLE_DEVICES=${training_cuda_num} mmf_run config=kfold_region.yaml model=visual_bert dataset=hateful_memes run_type=train checkpoint.resume_zoo=visual_bert.pretrained.cc.full
CUDA_VISIBLE_DEVICES=${predict_cuda_num} mmf_predict config=kfold_region.yaml model=visual_bert dataset=hateful_memes run_type=test checkpoint.resume_file=./save/models/model_3000.ckpt checkpoint.resume_pretrained=False
python mv_csv.py

CUDA_VISIBLE_DEVICES=${training_cuda_num} mmf_run config=kfold.yaml model=visual_bert dataset=hateful_memes run_type=train checkpoint.resume_zoo=visual_bert.pretrained.cc.small_fifty_pc
CUDA_VISIBLE_DEVICES=${predict_cuda_num} mmf_predict config=kfold.yaml model=visual_bert dataset=hateful_memes run_type=test checkpoint.resume_file=./save/models/model_3000.ckpt checkpoint.resume_pretrained=False
python mv_csv.py

CUDA_VISIBLE_DEVICES=${training_cuda_num} mmf_run config=kfold_region.yaml model=visual_bert dataset=hateful_memes run_type=train checkpoint.resume_zoo=visual_bert.pretrained.cc.full
CUDA_VISIBLE_DEVICES=${predict_cuda_num} mmf_predict config=kfold_region.yaml model=visual_bert dataset=hateful_memes run_type=test checkpoint.resume_file=./save/models/model_3000.ckpt checkpoint.resume_pretrained=False
python mv_csv.py

CUDA_VISIBLE_DEVICES=${training_cuda_num} mmf_run config=kfold_region.yaml model=visual_bert dataset=hateful_memes run_type=train checkpoint.resume_zoo=visual_bert.finetuned.hateful_memes.direct
CUDA_VISIBLE_DEVICES=${predict_cuda_num} mmf_predict config=kfold_region.yaml model=visual_bert dataset=hateful_memes run_type=test checkpoint.resume_file=./save/models/model_3000.ckpt checkpoint.resume_pretrained=False
python mv_csv.py
done

mv ${annotations_path}train.jsonl.bak ${annotations_path}train.jsonl
mv ${annotations_path}dev_unseen.jsonl.bak ${annotations_path}dev_unseen.jsonl
rm -r ${annotations_path}kfold

python process.py

