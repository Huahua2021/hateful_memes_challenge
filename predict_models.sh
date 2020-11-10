predict_cuda_num=2

if [ -d "./csv" ]; then
rm -r ./csv
fi

if [ -d "./treated_csv" ]; then
rm -r ./treated_csv
fi

for ((i=1; i<=5; i++))
do
CUDA_VISIBLE_DEVICES=${predict_cuda_num} mmf_predict config=kfold_region.yaml model=visual_bert dataset=hateful_memes run_type=test checkpoint.resume_file=./models/model${i}.ckpt checkpoint.resume_pretrained=False
python mv_csv.py
done

for ((i=6; i<=10; i++))
do
CUDA_VISIBLE_DEVICES=${predict_cuda_num} mmf_predict config=kfold.yaml model=visual_bert dataset=hateful_memes run_type=test checkpoint.resume_file=./models/model${i}.ckpt checkpoint.resume_pretrained=False
python mv_csv.py
done

for ((i=11; i<=20; i++))
do
CUDA_VISIBLE_DEVICES=${predict_cuda_num} mmf_predict config=kfold_region.yaml model=visual_bert dataset=hateful_memes run_type=test checkpoint.resume_file=./models/model${i}.ckpt checkpoint.resume_pretrained=False
python mv_csv.py
done

python process.py

