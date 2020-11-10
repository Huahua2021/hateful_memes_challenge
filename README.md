# Prerequisites
1. Install MMF following [the installation docs](https://mmf.sh/docs/getting_started/installation/).

2. Convert data into MMF format. Following [the prerequisites part of Hateful Memes Dataset](https://github.com/facebookresearch/mmf/tree/master/projects/hateful_memes).<br>
There will be a path that shows where is data of .jsonl and images. Copy it! It will be used in step 4.

3. Install requirement.txt
```
pip install -r requirements.txt
```

4. Download the pre-trained models. 
Address: https://pan.baidu.com/s/1KzRPwRo9BQORBdYLhoed_A
Password: 1122

5. Modify path<br>
In *triad_tuples.py*, change *annotations_fold* and *images_fold* to the path of data of .jsonl and images.<br>
In *predict_training.sh*, change *annotations_fold* to the path of data of .jsonl.<br>
The path is copied in step 2.

# Run
## If you want to predict with given models (fast)
1. Modify predict_models.sh<br>
In *predict_models.sh*, change *predict_cuda_num* to index of cuda that you want to predict.
2. Grant run permission
```
chmod a+x predict_models.sh
```
or
```
sudo chmod a+x predict_models.sh
```
3. Run predict_models.sh
```
./predict_models.sh
```
---
## If you want to predict with training (slow) 
1. Modify predict_training.sh<br>
In *predict_training.sh*, change *training_cuda_num* and *predict_cuda_num* to index of cuda that you want to train and predict.
2. Grant run permission
```
chmod a+x predict_training.sh
```
or
```
sudo chmod a+x predict_training.sh
```
3. Run predict_training.sh
```
./predict_training.sh
```

# Result
*final_result.csv* in the current directory is the final submission.

# P.S.
If it is interrupted when run *predict_training.sh*, change *train.jsonl.bak* and *dev_unseen.json.bak* to *train.jsonl* and *dev_unseen.jsonl* before run it again or run any other MMF models.<br>
Because I use KFold technique, *train.jsonl* and *dev_unseen.jsonl* are changed during runing.
