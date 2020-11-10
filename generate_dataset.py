import os
import shutil
import pandas as pd
from sklearn.model_selection import KFold
import triad_tuples


def save_split_jsonl(img_path, json_path, output_json_path):
    triad, img_tuple, text_tuple = triad_tuples.get_triad_and_tuples(
        img_path, json_path)

    test_samples = pd.read_json(json_path, lines=True).set_index('id')
    test_samples['label'] = None
    for tuple_3 in triad:
        test_samples.loc[tuple_3[0], 'label'] = 1
        test_samples.loc[tuple_3[1], 'label'] = 0
        test_samples.loc[tuple_3[2], 'label'] = 0
    test_samples_split = test_samples.dropna(how='any', axis=0).reset_index()
    test_samples_split.to_json(output_json_path, orient='records', lines=True)


def main():
    if not os.path.exists(triad_tuples.test_seen_img_path):
        os.mkdir(triad_tuples.test_seen_img_path)
    if not os.path.exists(triad_tuples.test_unseen_img_path):
        os.mkdir(triad_tuples.test_unseen_img_path)

    triad_tuples.img_data_split(
        triad_tuples.img_path, triad_tuples.test_seen_path, triad_tuples.test_seen_img_path)
    triad_tuples.img_data_split(
        triad_tuples.img_path, triad_tuples.test_unseen_path, triad_tuples.test_unseen_img_path)

    save_split_jsonl(triad_tuples.test_unseen_img_path, triad_tuples.test_unseen_path,
                     './test_unseen_split.jsonl')
    save_split_jsonl(triad_tuples.test_seen_img_path, triad_tuples.test_seen_path,
                     './test_seen_split.jsonl')

    train_samples = pd.read_json(triad_tuples.train_path, lines=True)
    label_unseen_samples = pd.read_json(
        './test_unseen_split.jsonl', lines=True)
    label_seen_samples = pd.read_json('./test_seen_split.jsonl', lines=True)
    train_samples = pd.concat(
        [train_samples, label_unseen_samples, label_seen_samples], ignore_index=True)
    train_samples.label = train_samples.label.map(int)

    kf = KFold(n_splits=5, random_state=2020, shuffle=True)
    kf_matrix = []
    for train_index, val_index in kf.split(train_samples):
        train, val = train_samples.loc[train_index], train_samples.loc[val_index]
        kf_matrix.append([train, val])

    path_data = 'kfold/'
    if os.path.exists(path_data):
        shutil.rmtree(path_data)
    os.mkdir(path_data)
    for i in range(5):
        kf_matrix[i][0].to_json(
            path_data+'train{0}.jsonl'.format(i), orient='records', lines=True)
        kf_matrix[i][1].to_json(
            path_data+'dev{0}.jsonl'.format(i), orient='records', lines=True)

    kfold_path = os.path.join(triad_tuples.annotations_fold, path_data)
    if os.path.exists(kfold_path):
        shutil.rmtree(kfold_path)
    shutil.move(path_data, triad_tuples.annotations_fold)


if __name__ == '__main__':
    main()
