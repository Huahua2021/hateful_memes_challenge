import os
import shutil
import csv
import pandas as pd
import triad_tuples


def csv_alter(triad, img_tuple, text_tuple, csv_path, csv_save_path):
    ori_f = csv.reader(open(csv_path, 'r'))
    ori_data = []
    for i in ori_f:
        ori_data.append(i)

    triad_alter_num = 0
    img_tuple_alter_num = 0
    text_tuple_alter_num = 0
    # alter 3-tuple
    for ids in triad:
        for data in ori_data:
            if data[0] == str(ids[0]):
                data[1] = '1'
                data[2] = '1'
                triad_alter_num += 1
            elif data[0] == str(ids[1]):
                data[1] = '0'
                data[2] = '0'
                triad_alter_num += 1
            elif data[0] == str(ids[2]):
                data[1] = '0'
                data[2] = '0'
                triad_alter_num += 1

    # alter img_tuple
    for ids in img_tuple:
        id_0 = str(ids[0])
        id_1 = str(ids[1])
        prob_0 = None
        label_0 = None
        prob_1 = None
        label_1 = None
        # find ids
        for data in ori_data:
            if data[0] == id_0:
                prob_0 = data[1]
                label_0 = data[2]
                break
        for data in ori_data:
            if data[0] == id_1:
                prob_1 = data[1]
                label_1 = data[2]
                break
        # compare and alter
        if prob_0 is not None and prob_1 is not None:
            if float(prob_0) >= float(prob_1):
                for data in ori_data:
                    if data[0] == id_0:
                        data[1] = '1'
                        data[2] = '1'
                        img_tuple_alter_num += 1
                        break
                for data in ori_data:
                    if data[0] == id_1:
                        data[1] = '0'
                        data[2] = '0'
                        img_tuple_alter_num += 1
                        break
            else:
                for data in ori_data:
                    if data[0] == id_0:
                        data[1] = '0'
                        data[2] = '0'
                        img_tuple_alter_num += 1
                        break
                for data in ori_data:
                    if data[0] == id_1:
                        data[1] = '1'
                        data[2] = '1'
                        img_tuple_alter_num += 1
                        break

    # alter text_tuple
    for ids in text_tuple:
        id_0 = str(ids[0])
        id_1 = str(ids[1])
        prob_0 = None
        label_0 = None
        prob_1 = None
        label_1 = None
        # find ids
        for data in ori_data:
            if data[0] == id_0:
                prob_0 = data[1]
                label_0 = data[2]
                break
        for data in ori_data:
            if data[0] == id_1:
                prob_1 = data[1]
                label_1 = data[2]
                break
        # compare and alter
        if prob_0 is not None and prob_1 is not None:
            if float(prob_0) >= float(prob_1):
                for data in ori_data:
                    if data[0] == id_0:
                        data[1] = '1'
                        data[2] = '1'
                        text_tuple_alter_num += 1
                        break
                for data in ori_data:
                    if data[0] == id_1:
                        data[1] = '0'
                        data[2] = '0'
                        text_tuple_alter_num += 1
                        break
            else:
                for data in ori_data:
                    if data[0] == id_0:
                        data[1] = '0'
                        data[2] = '0'
                        text_tuple_alter_num += 1
                        break
                for data in ori_data:
                    if data[0] == id_1:
                        data[1] = '1'
                        data[2] = '1'
                        text_tuple_alter_num += 1
                        break

    with open(csv_save_path, 'w') as alter_f:
        writer = csv.writer(alter_f)
        for i in ori_data:
            writer.writerow(i)

    print('triad_alter_num:{}, img_tuple_alter_num:{}, text_tuple_alter_num:{}'.format(
        triad_alter_num, img_tuple_alter_num, text_tuple_alter_num))


def treat_csv(path, save_path):
    triad, img_tuple, text_tuple = triad_tuples.get_triad_and_tuples(
        triad_tuples.test_unseen_img_path, triad_tuples.test_unseen_path)
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.mkdir(save_path)
    for file in os.listdir(path):
        input_csv = os.path.join(path, file)
        output_csv = os.path.join(save_path, file)
        csv_alter(triad, img_tuple, text_tuple, input_csv, output_csv)


def stacking(path, save_path):
    results = []
    for file in os.listdir(path):
        data = pd.read_csv(path+'/'+file, index_col=0)
        results.append(data)
    proba_test = pd.concat([rst.proba for rst in results], axis=1)
    proba_test.columns = range(proba_test.shape[1])
    rst = pd.DataFrame()
    rst['proba'] = proba_test.mean(1)
    rst['label'] = (rst['proba'] >= 0.5).map(int)
    rst.to_csv(save_path)


def main():
    treat_csv('./csv', './treated_csv')
    stacking('./treated_csv', './final_result.csv')


if __name__ == '__main__':
    main()
