from imagededup.methods import CNN, DHash, PHash
import os
import jsonlines
import shutil
from collections import Counter

annotations_fold = "/root/.cache/torch/mmf/data/datasets/hateful_memes/defaults/annotations/"
train_path = os.path.join(annotations_fold, "train.jsonl")
test_seen_path = os.path.join(annotations_fold, "test_seen.jsonl")
test_unseen_path = os.path.join(annotations_fold, "test_unseen.jsonl")

images_fold = "/root/.cache/torch/mmf/data/datasets/hateful_memes/defaults/images/"
img_path = os.path.join(images_fold, "img")
test_seen_img_path = os.path.join(images_fold, "test_seen")
test_unseen_img_path = os.path.join(images_fold, "test_unseen")


def img_data_split(total_img_path, json_path, split_img_path):
    if not os.path.exists(split_img_path):
        os.mkdir(split_img_path)
    with jsonlines.open(json_path) as f:
        for i in f:
            img_name = os.path.split(i['img'])[1]
            shutil.copyfile(os.path.join(total_img_path, img_name),
                            os.path.join(split_img_path, img_name))


def get_triad_and_tuples(img_path, json_path, test=True):
    triad = get_duplicate_triad(img_path, json_path, test=test)
    img_tuple = get_duplicate_imgs_tuple(img_path)
    text_tuple = get_duplicate_texts_tuple(json_path)
    treated_triad = break_chain_in_triad(triad)
    treated_img_tuple, treated_text_tuple = get_treated_tuples(
        img_tuple, text_tuple, triad)
    return treated_triad, treated_img_tuple, treated_text_tuple


def get_duplicate_triad(img_path, json_path, test=False):
    img_tuple = get_duplicate_imgs_tuple(img_path)
    text_tuple = get_duplicate_texts_tuple(json_path)
    # To speed up, create a new dictionary for id and label in json_path
    if not test:
        json_dict = {}
        with jsonlines.open(json_path) as f:
            for i in f:
                json_dict[str(i['id'])] = i['label']
    # build 3-tuple
    triad = []
    for text_ids in text_tuple:
        for img_ids in img_tuple:
            # get 3-tuple, ids -> triad_ids [id1, id2, id3]，where id1 is both duplicate, id2 is text duplicate, id3 is img duplicate
            if len(set(text_ids + img_ids)) == 3:  # detect duplicate combinations
                if text_ids[0] == img_ids[0]:
                    triad_ids = [text_ids[0], text_ids[1], img_ids[1]]
                elif text_ids[0] == img_ids[1]:
                    triad_ids = [text_ids[0], text_ids[1], img_ids[0]]
                elif text_ids[1] == img_ids[0]:
                    triad_ids = [text_ids[1], text_ids[0], img_ids[1]]
                elif text_ids[1] == img_ids[1]:
                    triad_ids = [text_ids[1], text_ids[0], img_ids[0]]
                # get 3-tuple, label -> triad_labels
                if not test:
                    triad_labels = [json_dict[str(triad_id)]
                                    for triad_id in triad_ids]
                    # put triad_ids and triad_label into triad
                    triad.append((triad_ids, triad_labels))
                else:
                    triad.append(triad_ids)
    return triad


def get_duplicate_imgs_tuple(img_path):
    phasher = PHash()
    encodings = phasher.encode_images(image_dir=img_path)
    duplicates = phasher.find_duplicates(encoding_map=encodings)
    # Find two duplicate imgs --> duplicate_imgs_tuple
    duplicate_imgs_tuple = []
    img_checked = []
    for img, imgs in duplicates.items():
        if len(imgs) == 1 and img not in img_checked:
            temp = [img] + imgs
            duplicate_imgs_tuple.append(temp)
            # deduplication
            for each in temp:
                img_checked.append(each)
    # transformation type
    for i in range(len(duplicate_imgs_tuple)):
        duplicate_imgs_tuple[i] = [int(each.split('.')[0])
                                   for each in duplicate_imgs_tuple[i]]
    # post-processing:
    # Previous tests found that phash might put three pictures into two similar pairs, so a round of selection was added
    flatten_tuple = []
    for i in duplicate_imgs_tuple:
        for j in i:
            flatten_tuple.append(j)
    counter_dict = Counter(flatten_tuple)  # counter count, check duplicate
    error_keys = [key for key, value in counter_dict.items()
                  if value > 1]  # pick out duplicate images
    treated_dup_imgs_tuple = [each for each in duplicate_imgs_tuple if len(
        set(each+error_keys))-len(error_keys) == len(each)]  # remove duplicate image pairs

    return treated_dup_imgs_tuple


def get_duplicate_texts_tuple(json_path):
    # find the text repeated 2 times under the specified json
    # find all duplicate texts
    duplicate_texts = {}
    with jsonlines.open(json_path) as f:
        for i in f:
            if duplicate_texts.get(i['text']) == None:
                duplicate_texts[i['text']] = [i['id']]
            else:
                duplicate_texts[i['text']].append(i['id'])
    # only keep the duplicated 2 times
    duplicate_tuple = []
    for text, ids in duplicate_texts.items():
        if len(ids) == 2:
            duplicate_tuple.append(ids)
    return duplicate_tuple


def break_chain_in_triad(triad):
    flatten_triad = []
    for i in triad:
        for j in i:
            flatten_triad.append(j)
    counter_dict = Counter(flatten_triad)
    error_keys = [key for key, value in counter_dict.items() if value > 1]
    treated_triad = [i for i in triad if len(
        set(i + error_keys))-len(error_keys) == len(i)]
    return treated_triad


# Note that the triad here is test_triad instead of treated_triad after deduplication
def get_treated_tuples(img_tuple, text_tuple, triad):
    flatten_triad = []
    for i in triad:
        for j in i:
            flatten_triad.append(j)
    treated_img_tuple = [i for i in img_tuple if len(
        set(flatten_triad + i)) - len(set(flatten_triad)) == len(i)]
    treated_text_tuple = [i for i in text_tuple if len(
        set(flatten_triad + i)) - len(set(flatten_triad)) == len(i)]
    return treated_img_tuple, treated_text_tuple
