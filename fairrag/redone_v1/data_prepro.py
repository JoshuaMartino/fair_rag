"in this file I impklment the preprocessing of the data for people in wikipedia and bias in bios datasets"

import pandas as pd
import numpy as np



def download_people_in_wikipedia(Folder_path):
    from nomic import AtlasDataset
    dataset = AtlasDataset("nomic/wiki-people")
    atlas_map = dataset.maps[0]
    df = atlas_map.data.df
    df.to_pickle(Folder_path + "/people_in_wikipedia.pkl")
    return df

def download_bias_in_bios(folder_path):
    splits = {'train': 'data/train-00000-of-00001-0ab65b32c47407e8.parquet', 'test': 'data/test-00000-of-00001-5598c840ce8de1ee.parquet', 'dev': 'data/dev-00000-of-00001-e6551072fff26949.parquet'}
    df = None
    for split in splits.keys():
        df = pd.read_parquet("hf://datasets/LabHC/bias_in_bios/" + splits[split])
        df.to_csv(folder_path + f"/bias_in_bios_{split}.csv")
    return df

def preprocess_people_in_wikipedia(wiki_path, mapping_path, file_path):
    wiki = pd.read_pickle(wiki_path)
    mapping = pd.read_csv(mapping_path)
    occupation_list = set(mapping['profession_name'])
    max_tokens =150
    passages = []
    for _, row in wiki .iterrows():
        if row['occupation_level_3'] in occupation_list and row["text"] is not None:
            words = row['text'].split()
            for start in range(0, len(words), max_tokens):
                passage_text = " ".join(words[start:start + max_tokens])
                passages.append({
                    'text': passage_text,
                    'gender': row['gender'],
                    'title': row['title'],
                'occupation': row['occupation_level_3']
            })
    wiki_subset = pd.DataFrame(passages)
    wiki_subset.to_csv(file_path+"wiki_24.csv", index=False)
    return wiki_subset

def preprocess_bias_in_bios(old_folder_path, new_folder_path,mapping_path):
    splits = ['train', 'test', 'dev']
    mapping = pd.read_csv(mapping_path)
    df=None
    for split in splits:
        df = pd.read_csv(old_folder_path + f"/bias_in_bios_{split}.csv")
        df = df.merge(mapping,on="profession", how='left' )
        df.rename(columns={'Unnamed: 0': 'bio_id', 'profession': 'profession_id', 'profession_name': 'profession', 'hard_text': 'bio'}, inplace=True)
        df_24_occ = df[~df['profession'].isin(mapping['profession_name'])].copy()
        df_24_occ["gender_string"] = df_24_occ["gender"].map({0: "male", 1: "female"})
        df_24_occ.to_csv(new_folder_path + f"/bias_in_bios_{split}.csv", index=False)
    return df
