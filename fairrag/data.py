'This file coantains the reading in methods for the datasets aswell as the creation of the Train and Test split of the data"'
from pathlib import Path
import pandas as pd

def load_bias_in_bios(split):
    """LabHC/bias_in_bios: hard_text, profession (ClassLabel), gender (0=male, 1=female)."""
    df = pd.read_csv(f"datasets/b_in_b/bias_in_bios_{split}.csv")
    return df

def load_wiki_24(file_path):
    df = pd.read_csv(file_path)
    return df

def populate_test_collection(b_in_b, folder_path):
    occupation_counts = pd.crosstab(b_in_b["profession"],b_in_b['gender'])
    # Appendix A.2: keep occupations with at least 100 biographies per gender.
    below_threshold = occupation_counts[(occupation_counts[1]<100) | (occupation_counts[0]<100)]
    occupations_to_remove = below_threshold.index.to_list()
    occupations_to_remove.append('paralegal')
    b_in_b = b_in_b[~b_in_b['profession'].isin(occupations_to_remove)].copy()
    occupations = b_in_b['profession'].unique().tolist()
    files = []
    for o in occupations:
        male_rel_samples = b_in_b[(b_in_b['profession']==o) & (b_in_b['gender']==0)].sample(n=100, random_state=42)
        male_rel_samples['y'] = 1
        male_rel_samples['target']=o
        
        female_rel_samples = b_in_b[(b_in_b['profession']==o) & (b_in_b['gender']==1)].sample(n=100, random_state=42)
        female_rel_samples['y'] = 1
        female_rel_samples['target']=o
        male_unrel_samples = b_in_b[(b_in_b['profession']!=o) & (b_in_b['gender']==0)].sample(n=100, random_state=42)
        male_unrel_samples['y'] = 0
        male_unrel_samples['target']=o
        female_unrel_samples = b_in_b[(b_in_b['profession']!=o) & (b_in_b['gender']==1)].sample(n=100, random_state=42)
        female_unrel_samples['y'] = 0
        female_unrel_samples['target']=o
        test_samples = pd.concat([male_rel_samples, female_rel_samples, male_unrel_samples, female_unrel_samples], axis=0)
        test_samples['source_profession'] = test_samples['profession']
        output_path =Path(folder_path) / f"{o}.csv"
        test_samples.to_csv(output_path, index=False)
        files.append(test_samples)
    return files

def populate_train_collection(b_in_b, folder_path):
    """CGLC training set (Appendix A.2). Build it from the Bias in Bios *test*
    split: 25 male + 25 female positives and 25 + 25 negatives per profession,
    without reuse."""
    # The preprocessed CSVs name the identifier "bio_id"; older files use "id".
    id_column = "id" if "id" in b_in_b.columns else "bio_id"
    used_ids = set()
    mini_sets = []
    occupation_counts = pd.crosstab(b_in_b["profession"],b_in_b['gender'])
    below_threshold = occupation_counts[(occupation_counts[1]<25) | (occupation_counts[0]<25)]
    occupations_to_remove = below_threshold.index.to_list() 
    b_in_b = b_in_b[~b_in_b['profession'].isin(occupations_to_remove)].copy()
    occupations = b_in_b['profession'].unique().tolist()
    for o in occupations:
        male_rel_samples = b_in_b[(b_in_b['profession']==o) & (b_in_b['gender']==0) & (~b_in_b[id_column].isin(used_ids))].sample(n=25, random_state=42)
        male_rel_samples['y'] = 1
            
        used_ids.update(male_rel_samples[id_column].tolist())
        female_rel_samples = b_in_b[(b_in_b['profession']==o) & (b_in_b['gender']==1) & (~b_in_b[id_column].isin(used_ids))].sample(n=25, random_state=42)
        female_rel_samples['y'] = 1
            
        used_ids.update(female_rel_samples[id_column].tolist())
        male_unrel_samples = b_in_b[(b_in_b['profession']!=o) & (b_in_b['gender']==0) & (~b_in_b[id_column].isin(used_ids))].sample(n=25, random_state=42)
        male_unrel_samples['y'] = 0
           
        used_ids.update(male_unrel_samples[id_column].tolist())
        female_unrel_samples = b_in_b[(b_in_b['profession']!=o) & (b_in_b['gender']==1) & (~b_in_b[id_column].isin(used_ids))].sample(n=25, random_state=42)
        female_unrel_samples['y'] = 0
            
        used_ids.update(female_unrel_samples[id_column].tolist())
        mini_set = pd.concat([male_rel_samples, female_rel_samples, male_unrel_samples, female_unrel_samples], axis=0)
        mini_set['target'] = o
        mini_set['source_profession'] =mini_set['profession']
        mini_sets.append(mini_set)
    train_samples = pd.concat(mini_sets, axis=0)
    shuffled_train_samples = train_samples.sample(frac=1, random_state=42).reset_index(drop=True)
    output_path = Path(folder_path)/"train.csv"
    shuffled_train_samples.to_csv(output_path, index=False)
    return shuffled_train_samples

def main():
    # Inference-time test set: built from the Bias in Bios train split.
    b_in_b = load_bias_in_bios("train")
    path = "datasets/occupations/"
    Path(path).mkdir(parents=True, exist_ok=True)
    test = populate_test_collection(b_in_b=b_in_b, folder_path=path)

    # CGLC optimisation set: built from the Bias in Bios test split and
    # restricted to the occupations that have a test file.
    b_in_b_test = load_bias_in_bios("test")
    test_occupations = {frame["target"].iloc[0] for frame in test}
    b_in_b_test = b_in_b_test[b_in_b_test["profession"].isin(test_occupations)]
    train_path = "datasets/train/"
    Path(train_path).mkdir(parents=True, exist_ok=True)
    populate_train_collection(b_in_b=b_in_b_test, folder_path=train_path)

if __name__ =="__main__":
    main()



