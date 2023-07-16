import pandas as pd
import numpy as np

METADATA = pd.read_csv("2DScanImages/Image-info.csv")
METADATA_BLURRY = pd.read_csv("Blurry/metadata.csv")
METADATA_OOD = pd.read_csv("OOD/metadata.csv")
BLACK_WHITE_THRESHOLD = 85  # Bottom third of grayscale: 255 / 3 = 85
WILLIAMS_PURPLE = "#500082"
WILLIAMS_GOLD = "#FFBE0A"
OPP_FOOT = {"R":"L", "L":"R"}


def _create_km_pairs(df, name: str):
    '''
    Documenting Known Mated Pairs data process
    '''

    kms = []

    df = df.sort_values(by='Shoe Number')
    for i in range(len(df)):
        for j in range(i+1, len(df)):
            if (df.iloc[i].loc['Shoe Number'] == df.iloc[j].loc['Shoe Number'] and
                    df.iloc[i].loc['Foot'] == df.iloc[j].loc['Foot'] and
                    df.iloc[i].loc['Image Number'] == df.iloc[j].loc['Image Number'] and
                    df.iloc[i].loc['Visit Number'] == df.iloc[j].loc['Visit Number'] and
                    df.iloc[i].loc['Shoe Size'] == df.iloc[j].loc['Shoe Size']
                ):
                kms.append((df.iloc[i].loc['File Name'],
                           df.iloc[j].loc['File Name']))
                break

    kms_df = pd.DataFrame(kms)
    kms_df.rename(columns={0: 'q', 1: 'k'}, inplace=True)
    kms_df.to_csv(name, index=False)


def _create_knm_pairs(df, name: str, size: int = 2407):
    '''
    Documenting Known Non-Mated Pairs data process
    '''
    knms = []

    for i in range(len(df)):
        non_mated_row = np.random.randint(len(df))

        while (df.iloc[i].loc['File Name'] == df.iloc[non_mated_row].loc['File Name'] or
               df.iloc[i].loc['Shoe Number'] == df.iloc[non_mated_row].loc['Shoe Number'] or
               df.iloc[i].loc['Shoe Size'] != df.iloc[non_mated_row].loc['Shoe Size'] or
               df.iloc[i].loc['Shoe Make/Model'] != df.iloc[non_mated_row].loc['Shoe Make/Model'] or
               df.iloc[i].loc['Foot'] != df.iloc[non_mated_row].loc['Foot']):
            
            non_mated_row = np.random.randint(len(df))

        knms.append((df.iloc[i].loc['File Name'],
                    df.iloc[non_mated_row].loc['File Name']))

    knms_df = pd.DataFrame(knms).sample(size)
    knms_df = knms_df.reset_index(drop=True)
    knms_df.rename(columns={0: 'q', 1: 'k'}, inplace=True)
    knms_df.to_csv(name, index=False)


def _create_km_pairs_for_baseline(df, name: str):
    '''
    Inputs:
        - df: (Pandas DataFrame) The metadata dataframe
        - name: (str) The name of the file you want to output to

    For Baseline: Same shoe number, same foot, same size, both visit number 
                  has to be 1. 
                  We don't require image number to be the same.
    '''

    kms = []

    df = df.sort_values(by='Shoe Number')

    for i in range(len(df)):
        print(i)
        for j in range(i+1, len(df)):
            if (df.iloc[i].loc['Shoe Number'] == df.iloc[j].loc['Shoe Number'] and
                    str(df.iloc[i].loc['Visit Number']) == "1" and
                    str(df.iloc[j].loc['Visit Number']) == "1" and
                    df.iloc[i].loc['Foot'] == df.iloc[j].loc['Foot'] and
                    df.iloc[i].loc['Shoe Size'] == df.iloc[j].loc['Shoe Size']
                ):
                kms.append((df.iloc[i].loc['File Name'],
                           df.iloc[j].loc['File Name']))
                break

    kms_df = pd.DataFrame(kms)
    kms_df.rename(columns={0: 'q', 1: 'k'}, inplace=True)
    kms_df.to_csv(name, index=False)

def _create_knm_pairs_for_baseline(df, name: str, size: int = 5000):
    '''
    Inputs:
        - df: (Pandas DataFrame) The metadata dataframe
        - name: (str) The name of the file you want to output to
        - size: (int) The upper limit of how many KNM pairs to create

    For Baseline: we want all the KNMs to be within visit number 1
    '''
    knms = []

    df = df[df['Visit Number'] == 1]

    for i in range(len(df)):
        print(i)
        non_mated_row = np.random.randint(len(df))

        counter = 0
        found = True

        # We keep iterating until we find something that satisfy the KNM condition
        while (df.iloc[i].loc['File Name'] == df.iloc[non_mated_row].loc['File Name'] or
               str(df.iloc[non_mated_row].loc['Visit Number']) != "1" or
               df.iloc[i].loc['Shoe Number'] == df.iloc[non_mated_row].loc['Shoe Number'] or
               df.iloc[i].loc['Shoe Size'] != df.iloc[non_mated_row].loc['Shoe Size'] or
               df.iloc[i].loc['Shoe Make/Model'] != df.iloc[non_mated_row].loc['Shoe Make/Model'] or
               df.iloc[i].loc['Foot'] != df.iloc[non_mated_row].loc['Foot']):
            non_mated_row = np.random.randint(len(df))
            counter += 1

            if counter > 100:
                found = False
                break

        if found:
            knms.append((df.iloc[i].loc['File Name'],
                        df.iloc[non_mated_row].loc['File Name']))

    knms_df = pd.DataFrame(knms).sample(size)
    knms_df = knms_df.reset_index(drop=True)
    knms_df.rename(columns={0: 'q', 1: 'k'}, inplace=True)
    knms_df.to_csv(name, index=False)

def _create_km_pairs_for_blurry(df: pd.DataFrame, blur_level: int, name: str):
    pairs = []
    df = df.sort_values(by=['ShoeID', 'Blur_Level', 'Foot', 'Replicate'])

    for i in range(len(df)):
        for j in range(i+1, len(df)):
            if (df.iloc[i].loc['ShoeID'] == df.iloc[j].loc['ShoeID'] and
                df.iloc[i].loc['Size'] == df.iloc[j].loc['Size'] and
                df.iloc[i].loc['Foot'] == df.iloc[j].loc['Foot'] and
                df.iloc[i].loc['Replicate'] == df.iloc[j].loc['Replicate'] and
                df.iloc[i].loc['Blur_Level'] == 0 and
                df.iloc[j].loc['Blur_Level'] == blur_level):
                pairs.append((df.iloc[i].loc['File_Name'], df.iloc[j].loc['File_Name']))
                break

    pairs_df = pd.DataFrame(pairs)
    pairs_df.rename(columns={0: 'k', 1: 'q'}, inplace=True)
    pairs_df['mated'] = True
    pairs_df.to_csv(name, index = False)

def _create_knm_pairs_for_blurry(df: pd.DataFrame, blur_level: int, name: str, random_seed = 0):
    np.random.seed(random_seed)
    
    pairs = []
    
    df_k = df[df['Blur_Level'] == 0].copy()
    df_q = df[df['Blur_Level'] == blur_level].copy()
    
    df_k = df_k.sort_values(by=['ShoeID', 'Foot', 'Replicate']).reset_index(drop=True)
    df_q = df_q.sort_values(by=['ShoeID', 'Foot', 'Replicate']).reset_index(drop=True)

    for i in range(len(df_k)):
        non_mated_row = np.random.randint(len(df_q))

        # We keep iterating until we find something that satisfy the KNM condition
        while (df_k.iloc[i]['ShoeID'] == df_q.iloc[non_mated_row]['ShoeID'] or
            df_k.iloc[i]['Size'] != df_q.iloc[non_mated_row]['Size'] or
            df_k.iloc[i]['Foot'] != df_q.iloc[non_mated_row]['Foot'] or
            df_k.iloc[i]['Replicate'] != df_q.iloc[non_mated_row]['Replicate']):
            non_mated_row = np.random.randint(len(df_q))

        pairs.append((df_k.iloc[i]['File_Name'], df_q.iloc[non_mated_row]['File_Name']))

    pairs_df = pd.DataFrame(pairs)
    pairs_df.rename(columns={0: 'k', 1: 'q'}, inplace=True)
    pairs_df['mated'] = False
    pairs_df.to_csv(name, index=False)

def create_km_pairs_OOD(df, name: str):
    '''
    Creating known-mated pairs for out of distribution data. Mated pairs were 
    constructed with the same person, same shoe number, same left and right shoe
    and different replicate.
    '''
    kms = []
    
    df = df.sort_values(by = 'Shoe Number')
    for i in range(len(df)):
        for j in range(i+1,len(df)):
            if (df.iloc[i].loc['Shoe Number'] == df.iloc[j].loc['Shoe Number'] and
                df.iloc[i].loc['Foot'] == df.iloc[j].loc['Foot'] and
                df.iloc[i].loc['Image Number'] == df.iloc[j].loc['Image Number'] and
                df.iloc[i].loc['Visit Number'] == df.iloc[j].loc['Visit Number'] and
                df.iloc[i].loc['Shoe Size'] == df.iloc[j].loc['Shoe Size']
            ) :
                kms.append((df.iloc[i].loc['File Name'], df.iloc[j].loc['File Name']))
                break

    kms_df = pd.DataFrame(kms)
    kms_df.rename(columns = {0: 'q', 1 : 'k'}, inplace=True)
    kms_df.to_csv(name, index=False)

def create_knm_pairs_OOD(df, name: str, size: int = 2407):
    '''
    Creating known-nonmated pairs for out of distribution data. Nonmated pairs
    were constructed with the same person, same shoe number, and different left
    and right shoes. One shoe will be flipped.
    '''
    knms = []

    for i in range(len(df)):
        non_mated_row = np.random.randint(len(df))

        while (df.iloc[i].loc['File Name'] == df.iloc[non_mated_row].loc['File Name'] or
               df.iloc[i].loc['Shoe Number'] == df.iloc[non_mated_row].loc['Shoe Number'] or
               df.iloc[i].loc['Shoe Size'] != df.iloc[non_mated_row].loc['Shoe Size'] or
               df.iloc[i].loc['Shoe Make/Model'] != df.iloc[non_mated_row].loc['Shoe Make/Model'] or
               df.iloc[i].loc['Foot'] != df.iloc[non_mated_row].loc['Foot']):
            non_mated_row = np.random.randint(len(df))
        
        knms.append((df.iloc[i].loc['File Name'], df.iloc[non_mated_row].loc['File Name']))

    knms_df = pd.DataFrame(knms).sample(size)
    knms_df = knms_df.reset_index(drop=True)
    knms_df.rename(columns = {0: 'q', 1 : 'k'}, inplace=True)
    knms_df.to_csv(name, index=False)


def _create_km_pairs_for_diff_times(df, 
                                    name: str, 
                                    K_visit_number: str = "2"):
    '''
    Inputs:
        - df: (Pandas DataFrame) The metadata dataframe
        - name: (str) The name of the file you want to output to
        - K_visit_number: (str) "2" or "3" depending on the user
    '''

    kms = []

    df = df.sort_values(by='Shoe Number')

    for i in range(len(df)):
        print(i)
        for j in range(i+1, len(df)):
            if (df.iloc[i].loc['Shoe Number'] == df.iloc[j].loc['Shoe Number'] and
                    str(df.iloc[i].loc['Visit Number']) == "1" and
                    str(df.iloc[j].loc['Visit Number']) == K_visit_number and
                    df.iloc[i].loc['Foot'] == df.iloc[j].loc['Foot'] and
                    df.iloc[i].loc['Shoe Size'] == df.iloc[j].loc['Shoe Size']
                ):
                kms.append((df.iloc[i].loc['File Name'],
                           df.iloc[j].loc['File Name']))
                break

    kms_df = pd.DataFrame(kms)
    kms_df.rename(columns={0: 'q', 1: 'k'}, inplace=True)
    kms_df.to_csv(name, index=False)


def _create_knm_pairs_for_diff_times(df, 
                                     name: str, 
                                     size: int = 5000, 
                                     K_visit_number: str = "2"):
    '''
    Inputs:
        - df: (Pandas DataFrame) The metadata dataframe
        - name: (str) The name of the file you want to output to
        - size: (int) The upper limit of how many KNM pairs to create
        - K_visit_number: (str) "2" or "3" depending on the user
    '''
    knms = []

    df[df['Visit Number'].isin([1, int(K_visit_number)])]

    for i in range(len(df)):
        print(i)
        non_mated_row = np.random.randint(len(df))

        counter = 0
        found = True

        # We keep iterating until we find something that satisfy the KNM condition
        while (df.iloc[i].loc['File Name'] == df.iloc[non_mated_row].loc['File Name'] or
               str(df.iloc[i].loc['Visit Number']) != "1" or
               str(df.iloc[non_mated_row].loc['Visit Number']) != K_visit_number or
               df.iloc[i].loc['Shoe Number'] == df.iloc[non_mated_row].loc['Shoe Number'] or
               df.iloc[i].loc['Shoe Size'] != df.iloc[non_mated_row].loc['Shoe Size'] or
               df.iloc[i].loc['Shoe Make/Model'] != df.iloc[non_mated_row].loc['Shoe Make/Model'] or
               df.iloc[i].loc['Foot'] != df.iloc[non_mated_row].loc['Foot']):
            non_mated_row = np.random.randint(len(df))
            counter += 1

            if counter > 100:
                found = False
                break

        if found:
            knms.append((df.iloc[i].loc['File Name'],
                        df.iloc[non_mated_row].loc['File Name']))

    knms_df = pd.DataFrame(knms).sample(size)
    knms_df = knms_df.reset_index(drop=True)
    knms_df.rename(columns={0: 'q', 1: 'k'}, inplace=True)
    knms_df.to_csv(name, index=False)