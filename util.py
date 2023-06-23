import pandas as pd
import numpy as np

METADATA = pd.read_csv("2DScanImages/Image-Info.csv")


def _create_km_pairs():
    '''
    Documenting Known Mated Pairs data process
    '''
    df = METADATA.sort_values('Shoe Number')

    kms = []

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
    kms_df.to_csv("KM_pairs.csv")

def _create_knm_pairs(size: int = 2407):
    '''
    Documenting Known Non-Mated Pairs data process
    '''
    df = METADATA
    knms = []

    for i in range(len(df)):
        non_mated_row = np.random.randint(len(df))
        while (df.iloc[i].loc['File Name'] == df.iloc[non_mated_row].loc['File Name'] or
            (df.iloc[i].loc['Shoe Number'] == df.iloc[non_mated_row].loc['Shoe Number'] and
                (df.iloc[i].loc['Image Number'] == df.iloc[non_mated_row].loc['Image Number'] or # same size same model
                df.iloc[i].loc['Visit Number'] == df.iloc[non_mated_row].loc['Visit Number']))):
            non_mated_row = np.random.randint(len(df))
        knms.append((df.iloc[i].loc['File Name'], df.iloc[non_mated_row].loc['File Name']))

    knms_df = pd.DataFrame(knms).sample(size)
    knms_df = knms_df.reset_index(drop=True)
    knms_df.rename(columns = {0: 'q', 1 : 'k'}, inplace=True)
    knms_df.to_csv("KNM_pairs.csv")