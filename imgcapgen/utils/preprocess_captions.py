# imgcapgen/utils/preprocess_captions.py

"""
Utility to preprocess raw captions file and convert into a structured CSV.

Typical usage:
    from imgcapgen.utils.preprocess_captions import preprocess_captions
    preprocess_captions("flickr8k/captions.txt", "outputs/flickr8k_captions.csv")
"""

import pandas as pd

def preprocess_captions(caption_file, csv_file):
    """
    Process the raw captions.txt file and save a structured CSV.

    Args:
        caption_file (str or Path): Path to captions.txt
        csv_file (str or Path): Path to output CSV file
    """
    df = pd.read_csv(
        caption_file,
        header=None,
        names=['image', 'caption'],
        sep=',',
        quotechar='"',
        skiprows=1
    )
    
    
    # Remove any leading spaces from the caption
    df['caption'] = df['caption'].str.lstrip()

    # Fill NaNs using other captions for the same image
    def fill_caption(group):
        valid_captions = group['caption'].dropna().tolist()
        return group['caption'].apply(lambda x: valid_captions[0] if pd.isna(x) and valid_captions else x)

    df['caption'] = df.groupby('image').apply(fill_caption).reset_index(level=0, drop=True)

    # Number captions for each image: 0,1,2,3...
    df['caption_number'] = df.groupby('image').cumcount()

    # Create unique id for each image filename
    df['id'] = df['image'].factorize()[0]

    # Reorder columns
    df = df[['image', 'caption_number', 'caption', 'id']]

    # Save
    df.to_csv(csv_file, index=False)
    print(f"✅ Saved preprocessed captions to {csv_file}")

    return df
