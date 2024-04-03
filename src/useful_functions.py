import pandas as pd
import numpy as np
from tqdm import tqdm
import json
import yaml
import re
import matplotlib.pyplot as plt

# Create empty dictionary but with correct shape
def empty_dict(tokens):
    dico = {}
    for col in list(tokens.keys()):
        dico[col] = None
    return dico
    
# Split to extract information between delimiters
def split_each_token(line, token_toward_col):
    tokens_toward_split = list(token_toward_col.keys())
    pattern = '(' + '|'.join(re.escape(token) for token in tokens_toward_split) + ')'

    # Seak information between delimiters
    splits = re.split(pattern, line)
    # Filter out empty strings
    splits = [piece.strip() for piece in splits if piece]
    return splits

# Get the dictionnary with the extracted information
def split_to_dict(split, tokens, token_toward_col, dict_split=None):
    # If no dictionnary passed, I create an empty one 
    if dict_split is None:
        dict_split = empty_dict(tokens)
    # I create the dictionnary with the correct structure
    for i in range(0, len(split), 2):
        element = split[i+1]
        token = split[i]
        column = token_toward_col[token]
        dict_split[column] = element
    return dict_split

def get_preprocessing_done(data, tokens,  token_dict):
    df_dict = {}
    iteration = 0
    for key in tqdm(data.keys()): 
        for line in data[key].split('\n'):
            try:
                split = split_each_token(line, token_dict)
                split_dict = split_to_dict(split, tokens,  token_dict)
                df_dict[iteration] = split_dict
                iteration += 1
            except:
                pass
    # From the dictionnary I create a dataframe
    df = pd.DataFrame().from_dict(df_dict, orient='index').fillna(value=np.nan)

    # Certain lines are empty
    indices_to_remove = []
    for i in range(len(df)):
        if np.all(df.iloc[i].isna()):
            indices_to_remove.append(i)
    
    df = df.loc[~df.index.isin(indices_to_remove)]

    return df

def plot_histogramm(df, features):
    # Plot histogram
    list_age = []
    for age in df[features].dropna():
        try :
            list_age.append(int(age[:2]))
        except:
            pass
    # Plot histogram with percentages
    plt.hist(list_age, bins=10, color='skyblue', edgecolor='black')
    plt.xlabel('Age')
    plt.ylabel('Percentage')
    plt.title('Distribution of Age')
    plt.grid(True)
    
    # Calculate the percentage values
    counts, bins, _ = plt.hist(list_age, bins=10, color='skyblue', edgecolor='black', density=True)
    percentage = counts / sum(counts) * 100
    counts, bins, _ = plt.hist(list_age, bins=10, color='skyblue', edgecolor='black')
    
    # Show the percentages on top of each bar
    for i in range(len(bins) - 1):
        plt.text(bins[i] + 5, counts[i] , f'{percentage[i]:.1f}%', ha='center', va='bottom')
    
    plt.show()

def plot_pie_chart(df, features, title, manuel_label=None, labels=None):
    # Calculate value counts for each category
    category_counts = df[features].value_counts()
    
    # Plot pie chart
    plt.figure(figsize=(8, 6))
    if manuel_label:
        plt.pie(category_counts, labels=labels, autopct='%1.1f%%', startangle=140)
    else:
        plt.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', startangle=140)
    plt.title(title)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.show()
