import pandas as pd
import numpy as np
from tqdm import tqdm
import json
import yaml
import re
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.inspection import permutation_importance
import time

import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

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
    plt.ylabel('Effectif')
    plt.title('Pyramide des âges')
    plt.grid(True)
    
    # Calculate the percentage values
    counts, bins, _ = plt.hist(list_age, bins=10, color='skyblue', edgecolor='black', density=True)
    percentage = counts / sum(counts) * 100
    counts, bins, _ = plt.hist(list_age, bins=10, color='skyblue', edgecolor='black')
    
    # Show the percentages on top of each bar
    for i in range(len(bins) - 1):
        plt.text(bins[i] + 5, counts[i] , f'{percentage[i]:.1f}%', ha='center', va='bottom')

    plt.savefig("histo.png")
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
    plt.savefig(title + ".png")
    plt.show()

def get_report(clf, X_train_enc, X_test_enc,  y_train_enc, y_test_enc, description_features_string, description_features_num, target_features):
    start = time.time()
    y_pred = clf.predict(X_test_enc)
    stop = time.time()
    print(f"Inference Time: {round(stop - start,3)}s")
    print("Notre accuracy de test est de : " + str(accuracy_score(y_test_enc, y_pred)))

    print("Classification report")
    print(classification_report(y_test_enc, y_pred, zero_division=0.))

    disp = ConfusionMatrixDisplay.from_predictions(y_test_enc, y_pred)
    fig = disp.figure_
    fig.set_figwidth(5)
    fig.set_figheight(5) 
    fig.show()

    #Let's find our which features contributes the most to the diagnosis
    result = permutation_importance(
        clf, X_test_enc, y_test_enc, n_repeats=10, random_state=42, n_jobs=2)
    forest_importances = pd.Series(result.importances_mean,
                                    index=description_features_num+
                                   description_features_string)
    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
    ax.set_title("Feature importances using permutation on full model")
    ax.set_ylabel("Mean accuracy decrease")
    fig.tight_layout()
    plt.show()

def get_report_hugging_face_model(device, model, tokenizer, X_test, y_test):
    # Preprocess the test data
    tokenized_texts_test = tokenizer(np.array(X_test).tolist(), padding=True, truncation=True, return_tensors='pt', max_length=512, add_special_tokens=True)
    labels_test = torch.tensor(np.array(y_test))
    
    # Create DataLoader for the test dataset
    test_dataset = TensorDataset(tokenized_texts_test['input_ids'], tokenized_texts_test['attention_mask'], labels_test)
    test_dataloader = DataLoader(test_dataset, batch_size=4)
    
    # Evaluate the model
    model.eval()
    predicted_labels = []
    true_labels = []
    
    start = time.time()
    for batch in test_dataloader:
        batch = tuple(t.to(device) for t in batch)
        inputs = {'input_ids': batch[0], 'attention_mask': batch[1]}
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        predicted_labels.extend(torch.argmax(logits, dim=1).cpu().numpy())
        true_labels.extend(batch[2].cpu().numpy())
    
    stop = time.time()
    print(f"Inference time: {round(stop - start,3)}s")
    
    # Calculate evaluation metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    print("Accuracy:", accuracy)
    
    print("Classification report")
    print(classification_report(true_labels, predicted_labels, zero_division=0.))
    
    disp = ConfusionMatrixDisplay.from_predictions(true_labels, predicted_labels)
    fig = disp.figure_
    fig.set_figwidth(5)
    fig.set_figheight(5) 
    fig.show()

