import pandas as pd 
import numpy as np
import re 
import nltk 
import torch.nn as nn
import pickle
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Any
import os
from transformers import AutoTokenizer
import string
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
stwords = list(set(stopwords.words("english")))


PATH_TO_SAVE_PREP_DATASET = "/Users/apple/Desktop/CaseStudy/task_1/emotion_detection/dataset/"
class preprocessing:
    def __init__(self, filepath: str) -> None:
        """
            Preprocessing of the emotion dataset, to make it ready for training
        """

        self.filepath = filepath
        self.dataset = pd.read_csv(self.filepath)
        self.dataset = self.dataset[["sentiment", "content"]]
        self.dataset["content_filtered"] = self.dataset["content"].apply(self._preprocess_data)
        self.dataset.drop("content", axis=1, inplace=True)

        self.unqiue_labels = list(set(self.dataset["sentiment"].values.tolist()))
        self.label2index = {value: key for key, value in enumerate(self.unqiue_labels)}
        self.index2label = {value: key for key, value in self.label2index.items()}

        # save the dictionary
        self._save_pickle(os.path.join(PATH_TO_SAVE_PREP_DATASET, "label2index.pickle"), self.label2index)
        self._save_pickle(os.path.join(PATH_TO_SAVE_PREP_DATASET, "index2label.pickle"), self.index2label)
        
    
    def _save_pickle(self, path, object_to_save):
        with open(path, "wb") as fp:
            pickle.dump(object_to_save, fp)
        

    def _load_pickle(self, path):
        return pickle.load(open(path, "rb"))



    def _preprocess_data(self, data):
        data = re.sub(r"http\S+", '', data) # remove all the links which start with http
        data = re.sub(r"www\s+", "", data) # all the links if it's start with www
        # removing the punctuation
        data = data.translate(str.maketrans('', '',string.punctuation))
        # remove the stop words
        splitted_data = data.lower().split()
        filtered_data = list(filter(lambda x: x not in stwords, splitted_data))
        data = " ".join(filtered_data)
        return data

class crossvalidationdata(preprocessing):
    def __init__(self, filepath: str) -> None:
        """
            I am simply using the train_test_split instead of kFold.
        """
        super().__init__(filepath)
        

        self.train_file_path = os.path.join(PATH_TO_SAVE_PREP_DATASET, "train.csv")
        self.dev_file_path   = os.path.join(PATH_TO_SAVE_PREP_DATASET, "dev.csv")

        if not (os.path.isfile(self.train_file_path) and os.path.isfile(self.dev_file_path)):
            train, test = train_test_split(self.dataset, test_size=0.3, random_state=42)
            train.to_csv(self.train_file_path, index=False)
            test.to_csv(self.dev_file_path, index=False)


class dataset(Dataset):
    """
        1. This is the pytorch custom dataset class which will return the tokenized output, along with labels.
        These output is than passed to the model for fine-tuning.
    
    """
    def __init__(self, filePath: str, label2index: Dict) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
        self.filePath = filePath
        self.label2index = label2index
        # read the dataset 
        self.df_data = pd.read_csv(self.filePath)

    def __getitem__(self, index) -> Any:
        row = self.df_data.iloc[index]
  
        text, label = row["content_filtered"], row["sentiment"]
        tokenized = self.tokenizer(text, max_length=256, padding="max_length", truncation=True, return_tensors="pt")
        tokenized["labels"] = self.label2index[label]
        return tokenized


    def __len__(self):
        return self.df_data.shape[0]


