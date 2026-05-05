from recoxplainer.config import cfg
from recoxplainer.recommender import Recommender
from recoxplainer.data_reader.data_reader import DataReader
from recoxplainer.explain import ARPostHocExplainer, KNNPostHocExplainer

import pandas as pd

import warnings
import logging
import torch
import yaml

import time
import argparse
from pathlib import Path

from Code.EBPR_model import BPR

def read_mapping(mapping_path):
    ## mapping ids for model prediction
    mapped_df = pd.read_csv(mapping_path)
    user_mapping = { int(row['uid']): int(row['userId']) for _, row in mapped_df.drop_duplicates(subset=['uid'], keep='first').reindex().iterrows() }
    item_mapping = { int(row['mid']): int(row['itemId']) for _, row in mapped_df.drop_duplicates(subset=['mid'], keep='first').reindex().iterrows() }
    return user_mapping, item_mapping


class CustomModel(torch.nn.Module):

    def __init__(self, model_path, recoxplainer_dataset, user_mapping, item_mapping):
        
        super().__init__()
        
        self.model = self.load_model(len(user_mapping.keys()), len(item_mapping.keys()), model_path)
        self.recoxplainer_dataset = recoxplainer_dataset
        self.user_mapping = user_mapping
        self.item_mapping = item_mapping
        # print(self.user_mapping, self.item_mapping)
        print(len(user_mapping.keys()), len(item_mapping.keys()))

    def resume_checkpoint(self, model, model_dir):
        state_dict = torch.load(model_dir)
        model.load_state_dict(state_dict)
    
    def load_model(self, num_users, num_items, model_dir):
        config = {
            'num_users': num_users, 'num_items': num_items,
            'num_latent': 50, 'loo_eval': False
            }
        model = BPR(config)
        self.resume_checkpoint(model, model_dir)
        return model
    
    def predict(self, user_id, item_id):
        uid = self.recoxplainer_dataset.original_user_id.loc[user_id]["user_id"]
        iid = [self.recoxplainer_dataset.original_item_id.loc[i]["item_id"] for i in item_id]

        predictions = []

        for i in iid:
            if uid not in self.user_mapping.keys() or i not in self.item_mapping.keys():
                predictions.append(0.0)
                continue
            user_latent = self.model.embed_user(torch.tensor(self.user_mapping[uid]))
            item_latent = self.model.embed_item(torch.tensor(self.item_mapping[i]))
            prediction = (user_latent * item_latent).sum(dim=-1)
            predictions.append(float(prediction))
        # print(predictions)
        # print(user_id, item_id)
        # print(uid, iid)
        # print(self.user_mapping)
        # print(self.item_mapping)
        return predictions

def post_hoc_ar(model, rec, data):
    star_time = time.time()
    print("Starting ARPostHocExplanation at:", star_time)

    recommmendations = rec.copy()
    explainer = ARPostHocExplainer(model, recommmendations, data,
                            min_support=.01,
                            max_len=2,
                            metric="lift",
                            min_threshold=.1,
                            min_confidence=.1,
                            min_lift=.1)
    # https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/apriori/
    explanations = explainer.explain_recommendations()

    end_time = time.time()
    print("Finishing ARPostHocExplanation at:", end_time)
    print('It tooks', end_time-star_time, 'seconds.')
    return explanations

def post_hoc_knn(model, rec, data):
    star_time = time.time()
    print("Starting KNNPostHocExplanation at:", star_time)

    recommmendations = rec.copy()
    explainer = KNNPostHocExplainer(model, recommmendations, data)
    explanations = explainer.explain_recommendations()

    end_time = time.time()
    print("Finishing KNNPostHocExplanation at:", end_time)
    print('It tooks', end_time-star_time, 'seconds.')
    return explanations

def load_model(model_path, mapping_path, data_name=["ml100k"]):
    print("Model:", model_path)
    print("Data:", model_path)
    
    logging.disable()
    
    warnings.filterwarnings("ignore", category=FutureWarning)

    user_mapping, item_mapping = read_mapping(mapping_path)

    data = DataReader(**cfg[data_name])
    data.make_consecutive_ids_in_dataset(data.names)

    star_time = time.time()
    print("Starting Recommendation at:", star_time)

    custom_model = CustomModel(model_path, data, user_mapping, item_mapping)

    rec = Recommender(data, custom_model)
    rec = rec.recommend_all()

    end_time = time.time()
    print("Finishing Recommendation at:", end_time)
    print('It tooks', end_time-star_time, 'seconds.')
    return custom_model, rec, data


def recommendations_to_original_id(recs, data):
    explain = "explanations" in recs.columns
    rec_original = []
    for _, r in recs.iterrows():
        rec_sample = {}
        rec_sample["userId"] = data.original_user_id.loc[r["userId"]]["user_id"]
        rec_sample["itemId"] = data.original_item_id.loc[r["itemId"]]["item_id"]
        rec_sample["rank"] = r["rank"]
        if explain:
            rec_sample["explanations"] = {data.original_item_id.loc[e]["item_id"] for e in r["explanations"]}
        rec_original.append(rec_sample)
    return pd.DataFrame(rec_original)

# def save_recommendations_explanations(rec, exp_ar, exp_knn, data, output_path):
#     recommendations_to_original_id(rec, data).to_csv(output_path + 'recommendations.csv', index=False, header=False)
#     recommendations_to_original_id(exp_ar, data).to_csv(output_path + 'exp_ar.csv', index=False, header=False)
#     recommendations_to_original_id(exp_knn, data).to_csv(output_path + 'exp_knn.csv', index=False, header=False)

def save_recommendations_explanations(recommendations, data, output_path):
    recommendations_to_original_id(recommendations, data).to_csv(output_path, index=False, header=False)

parser = argparse.ArgumentParser()
parser.add_argument("--config_file", help="Path to the config YAML file")
args = parser.parse_args()

# Open the YAML file
with open(args.config_file, 'r') as file:
    datasets_config = yaml.safe_load(file)

for dataset in datasets_config:
    for model in dataset["models"]:
        output_path = dataset['output_path'] + '/' + model['model_name']
        Path(output_path).mkdir(parents=True, exist_ok=True)

        model_instance, rec, data = load_model(
            dataset['input_path'] + '/' + model['model_checkpoint_path'],
            dataset['mapping_path'],
            dataset['dataset_name']
            )
        save_recommendations_explanations(rec, data, output_path + 'recommendations.csv')

        exps = post_hoc_ar(model_instance, rec, data)
        save_recommendations_explanations(exps, data, output_path + 'exp_ar.csv')

        exps = post_hoc_knn(model_instance, rec, data)
        save_recommendations_explanations(exps, data, output_path + 'exp_knn.csv')

#         # %mkdir {recoxplainer_path}ml100k/BPR/
#         # save_recommendations_explanations(rec, exp_ar, exp_knn, data, f'{recoxplainer_path}ml100k/BPR/')
# # Create nested directories (similar to 'mkdir -p')
# # Path("path/to/nested/folder").mkdir(parents=True, exist_ok=True)
# # data_path = "../data/random/"
# # recoxplainer_path = f"{data_path}recoxplainer/"

# # %mkdir {recoxplainer_path}
# # %mkdir {recoxplainer_path}ml100k/

# model, rec, data = load_model('data/random/recbole/ml100k/saved/BPR-Apr-27-2026_03-55-18.pth', 'ml100k')
# exp_ar = post_hoc_ar(model, rec, data)
# exp_knn = post_hoc_knn(model, rec, data)
# print(exp_knn)
# # %mkdir {recoxplainer_path}ml100k/BPR/
# # save_recommendations_explanations(rec, exp_ar, exp_knn, data, f'{recoxplainer_path}ml100k/BPR/')