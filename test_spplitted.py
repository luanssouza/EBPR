import torch
import pandas as pd
import argparse

from Code.EBPR_model import BPR

DATA_DIR = {
    'ml-100k': 'Data/ml100k/',
    'ml-1m': 'Data/ml1m/'
}

def resume_checkpoint(model, model_dir, device_id):
    state_dict = torch.load(model_dir)#,
                            # map_location=lambda storage, loc: storage.cuda(device=device_id))  # ensure all storage are on gpu
    model.load_state_dict(state_dict)

def read_train_test(dataset):
    train_df = pd.read_csv(DATA_DIR[dataset] + 'train.csv', header=None)
    test_df = pd.read_csv(DATA_DIR[dataset] + 'test.csv', header=None)
    all_df = pd.concat([train_df, test_df], ignore_index=True)

    ## mapping ids for model prediction
    mapped_df = pd.read_csv('Output/' + dataset + '_mapped_dataset.csv')
    user_mapping = { int(row['uid']): int(row['userId']) for _, row in mapped_df.drop_duplicates(subset=['uid'], keep='first').reindex().iterrows() }
    item_mapping = { int(row['mid']): int(row['itemId']) for _, row in mapped_df.drop_duplicates(subset=['mid'], keep='first').reindex().iterrows() }

    all_df = all_df[all_df[0].isin(user_mapping)]
    all_df = all_df[all_df[1].isin(item_mapping)]
    all_df['userId'] = all_df[0].apply(lambda x: user_mapping[int(x)])
    all_df['itemId'] = all_df[1].apply(lambda x: item_mapping[int(x)])

    return train_df, test_df, all_df

def load_model(num_users, num_items, model_dir):
    config = {
        'num_users': num_users,
        'num_items': num_items,
        'num_latent': 50,  # Number of latent factors.
        'loo_eval': False
        }

    model = BPR(config)

    resume_checkpoint(model, model_dir, 0)

    return model

def main(dataset, model_dir):
    train_df, _, all_df = read_train_test(dataset)
    num_users = len(train_df[0].unique())
    num_items = len(train_df[1].unique())
    print("Dataset:", dataset)
    print("Number of users:", num_users)
    print("Number of items:", num_items)

    model = load_model(num_users, num_items, 'Output/checkpoints/' + model_dir)
    model_name = model_dir.split("_")[0]
    print("Model name:", model_name)
    print("Model:", model)

    predictions = []
    for index, interactions in all_df.groupby(0):
        if index > num_users:
            continue
        for _, row in interactions.iterrows():
            if row[1] > num_items:
                continue
            if row[0] < len(all_df[0].unique()):
                user_latent = model.embed_user(torch.tensor(row[0] - 1))
                item_latent = model.embed_item(torch.tensor(row[1] - 1))
                prediction = (user_latent * item_latent).sum(dim=-1)
                predictions.append({
                    "uid": row[0],
                    "iid": row[1],
                    "original": row[2],
                    "prediction": float(prediction)
                })
    output_file = "Results/" + model_name + "_" + dataset + "_predictions.csv"
    print("Output file:", output_file)
    pd.DataFrame(predictions).to_csv(output_file, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script.")
    parser.add_argument("--model_dir", type=str, help="Model to test: 'BPR', 'UBPR', 'EBPR', 'pUEBPR', 'UEBPR'.")
    parser.add_argument("--dataset", type =str, default='ml-100k', help="'ml-100k' for Movielens 100K. 'ml-1m' for "
                                                                        "the Movielens 1M dataset. 'lastfm-2k' for "
                                                                        "the Last.FM 2K dataset. 'yahoo-r3' for the "
                                                                        "Yahoo! R3 dataset.")
    args = parser.parse_args()

    model_dir = args.model_dir
    dataset = args.dataset

    main(dataset, model_dir)