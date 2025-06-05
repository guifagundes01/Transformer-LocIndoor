from collections import defaultdict
import json
import re

with open("data/results.txt") as fp:
    raw_data = fp.read()

def nested_dict():
    return defaultdict(nested_dict)

train = nested_dict()
val = nested_dict()
test = nested_dict()

block_pattern = re.compile(r"Source dim: (\d+).*?Embedding Dim: (\d+).*?Hidden Size: (\d+).*?Train RMSE: ([\d.]+).*?Val RMSE: ([\d.]+).*?MAE: ([\d.]+)", re.DOTALL)
for match in block_pattern.finditer(raw_data):
    source_dim = int(match[1])
    embedding_dim = int(match[2])
    hidden_size = int(match[3])
    train_rmse = float(match[4])
    val_rmse = float(match[5])
    test_rmse = float(match[6])

    train[source_dim][embedding_dim][hidden_size] = train_rmse
    val[source_dim][embedding_dim][hidden_size] = val_rmse
    test[source_dim][embedding_dim][hidden_size] = test_rmse

train_dict = json.loads(json.dumps(train))
val_dict = json.loads(json.dumps(val))
test_dict = json.loads(json.dumps(test))

with open('data/train.json', 'w') as fp:
    json.dump(train_dict, fp)

with open('data/val.json', 'w') as fp:
    json.dump(val_dict, fp)

with open('data/test.json', 'w') as fp:
    json.dump(test_dict, fp)
