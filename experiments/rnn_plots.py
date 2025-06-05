import json
import matplotlib.pyplot as plt

with open("data/train.json") as fp:
    train = json.load(fp)

with open("data/val.json") as fp:
    val = json.load(fp)

with open("data/test.json") as fp:
    test = json.load(fp)

def plot_data(src_dim, datasets):
    plt.figure(figsize=(8, 6))

    colors = {
        "64": 'r',
        "128": 'g',
        "256": 'b'
    }
    markers = {
        "train": 'o',
        "val": 's',
        "test": '^'
    }

    for name, data in datasets:
        for emb, color in colors.items():
            data_sd_emb = data[src_dim][emb]
            x = [int(k) for k in data_sd_emb.keys()]
            y = [data_sd_emb[k] for k in data_sd_emb]
            plt.plot(x, y, marker=markers[name], color=color, label=f"{name}: {emb}")

    plt.xlabel("Hidden size")
    plt.ylabel("MAE (m)")
    plt.xticks([8, 16, 32, 64])
    plt.grid(True, alpha=1.0)
    plt.legend(title="Embedding dim", framealpha=1.0)
    plt.tight_layout()
    plt.savefig(f"figure/error_{src_dim}.png", bbox_inches='tight', format="png")
    plt.show()


datasets = [
        ("train", train),
        ("val", val),
        ("test", test)
]

plot_data("15", datasets)
plot_data("20", datasets)
