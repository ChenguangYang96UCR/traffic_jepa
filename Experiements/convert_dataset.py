import numpy as np

def convert_split(input_file, output_file):
    samples = np.load(input_file, allow_pickle=True)

    x = np.stack([
        sample["x_data"][..., 0:1]
        for sample in samples
    ]).astype(np.float32)

    y = np.stack([
        sample["y_data"][..., 0:1]
        for sample in samples
    ]).astype(np.float32)

    np.savez_compressed(output_file, x=x, y=y)

    print("x:", x.shape)
    print("y:", y.shape)


for split in ["train", "val", "test"]:
    convert_split(
        f"../TopoJEPA/dataset/Fremont/incident_{split}.npy",
        f"../TopoJEPA/dataset/Fremont/{split}.npz",
    )

import pickle
import numpy as np

adj = np.load(
    "../../TopoJEPA/dataset/Fremont/adj_matrix.npy"
).astype(np.float32)
assert adj.shape == (93, 93)

sensor_ids = [str(i) for i in range(93)]
sensor_id_to_ind = {
    sensor_id: i for i, sensor_id in enumerate(sensor_ids)
}

with open("data/FREMONT/adj_mx.pkl", "wb") as f:
    pickle.dump(
        [sensor_ids, sensor_id_to_ind, adj],
        f,
        protocol=2,
    )