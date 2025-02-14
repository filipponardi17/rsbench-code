import os
import joblib

def count_labels(folder):
    true_count = 0
    false_count = 0
    
    for file in os.listdir(folder):
        if file.endswith(".joblib"):
            data = joblib.load(os.path.join(folder, file))
            label = data["label"]
            if label is True:
                true_count += 1
            elif label is False:
                false_count += 1
    
    return true_count, false_count

base_folder = "/home/filippo.nardi/rsbench-code/rssgen/FIX1_MNIST_LOGIC_OUT_FOLDER"  # Sostituisci con il percorso corretto

for split in ["train", "val", "test"]:
    folder = os.path.join(base_folder, split)
    if os.path.exists(folder):
        true_count, false_count = count_labels(folder)
        print(f"Nel sottoinsieme '{split}': True={true_count}, False={false_count}")
    else:
        print(f"Il sottoinsieme '{split}' non esiste.")