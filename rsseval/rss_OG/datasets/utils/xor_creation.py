import os
import torch
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np, glob
import re
import matplotlib.pyplot as plt
import joblib
from torchvision.datasets.folder import pil_loader


class XORDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path,
        split,
        c_sup=1,
        which_c=[-1],
    ):
        
        self.base_path = base_path
        self.split = split

        # collecting images
        self.list_images = glob.glob(os.path.join(self.base_path, self.split, "*.png"))

        # sort the images
        self.list_images = sorted(self.list_images, key=self._extract_number)

        # ok transform
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        self.labels, self.concepts = [], []

        # lmao
        new_images = self.list_images.copy()

        # extract labels and concepts
        for idx, item in enumerate(self.list_images):
            name = os.path.splitext(os.path.basename(item))[0]
            # extract the ids out of the images
            meta_id = name.split("_")[-1]

            # get the target meta
            meta_scene = os.path.join(
                self.base_path,
                self.split,
                str(meta_id) + ".joblib",
            )

            if not os.path.exists(meta_scene):
                new_images.remove(
                    os.path.join(self.base_path, self.split, str(meta_id) + ".joblib")
                )
                continue


            # concepts and labels
            concepts, labels = [], []

            # load data from joblib
            data = joblib.load(meta_scene)

            # take the label
            label = data["label"]
            concept_values = data["meta"]["concepts"]

            labels = np.array(label)
            self.labels.append(labels)

            concepts = np.array(concept_values)
            self.concepts.append(concepts)
            # if idx >= 995:
            #     print(item, label, concept_values)
            # if idx >= 1005:
            #     quit()
#MODFIED HERE
        # self.concepts = np.stack(self.concepts, axis=0)
        # self.labels = np.stack(self.labels, axis=0)
#WITH THIS
        if self.concepts:  # Check if the list is not empty
            self.concepts = np.stack(self.concepts, axis=0)
        else:
            print("Warning: No concept data available to stack.")

        if self.labels:  # Check if the list is not empty
            self.labels = np.stack(self.labels, axis=0)
        else:
            print("Warning: No label data available to stack.")
            self.list_images = np.array(new_images)

    def _extract_number(self, path):
        match = re.search(r"\d+", path)
        return int(match.group()) if match else 0

    def __getitem__(self, item):

        labels = self.labels[item]
        concepts = self.concepts[item]
        img_path = self.list_images[item]
        image = pil_loader(img_path)

        # grayscale
        image = image.convert("L")
        
    
        #print("Shape dell' immagine (H, W):", (image.size[1], image.size[0]))

        return self.transform(image), labels, concepts

    def __len__(self):
        return len(self.list_images)


if __name__ == "__main__":
    print("Hello World")

    train_data = XORDataset("../../data/xor_out_bits", "train")
    val_data = XORDataset("../../data/xor_out_bits", "val")
    test_data = XORDataset("../../data/xor_out_bits", "test")
    ood_data = XORDataset("../../data/xor_out_bits", "ood")

    img, label, concepts = train_data[0]
    print(img.shape, concepts.shape, label.shape)

    plt.imshow(img.permute(1, 2, 0))
    plt.savefig("lmao.png")
    plt.close()
    quit()
