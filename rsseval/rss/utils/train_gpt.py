# Module which contains the code for training a model
import torch
import numpy as np


import csv
import os
from tqdm import tqdm

from torchvision.utils import make_grid

from utils.status import progress_bar
from datasets.utils.base_dataset import BaseDataset
from models.mnistdpl import MnistDPL
from utils.dpl_loss import ADDMNIST_DPL
from utils.metrics import (
    evaluate_metrics,
    evaluate_mix,
    mean_entropy,
    accuracy_binary,
)

from utils import fprint
import matplotlib.pyplot as plt

from warmup_scheduler import GradualWarmupScheduler
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix
import numpy as np


def convert_to_categories(elements):
    # Convert vector of 0s and 1s to a single binary representation along the first dimension
    binary_rep = np.apply_along_axis(
        lambda x: "".join(map(str, x)), axis=1, arr=elements
    )
    return np.array([int(x, 2) for x in binary_rep])


def entropy(p):
    """Compute entropy given a probability distribution."""
    p = np.clip(p, 1e-15, 1)

    return -np.sum(p * np.log(p)) / np.log(len(p))


def compute_coverage(confusion_matrix):
    """Compute the coverage of a confusion matrix.

    Essentially this metric is
    """

    max_values = np.max(confusion_matrix, axis=0)
    clipped_values = np.clip(max_values, 0, 1)

    # Redefinition of soft coverage
    coverage = np.sum(clipped_values) / len(clipped_values)

    return coverage


def plot_confusion_matrix(
    y_true, y_pred, labels=None, title="Confusion Matrix", save_path=None
):
    """
    Generate and plot a confusion matrix using Matplotlib with normalization.

    Parameters:
        y_true (array-like): Ground truth labels.
        y_pred (array-like): Predicted labels.
        labels (array-like, optional): List of class labels (default: None).
        title (str, optional): Title of the plot (default: 'Confusion Matrix').
        save_path (str, optional): Path to save the plot image (default: None).
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Normalize confusion matrix
    cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()

    if labels is not None:
        tick_marks = np.arange(len(labels))
        plt.xticks(tick_marks, labels)
        plt.yticks(tick_marks, labels)

    plt.tight_layout()
    plt.ylabel("True Labels")
    plt.xlabel("Predicted Labels")

    if save_path is not None:
        print("Saved", save_path)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.close()

    return cm


def plot_multilabel_confusion_matrix(
    y_true, y_pred, class_names, title, save_path=None
):
    y_true_categories = convert_to_categories(y_true.astype(int))
    y_pred_categories = convert_to_categories(y_pred.astype(int))

    to_rtn_cm = confusion_matrix(y_true_categories, y_pred_categories)

    cm = multilabel_confusion_matrix(y_true, y_pred)
    num_classes = len(class_names)
    num_rows = (num_classes + 4) // 5  # Calculate the number of rows needed

    plt.figure(figsize=(20, 4 * num_rows))  # Adjust the figure size

    for i in range(num_classes):
        plt.subplot(num_rows, 5, i + 1)  # Set the subplot position
        plt.imshow(cm[i], interpolation="nearest", cmap=plt.cm.Blues)
        plt.title(f"Class: {class_names[i]}")
        plt.colorbar()
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ["0", "1"])
        plt.yticks(tick_marks, ["0", "1"])

        fmt = ".0f"
        thresh = cm[i].max() / 2.0
        for j in range(cm[i].shape[0]):
            for k in range(cm[i].shape[1]):
                plt.text(
                    k,
                    j,
                    format(cm[i][j, k], fmt),
                    ha="center",
                    va="center",
                    color="white" if cm[i][j, k] > thresh else "black",
                )

        plt.ylabel("True label")
        plt.xlabel("Predicted label")

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.suptitle(title)

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

    plt.close()

    return to_rtn_cm


def plot_actions_confusion_matrix(c_true, c_pred, title, save_path=None):

    # Define scenarios and corresponding labels
    scenarios = {
        "forward": [slice(0, 3), slice(0, 3)],
        "stop": [slice(3, 9), slice(3, 9)],
        #'forward_stop': [slice(None, 9), slice(None, 9)],
        "left": [slice(9, 15), slice(9, 15)],
        "right": [slice(15, 21), slice(15, 21)],
    }

    to_rtn = {}

    # Plot confusion matrix for each scenario
    for scenario, indices in scenarios.items():

        g_true = convert_to_categories(c_true[:, indices[0]].astype(int))
        c_pred_scenario = convert_to_categories(c_pred[:, indices[1]].astype(int))

        # Compute confusion matrix
        cm = confusion_matrix(g_true, c_pred_scenario)

        # Plot confusion matrix
        plt.figure()
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title(f"{title} - {scenario}")
        plt.colorbar()

        n_classes = c_true[:, indices[0]].shape[1]

        tick_marks = np.arange(2**n_classes)
        plt.xticks(tick_marks, ["" for _ in range(len(tick_marks))])
        plt.yticks(tick_marks, ["" for _ in range(len(tick_marks))])

        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.tight_layout()

        # Save or show plot
        if save_path:
            plt.savefig(f"{save_path}_{scenario}.png")
        else:
            plt.show()

        to_rtn.update({scenario: cm})

        plt.close()

    return to_rtn


def save_embeddings(dataset: BaseDataset, device, name):
    dataset.return_embeddings = True
    dataset.args.batch_size = 1  # 1 as batch size
    train_loader, val_loader, test_loader = dataset.get_data_loaders()
    ood_loader = dataset.ood_loader
    dataset.print_stats()
    encoder, _ = dataset.get_backbone()
    encoder.to(device)

    if not os.path.exists(f"embeddings_{name}"):
        os.makedirs(f"embeddings_{name}")

    for loader, subfolder_name in zip(
        [train_loader, val_loader, test_loader, ood_loader],
        ["train", "val", "test", "ood"],
    ):
        encoder.eval()

        # create the folder
        if not os.path.exists(f"embeddings_{name}/{subfolder_name}"):
            os.makedirs(f"embeddings_{name}/{subfolder_name}")

        for _, data in enumerate(loader):
            images, labels, concepts, names = data
            images, labels, concepts, names = (
                images.to(device),
                labels.to(device),
                concepts.to(device),
                names,
            )

            embeddings = encoder(images)
            embeddings = embeddings.squeeze(dim=0)

            # Save embeddings
            file_name = names[0]  # Remove extension
            save_path = os.path.join(
                f"embeddings_{name}/{subfolder_name}", f"{file_name}.pt"
            )
            torch.save(embeddings, save_path)


def save_predictions_to_csv(model, test_set, csv_name, dataset):
    model.eval()

    ys, y_true, cs, cs_true = None, None, None, None

    for data in tqdm(test_set, desc="Saving predictions to CSV..."):
        images, labels, concepts = data
        images, labels, concepts = (
            images.to(model.device),
            labels.to(model.device),
            concepts.to(model.device),
        )

        out_dict = model(images)
        out_dict.update({"LABELS": labels, "CONCEPTS": concepts})

        if ys is None:
            ys = out_dict["YS"].cpu()
            y_true = out_dict["LABELS"].cpu()
            cs = out_dict["pCS"].cpu()
            cs_true = out_dict["CONCEPTS"].cpu()
        else:
            ys = torch.concatenate((ys, out_dict["YS"].cpu()), dim=0)
            y_true = torch.concatenate((y_true, out_dict["LABELS"].cpu()), dim=0)
            cs = torch.concatenate((cs, out_dict["pCS"].cpu()), dim=0)
            cs_true = torch.concatenate((cs_true, out_dict["CONCEPTS"].cpu()), dim=0)

    if dataset.endswith("mnist"):
        y_true = y_true.unsqueeze(1)
        cs = cs.reshape(cs.shape[0], cs.shape[1] * cs.shape[2])
    elif "kand" in dataset:
        cs = cs.reshape(cs.shape[0], cs.shape[1] * cs.shape[2])
        cs_true = cs_true.reshape(cs_true.shape[0], cs_true.shape[1] * cs_true.shape[2])

    concatenated_tensor = (
        torch.concatenate((ys, y_true, cs, cs_true), dim=1).cpu().detach().numpy()
    )

    # Save predictions to CSV file
    csv_path = os.path.join(csv_name)

    with open(csv_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(concatenated_tensor)


def train(model: MnistDPL, dataset: BaseDataset, _loss: ADDMNIST_DPL, args):
 import torch
from tqdm import tqdm
from utils.status import progress_bar
from utils.metrics import evaluate_metrics, evaluate_mix, mean_entropy
from utils import fprint

def train(model: MnistDPL, dataset: BaseDataset, _loss: ADDMNIST_DPL, args):

    """COMMENTI FILIPPO NARDI :
    
    MODIFICATO da GPT 01 PER ESSERE IL MINIMO INDESPENSABILE PER ESEGUIRE MNISTDPL CON 
    
    python main.py --dataset shortmnist --model mnistdpl --n_epochs 2 --lr 0.001 --seed 0 \
    --batch_size 64 --exp_decay 0.9 --c_sup 0 --task addition --backbone conceptizer

    LA DEF FUNZIONA E NON DA ERRORI, è STATO TOLTO TUTTO QUELLO CHE FA FUNZIONARE ALTRI DATASET O ARG 
    MA A DIFFERENZA DEL MIO CI SONO ANCHE AGGIUNTE

    TRAINING

    Args:
        model (MnistDPL): The neural network model.
        dataset (BaseDataset): The dataset object.
        _loss (ADDMNIST_DPL): The loss function.
        args: Command-line arguments and configurations.
    """

    # Name for saving the model
    save_path = f"best_model_{args.dataset}_{args.model}_{args.seed}.pth"

    # Move model to device
    model.to(model.device)
    if args.dataset == "shortmnist":
        model = model.float()

    # Get data loaders
    train_loader, val_loader, test_loader = dataset.get_data_loaders()

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ExponentialLR(model.opt, args.exp_decay)

    # Start training
    for epoch in range(args.n_epochs):
        model.train()
        ys, y_true = None, None

        for i, data in enumerate(train_loader):
            images, labels, concepts = data
            images, labels, concepts = (
                images.to(model.device),
                labels.to(model.device),
                concepts.to(model.device),
            )

            # Forward pass
            out_dict = model(images)
            out_dict.update({"LABELS": labels, "CONCEPTS": concepts})

            # Compute loss
            model.opt.zero_grad()
            loss, _ = _loss(out_dict, args)
            loss.backward()
            model.opt.step()

            # Collect predictions for training accuracy
            if ys is None:
                ys = out_dict["YS"]
                y_true = out_dict["LABELS"]
            else:
                ys = torch.cat((ys, out_dict["YS"]), dim=0)
                y_true = torch.cat((y_true, out_dict["LABELS"]), dim=0)

            # Progress bar
            if i % 10 == 0:
                progress_bar(i, len(train_loader), epoch, loss.item())

        # Calculate training accuracy
        y_pred = torch.argmax(ys, dim=-1)
        acc = (
            (y_pred.detach().cpu() == y_true.detach().cpu()).sum().item()
            / len(y_true)
            * 100
        )
        print(f"\nEpoch {epoch+1}/{args.n_epochs} - Training Accuracy: {acc:.2f}%")

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss, val_cacc, val_yacc, val_f1 = evaluate_metrics(model, val_loader, args)
        print(f"Validation Accuracy: {val_yacc:.2f}%, F1 Score: {val_f1:.2f}")

        # Update scheduler
        scheduler.step()

    # Save the trained model
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    # Evaluate on the test set
    model.load_state_dict(torch.load(save_path))
    model.to(model.device)
    model.eval()
    with torch.no_grad():
        y_true, c_true, y_pred, c_pred, p_cs, p_ys, p_cs_all, p_ys_all = evaluate_metrics(
            model, test_loader, args, last=True
        )

    # Compute test metrics
    yac, yf1 = evaluate_mix(y_true, y_pred)
    cac, cf1 = evaluate_mix(c_true, c_pred)
    h_c = mean_entropy(p_cs_all, model.n_facts)

    print(f"Test Label Accuracy: {yac:.2f}%, F1 Score: {yf1:.2f}")
    print(f"Test Concept Accuracy: {cac:.2f}%, F1 Score: {cf1:.2f}")
    print(f"Concept Entropy: {h_c:.4f}")

    # Plot confusion matrices
    plot_confusion_matrix(
        y_true,
        y_pred,
        labels=dataset.get_labels(),
        title="Labels",
        save_path=f"labels_{args.dataset}_{args.model}_lr_{args.lr}.png",
    )

    cf = plot_confusion_matrix(
        c_true,
        c_pred,
        labels=dataset.get_concept_labels(),
        title="Concepts",
        save_path=f"concepts_{args.dataset}_{args.model}_lr_{args.lr}.png",
    )

    print("Concept collapse:", 1 - compute_coverage(cf))