import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from torchkit.backbone.model_irse import IR_50
from torchkit.data.dataset import TF_SyntheticDataset


def main():
    # ----------- Config -----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = "../dataset/Syn_10k/"
    batch_size = 16
    checkpoint_path = "ckpt/Backbone_Epoch_40_checkpoint.pth"

    # ----------- Transform -----------
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # ----------- Dataset and Dataloader -----------
    classes = ["all"]
    ds = TF_SyntheticDataset(data_root, classes, transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # ----------- Model -----------
    model = IR_50(input_size=[112, 112])
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt)
    model = model.to(device)
    model.eval()

    # ----------- Create Class Centers -----------
    class_to_embeddings = {}
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Building class centers"):
            images = images.to(device)
            labels = labels.to(device)

            embeddings = model(images)

            for emb, label in zip(embeddings, labels):
                label = label.item()
                if label not in class_to_embeddings:
                    class_to_embeddings[label] = []
                class_to_embeddings[label].append(emb.cpu())

    # Compute class centers
    class_centers = {}
    for label, embs in class_to_embeddings.items():
        embs = torch.stack(embs, dim=0)
        center = embs.mean(dim=0)
        class_centers[label] = center

    # ----------- Evaluate using nearest center -----------
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)

            embeddings = model(images)

            for emb, label in zip(embeddings, labels):
                distances = {k: torch.norm(emb.cpu() - v) for k, v in class_centers.items()}
                pred = min(distances, key=distances.get)

                if pred == label.item():
                    correct += 1
                total += 1

    acc = 100.0 * correct / total
    print(f"[RESULT] Accuracy: {acc:.2f}%")


if __name__ == "__main__":
    main()
