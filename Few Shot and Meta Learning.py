import os
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import timm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# --- Dataset Loading and Episodic Sampling ---

def load_nwpu_resisc45(dataset_path, transform=None, max_images=None):
    class_names = sorted(os.listdir(dataset_path))
    images = []
    labels = []

    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_dir):
            continue

        img_files = os.listdir(class_dir)
        for img_file in tqdm(img_files, desc=f"Loading {class_name}"):
            img_path = os.path.join(class_dir, img_file)
            try:
                img = Image.open(img_path).convert('RGB')
                if transform:
                    img = transform(img)
                images.append(img)
                labels.append(class_idx)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")

            if max_images and len(images) >= max_images:
                break
        if max_images and len(images) >= max_images:
            break

    return images, labels, class_names

class FewShotDataset:
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        self.classes = list(set(labels))
        self.class_to_indices = {c: np.where(np.array(labels)==c)[0] for c in self.classes}

    def sample_episode(self, n_way=5, n_shot=5, n_query=15):
        chosen_classes = np.random.choice(self.classes, n_way, replace=False)
        support_images, support_labels = [], []
        query_images, query_labels = [], []
        for i, c in enumerate(chosen_classes):
            indices = np.random.choice(self.class_to_indices[c], n_shot + n_query, replace=False)
            support_idx = indices[:n_shot]
            query_idx = indices[n_shot:]
            for idx in support_idx:
                support_images.append(self.images[idx])
                support_labels.append(i)
            for idx in query_idx:
                query_images.append(self.images[idx])
                query_labels.append(i)
        return support_images, support_labels, query_images, query_labels

# --- Visual Prompt Tuning Module ---

class PromptTuning(nn.Module):
    def __init__(self, prompt_len, embed_dim):
        super().__init__()
        self.prompt_tokens = nn.Parameter(torch.randn(prompt_len, embed_dim))

    def forward(self, x):
        batch_size = x.shape[0]
        prompt_expanded = self.prompt_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        return torch.cat([prompt_expanded, x], dim=1)

# --- Few-Shot Model with ViT Backbone + Prompt Tuning ---

class FewShotModel(nn.Module):
    def __init__(self, vit_model_name='vit_base_patch32_224', prompt_len=10, n_classes=5):
        super().__init__()
        self.n_classes = n_classes
        self.backbone = timm.create_model(vit_model_name, pretrained=True, num_classes=0)
        embed_dim = self.backbone.embed_dim
        self.prompt_tuning = PromptTuning(prompt_len=prompt_len, embed_dim=embed_dim)
        self.classifier = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        x = self.backbone.patch_embed(x)  # [B, N, E]
        cls_token = self.backbone.cls_token.expand(x.shape[0], -1, -1)  # [B,1,E]
        x = torch.cat((cls_token, x), dim=1)  # prepend cls token
        x = x + self.backbone.pos_embed[:, :(x.size(1)), :]
        x = self.backbone.pos_drop(x)

        # Insert prompt tokens after cls token
        cls_token_expanded = x[:, :1, :]
        patch_tokens = x[:, 1:, :]
        prompt_tokens = self.prompt_tuning.prompt_tokens.unsqueeze(0).expand(x.size(0), -1, -1)
        x = torch.cat((cls_token_expanded, prompt_tokens, patch_tokens), dim=1)

        for blk in self.backbone.blocks:
            x = blk(x)

        x = self.backbone.norm(x)
        cls_out = x[:, 0]
        logits = self.classifier(cls_out)
        return logits

# --- MAML Inner Loop ---

def maml_inner_loop(model, support_images, support_labels, loss_fn, inner_steps=5, inner_lr=0.01):
    prompt_params = [p for p in model.prompt_tuning.parameters()]
    fast_params = [p.clone().detach().requires_grad_(True) for p in prompt_params]

    optimizer = torch.optim.SGD(fast_params, lr=inner_lr)

    for _ in range(inner_steps):
        optimizer.zero_grad()
        with torch.no_grad():
            for p_old, p_new in zip(prompt_params, fast_params):
                p_old.copy_(p_new)

        outputs = model(support_images)
        loss = loss_fn(outputs, support_labels)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        for p_old, p_new in zip(prompt_params, fast_params):
            p_old.copy_(p_new)

    return model

# --- Training and Evaluation Loops ---

def train_epoch(model, dataset, optimizer, loss_fn, device, n_way=5, n_shot=5, n_query=15, inner_steps=5, inner_lr=0.01):
    model.train()
    episode_acc = []
    for episode in range(20):
        support_imgs, support_labels, query_imgs, query_labels = dataset.sample_episode(n_way, n_shot, n_query)

        support_imgs = torch.stack([img.to(device) for img in support_imgs])
        query_imgs = torch.stack([img.to(device) for img in query_imgs])
        support_labels_tensor = torch.tensor(support_labels, dtype=torch.long, device=device)
        query_labels_tensor = torch.tensor(query_labels, dtype=torch.long, device=device)

        model = maml_inner_loop(model, support_imgs, support_labels_tensor, loss_fn, inner_steps, inner_lr)

        optimizer.zero_grad()
        outputs = model(query_imgs)
        loss = loss_fn(outputs, query_labels_tensor)
        loss.backward()
        optimizer.step()

        preds = torch.argmax(outputs, dim=1)
        acc = (preds == query_labels_tensor).float().mean().item()
        episode_acc.append(acc)

    return np.mean(episode_acc)

def eval_model(model, dataset, device, n_way=5, n_shot=5, n_query=15):
    model.eval()
    episode_acc = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for episode in range(10):
            support_imgs, support_labels, query_imgs, query_labels = dataset.sample_episode(n_way, n_shot, n_query)

            support_imgs = torch.stack([img.to(device) for img in support_imgs])
            query_imgs = torch.stack([img.to(device) for img in query_imgs])
            support_labels_tensor = torch.tensor(support_labels, dtype=torch.long, device=device)
            query_labels_tensor = torch.tensor(query_labels, dtype=torch.long, device=device)

            model = maml_inner_loop(model, support_imgs, support_labels_tensor, nn.CrossEntropyLoss(), inner_steps=5, inner_lr=0.005)

            outputs = model(query_imgs)
            preds = torch.argmax(outputs, dim=1)

            acc = (preds == query_labels_tensor).float().mean().item()
            episode_acc.append(acc)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(query_labels)

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')

    return np.mean(episode_acc), accuracy, precision, recall, f1

# --- Main ---

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset_dir = "/path_to/NWPU-RESISC45"  # <-- Set your dataset path here

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # Normalize if needed, e.g.:
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225])
    ])

    print("Loading dataset...")
    images, labels, class_names = load_nwpu_resisc45(dataset_dir, transform=transform)
    print(f"Loaded {len(images)} images from {len(class_names)} classes")

    dataset = FewShotDataset(images, labels)

    model = FewShotModel(vit_model_name='vit_base_patch32_224', prompt_len=10, n_classes=5).to(device)
    optimizer = torch.optim.Adam(model.prompt_tuning.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    epochs = 10
    train_accs, val_accs, precisions, recalls, f1s = [], [], [], [], []

    for epoch in range(epochs):
        train_acc = train_epoch(model, dataset, optimizer, loss_fn, device)
        val_acc, acc, prec, rec, f1 = eval_model(model, dataset, device)

        print(f"Epoch {epoch+1} | Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f} | Precision: {prec:.3f} | Recall: {rec:.3f} | F1: {f1:.3f}")
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

    # Plot Accuracy
    plt.figure(figsize=(10,6))
    plt.plot(range(1, epochs+1), train_accs, label='Train Accuracy')
    plt.plot(range(1, epochs+1), val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.show()

    # Plot Precision, Recall, F1
    plt.figure(figsize=(10,6))
    plt.plot(range(1, epochs+1), precisions, label='Precision')
    plt.plot(range(1, epochs+1), recalls, label='Recall')
    plt.plot(range(1, epochs+1), f1s, label='F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Precision, Recall, and F1 Score')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
