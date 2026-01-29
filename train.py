from torch.optim import AdamW
from tqdm import tqdm
from dataset import SentenceDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import POSTransformer
from embeddings import AlBI, RoPE, LearnedEmbeddings
from utils import create_data_set
import argparse

EPOCHS = 20

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--e', '--embedding')

    args = parser.parse_args()

    

    print(F'Embedding Type: {args.e}')


    dataset, vocab_mapping, cat2idx = create_data_set(
        'en-ud-v2/en-ud-tag.v2.train.txt')
    val_dataset, _, _ = create_data_set(
        'en-ud-v2/en-ud-tag.v2.test.txt', vocab_mapping=vocab_mapping, cat2idx=cat2idx)
    val_dataloader = DataLoader(val_dataset, batch_size=512)
    
    train_dataloader = DataLoader(
        batch_size=256,
        dataset=dataset,
        shuffle=True
    )
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    d_hidden = 128
    length = dataset.max_len
    print(f'Seq Length: {length}')

    device = torch.device("mps")
    embedding_type = RoPE(d_hidden, length)
    if args.e == 'albi':
        embedding_type = AlBI(
            num_heads=4,
            seq_len=length,
            device=device
        )
    elif args.e == 'rope':
        embedding_type = RoPE(d_hidden, length)
    elif args.e == 'learned':
        embedding_type = LearnedEmbeddings(
            length, d_hidden, device)

    model = POSTransformer(num_layers=4,
                        num_heads=4,
                        ffn_ratio=4.0,
                        vocab_size=len(vocab_mapping),
                        num_pos=len(cat2idx),
                        positional_layer=embedding_type,
                        d_hidden=128).to(device=device)

    optimizer = AdamW(model.parameters(), lr=0.0005)
    val_dataset, _, _ = create_data_set(
        'en-ud-v2/en-ud-tag.v2.test.txt', vocab_mapping=vocab_mapping, cat2idx=cat2idx)
    val_dataloader = DataLoader(val_dataset, batch_size=512)
    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        train_pbar = tqdm(train_dataloader,
                        desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")

        for i, (x, y) in enumerate(train_pbar):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            predicted = model(x)
            loss = criterion(predicted.transpose(1, 2), y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if i % 20 == 0:
                train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_train_loss = train_loss / len(train_dataloader)

        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for x, y in tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
                x, y = x.to(device), y.to(device)

                outputs = model(x)
                v_loss = criterion(outputs.transpose(1, 2), y)
                val_loss += v_loss.item()

                # Accuracy
                preds = torch.argmax(outputs, dim=-1)
                mask = (y != -100)
                val_correct += ((preds == y) & mask).sum().item()
                val_total += mask.sum().item()

        avg_val_loss = val_loss / len(val_dataloader)
        avg_val_acc = val_correct / val_total if val_total > 0 else 0

        # Print summary for the epoch
        print(f"\nSummary Epoch {epoch+1}:")
        print(
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {avg_val_acc:.4f}")

        # Checkpoint: Save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pt')
            print(" New best model saved!")
        print("-" * 30)
