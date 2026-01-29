import torch
from torch.utils.data import Dataset



class SentenceDataset(Dataset):
    def __init__(self, sentences, vocab_mapping, cat2idx, max_len=None):
        self.sentences = []
        self.pos = []

        # Define padding info
        self.pad_token = '<PAD>'
        self.pad_idx_vocab = vocab_mapping[self.pad_token]
        self.pad_idx_pos = -100  # Standard for CrossEntropyLoss to ignore padding

        # Use provided max_len or calculate from data
        self.max_len = max_len if max_len else max(len(s) for s in sentences)

        for sentence in sentences:
            # 1. Map words to indices, then pad
            word_indices = [vocab_mapping.get(
                v[0], vocab_mapping.get('<UNK>', 0)) for v in sentence]
            # 2. Map tags to indices, then pad
            tag_indices = [cat2idx[v[1]] for v in sentence]

            num_padding = self.max_len - len(word_indices)

            # Apply Padding
            word_indices.extend([self.pad_idx_vocab] * num_padding)
            tag_indices.extend([self.pad_idx_pos] * num_padding)

            self.sentences.append(word_indices)
            self.pos.append(tag_indices)

    def __getitem__(self, idx):
        # Convert lists to tensors here
        return (
            torch.LongTensor(self.sentences[idx]),
            torch.LongTensor(self.pos[idx])
        )

    def __len__(self):
        return len(self.sentences)
