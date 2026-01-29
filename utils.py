from dataset import SentenceDataset
def read_file(curr_file):
    sentences = []
    current_sentence = []

    with open(curr_file, 'r', encoding='utf-8') as file:
        for line in file:
            clean_line = line.strip()

            if not clean_line:
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
            else:
                # words are tab seperated
                parts = clean_line.split('\t')
                # grab word
                if len(parts) >= 2:
                    current_sentence.append((parts[0], parts[1]))

    if current_sentence:
        sentences.append(current_sentence)
    return sentences


def create_vocab(sentences):
    vocab = set(['<UNK>', '<PAD>'])
    pos = set()

    for sentence in sentences:
        words = [v[0] for v in sentence]
        poss = [v[1] for v in sentence]
        vocab.update(words)
        pos.update(poss)

    cat2idx = {cat: idx for idx, cat in enumerate(pos)}
    vocab_mapping = {cat: idx for idx, cat in enumerate(vocab)}

    return vocab, pos, cat2idx, vocab_mapping


def create_data_set(file_name, vocab_mapping=None, cat2idx=None):
    sentences = read_file(file_name)
    vocab, pos, _cat2idx, _vocab_mapping = create_vocab(sentences)
    if vocab_mapping is None:
        vocab_mapping = _vocab_mapping
    if cat2idx is None:
        cat2idx = _cat2idx
    dataset = SentenceDataset(
        sentences, vocab_mapping=vocab_mapping, cat2idx=cat2idx)

    return dataset, vocab_mapping, cat2idx
