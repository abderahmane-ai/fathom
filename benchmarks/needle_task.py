import torch
from torch.utils.data import Dataset, DataLoader

class NeedleDataset(Dataset):
    """
    Synthetic 'Depth-Needle' task to measure information preservation across layers.
    
    Format: [NEEDLE, PAYLOAD, RND, RND, ..., PAYLOAD_TARGET]
    Sequence length: 256
    Vocab size: 1024
    Needle position: 0
    Payload position: 1
    Target: The payload token, to be predicted at the final position.
    """
    def __init__(self, size: int = 10000, seq_len: int = 256, vocab_size: int = 1024):
        self.size = size
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.needle_token = 1023  # Last token in vocab
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx: int):
        # Generate random tokens (excluding needle)
        tokens = torch.randint(0, self.vocab_size - 1, (self.seq_len,))
        
        # Set needle and payload
        tokens[0] = self.needle_token
        payload = torch.randint(0, self.vocab_size - 1, (1,)).item()
        tokens[1] = payload
        
        # In causal LM, the target at index i is the token at index i+1.
        # We want to predict 'payload' at the final position.
        # So the input sequence is tokens[0:255] and the last target is tokens[255].
        # But for simplicity in this diagnostic, we can just return the full sequence
        # and ensure the last token is the payload.
        tokens[-1] = payload
        
        return tokens

def evaluate_needle(model, device, size: int = 1000, seq_len: int = 256) -> float:
    """Evaluate needle accuracy and restore train/eval mode afterward."""
    was_training = model.training
    model.eval()
    dataset = NeedleDataset(size=size, seq_len=seq_len)
    loader = DataLoader(dataset, batch_size=32)

    correct = 0
    total = 0

    try:
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                inputs = batch[:, :-1]
                targets = batch[:, -1]

                logits = model(inputs)
                last_logits = logits[:, -1, :]
                preds = last_logits.argmax(dim=-1)

                correct += (preds == targets).sum().item()
                total += targets.size(0)
    finally:
        model.train(was_training)

    if total == 0:
        raise RuntimeError("Needle evaluation produced zero samples.")

    return correct / total
