from sentence_transformers import SentenceTransformer, models, losses
from sentence_transformers.datasets import DenoisingAutoEncoderDataset
from torch.utils.data import DataLoader
import random

# Load queries
with open("queries.txt") as f:
    queries = [line.strip() for line in f if line.strip()]

# shuffle
random.shuffle(queries)

train_queries = queries[:27000]
val_queries = queries[27000:]

# base encoder
model_name = "all-MiniLM-L6-v2"
model = SentenceTransformer(model_name)

# TSDAE dataset
train_dataset = DenoisingAutoEncoderDataset(train_queries)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

# loss
train_loss = losses.DenoisingAutoEncoderLoss(
    model,
    decoder_name_or_path=model_name
)

# train
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=8,
    weight_decay=0,
    scheduler="constantlr",
    optimizer_params={"lr": 3e-5},
    show_progress_bar=True,
)

# save model
model.save("minilm-product-domain")
