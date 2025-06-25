import urllib.request
import certifi
import ssl
from bs4 import BeautifulSoup
import time
from transformers import DistilBertTokenizer, DistilBertModel
import numpy as np
import torch
from tqdm import tqdm
import umap
import pickle
import matplotlib.pyplot as plt

# ----------------------------
#Patent Data Retrieval
# ----------------------------
patent_numbers = ['US2020057781A1',	'US10664487B2',	'US10474562B2',	'US9910911B2']
 # Add more patent numbers here as needed
claim_patent_mapping = []
ssl_context = ssl.create_default_context(cafile=certifi.where())

patents = []
claim_texts = []

for idx, pn in enumerate(patent_numbers):
    try:
        url = f"https://patents.google.com/patent/{pn}"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        webpage = urllib.request.urlopen(req, context=ssl_context, timeout=10).read()
        soup = BeautifulSoup(webpage, 'lxml')

        # Extract claims
        claims_section = soup.find('section', {'itemprop': 'claims'})
        if not claims_section:
            print(f"Skipping {pn}: No claims section")
            continue

        claims = [c.get_text(' ', strip=True) for c in claims_section.find_all('div', class_='claim-text')]
        if not claims:
            print(f"Skipping {pn}: Empty claims")
            continue

        claim_texts.extend(claims)
        claim_patent_mapping.extend([pn] * len(claims))
        patents.append({
            'number': pn,
            'claims': claims,
            'abstract': soup.find('meta', attrs={'name':'DC.description'})['content']
                        if soup.find('meta', attrs={'name':'DC.description'}) else None
        })

        # Progress tracking
        if (idx + 1) % 5 == 0 or (idx + 1) == len(patent_numbers):
            print(f"Processed {idx+1}/{len(patent_numbers)} patents")

        # Rate limiting
        time.sleep(1)

    except Exception as e:
        print(f"Error on {pn}: {str(e)}")
        continue

# Save patent numbers, claims, and abstract for backup

claim_texts = [str(t) for t in claim_texts]  # Ensure string format

assert len(claim_texts) > 0, "No claims loaded from file"
print(f"Loaded {len(claim_texts)} patent claims")

with open('patents.pkl', 'wb') as f:
    pickle.dump(patents, f)

# ----------------------------
# 2. Batched Processing Pipeline (FIXED)
# ----------------------------
def batch_tokenize(texts, tokenizer, batch_size=64):
    #Tokenize text in batches with consistent padding
    if not texts:
        raise ValueError("Empty input list - check your claims.pkl file")

    # Calculate global max length
    max_length = 0
    for txt in tqdm(texts, desc="Calculating max length"):
        encoded = tokenizer.encode(txt, truncation=True, add_special_tokens=True)
        max_length = max(max_length, len(encoded))
    max_length = min(max_length, 512)

    all_inputs = {'input_ids': [], 'attention_mask': []}

    for i in tqdm(range(0, len(texts), batch_size),
                 desc="Tokenizing",
                 unit="batch"):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(
            batch,
            return_tensors='pt',
            padding='max_length',  # Crucial fix
            truncation=True,
            max_length=max_length,
            add_special_tokens=True
        )
        all_inputs['input_ids'].append(inputs['input_ids'])
        all_inputs['attention_mask'].append(inputs['attention_mask'])

    return {
        'input_ids': torch.cat(all_inputs['input_ids'], dim=0),
        'attention_mask': torch.cat(all_inputs['attention_mask'], dim=0)
    }

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(device)
print(f"\nModel loaded on {device}...")

# Get properly padded inputs
inputs = batch_tokenize(claim_texts, tokenizer)


def batched_inference(model, inputs, batch_size=32):
    """Run model inference with proper batching"""
    embeddings = []
    total_batches = len(inputs['input_ids']) // batch_size + 1

    for i in tqdm(range(0, len(inputs['input_ids']), batch_size),
                 desc="Generating Embeddings",
                 total=total_batches,
                 unit="batch"):
        batch = {
            'input_ids': inputs['input_ids'][i:i+batch_size].to(device),
            'attention_mask': inputs['attention_mask'][i:i+batch_size].to(device)
        }

        with torch.no_grad():
            outputs = model(**batch)
            embeddings.append(outputs.last_hidden_state.mean(dim=1).cpu().numpy())

    return np.concatenate(embeddings, axis=0)

# Generate embeddings
embeddings = batched_inference(model, inputs)

# ----------------------------
#Data Cleansing
# ----------------------------
valid_mask = ~np.isnan(embeddings).any(axis=1) & ~np.isinf(embeddings).any(axis=1)
claim_patent_mapping_clean = [claim_patent_mapping[i] for i in np.where(valid_mask)[0]]
clean_embeddings = embeddings[valid_mask]

norms = np.linalg.norm(clean_embeddings, axis=1)
median_norm = np.median(norms)
clean_embeddings = clean_embeddings[norms > median_norm * 0.01]

print(f"\nCleaning report:")
print(f"Original: {embeddings.shape[0]} vectors")
print(f"Final: {clean_embeddings.shape[0]} vectors")
print(f"Removed: {embeddings.shape[0] - clean_embeddings.shape[0]} vectors")
reducer = umap.UMAP(n_components=2, random_state=42)
embeddings_2d = reducer.fit_transform(clean_embeddings)

with open('embeddings_2d.pkl', 'wb') as f:
    pickle.dump(embeddings_2d, f)

with open('claim_patent_mapping_clean.pkl' 'wb') as f:
    pickle.dump(claim_patent_mapping_clean, f)

# Find points far from cluster centers
from sklearn.neighbors import NearestNeighbors

nbrs = NearestNeighbors(n_neighbors=50).fit(embeddings_2d)
distances, indices = nbrs.kneighbors(embeddings_2d)
mean_distances = distances.mean(axis=1)

# Use percentile threshold instead of fixed contamination
threshold = np.percentile(mean_distances, 99)
outlier_indices = np.where(mean_distances > threshold)[0]
outlier_patents = list(set([claim_patent_mapping_clean[i] for i in outlier_indices]))
print(f"Outlier patents: {outlier_patents}")

# Modify alpha values for patent outlier detection
plt.figure(figsize=(12,8))
plt.scatter(embeddings_2d[:,0], embeddings_2d[:,1], s=2, alpha=0.5, label='Normal')
plt.scatter(embeddings_2d[outlier_indices,0],
            embeddings_2d[outlier_indices,1],
            color='red', s=20, label='Outlier Claims')
plt.title("Patent Claim Landscape with Outliers Highlighted")
plt.legend()
plt.show()
