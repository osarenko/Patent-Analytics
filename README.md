# Patent Analytics Tool

A Python-based patent analytics tool that retrieves patent data from Google Patents, processes patent claims using natural language processing, and performs clustering analysis to identify patterns and outliers in patent landscapes.

## Overview

This tool performs the following key functions:
1. **Patent Data Retrieval**: Scrapes patent information from Google Patents
2. **Text Processing**: Uses DistilBERT to generate embeddings for patent claims
3. **Dimensionality Reduction**: Applies UMAP for 2D visualization
4. **Outlier Detection**: Identifies unusual patents based on claim similarity
5. **Visualization**: Creates scatter plots showing patent claim landscapes

## Features

- 🔍 **Web Scraping**: Automated retrieval of patent claims and abstracts
- 🤖 **NLP Processing**: DistilBERT-based semantic embeddings
- 📊 **Data Visualization**: Interactive scatter plots with outlier highlighting
- 🧹 **Data Cleaning**: Robust preprocessing and outlier filtering
- 💾 **Data Persistence**: Saves processed data for future analysis

## Requirements

### Dependencies

```bash
pip install -r requirements.txt
```

**Required packages:**
- `beautifulsoup4` - Web scraping
- `certifi` - SSL certificate handling
- `transformers` - DistilBERT model
- `torch` - PyTorch backend
- `numpy` - Numerical computations
- `matplotlib` - Plotting
- `scikit-learn` - Machine learning utilities
- `umap-learn` - Dimensionality reduction
- `tqdm` - Progress bars
- `lxml` - XML/HTML parsing

### Hardware Requirements

- **CPU**: Multi-core recommended for faster processing
- **RAM**: Minimum 8GB (16GB+ recommended for larger datasets)
- **GPU**: Optional but recommended for faster embeddings generation

## Usage

### Basic Usage

1. **Configure Patent Numbers**: Edit the `patent_numbers` list in the script:
```python
patent_numbers = ['US2020057781A1', 'US10664487B2', 'US10474562B2', 'US9910911B2']
```

2. **Run the Analysis**:
```bash
python patent_analytics.py
```

### What the Script Does

#### 1. Patent Data Retrieval
- Fetches patent data from Google Patents
- Extracts claims and abstracts
- Implements rate limiting to avoid overwhelming the server
- Handles SSL certificates properly

#### 2. Text Processing Pipeline
- Tokenizes patent claims using DistilBERT tokenizer
- Processes text in batches for memory efficiency
- Generates semantic embeddings for each claim
- Implements proper padding and truncation

#### 3. Data Cleaning
- Removes invalid embeddings (NaN, infinity values)
- Filters out low-norm vectors
- Reports cleaning statistics

#### 4. Dimensionality Reduction
- Uses UMAP to reduce embeddings to 2D
- Preserves local and global structure
- Enables visualization of patent landscapes

#### 5. Outlier Detection
- Identifies patents with unusual claim patterns
- Uses k-nearest neighbors approach
- Applies percentile-based thresholding

## Output Files

The script generates several output files:

| File | Description |
|------|-------------|
| `patents.pkl` | Raw patent data (claims, abstracts, numbers) |
| `embeddings_2d.pkl` | 2D UMAP embeddings |
| `claim_patent_mapping_clean.pkl` | Cleaned mapping of claims to patents |

## Configuration Options

### Batch Sizes
```python
# Tokenization batch size (adjust based on memory)
batch_size = 64

# Inference batch size (adjust based on GPU memory)
inference_batch_size = 32
```

### Model Selection
```python
# Change model for different embeddings
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')
```

### UMAP Parameters
```python
# Adjust for different clustering behavior
reducer = umap.UMAP(n_components=2, random_state=42)
```

## Visualization

The script generates a scatter plot showing:
- **Blue dots**: Normal patent claims
- **Red dots**: Outlier claims
- **Clusters**: Groups of similar patents

## Error Handling

The script includes robust error handling for:
- Network timeouts during web scraping
- Invalid patent numbers
- Missing patent sections
- SSL certificate issues
- Memory constraints

## Performance Optimization

### For Large Datasets
1. **Reduce batch sizes** if running out of memory
2. **Use GPU** if available for faster processing
3. **Implement checkpointing** for very large patent sets
4. **Consider distributed processing** for production use

### Memory Management
- The script uses batched processing to handle memory constraints
- Embeddings are generated in chunks to avoid OOM errors
- Intermediate results are saved to disk

## Troubleshooting

### Common Issues

**SSL Certificate Errors**:
```python
# The script uses certifi for proper SSL handling
ssl_context = ssl.create_default_context(cafile=certifi.where())
```

**Memory Issues**:
- Reduce batch sizes
- Use CPU instead of GPU if memory is limited
- Process fewer patents at once

**Network Issues**:
- Increase timeout values
- Add retry mechanisms
- Check internet connectivity

### Debugging

Enable verbose output by adding:
```python
import logging
logging.basicConfig(level=logging.INFO)
```

## Example Results

The tool can identify:
- **Patent clusters**: Groups of similar technologies
- **Outlier patents**: Unique or innovative claims
- **Technology trends**: Evolution of patent landscapes
- **Prior art relationships**: Similar existing patents

## Contributing

To extend the functionality:
1. Add new patent databases
2. Implement different embedding models
3. Add more sophisticated clustering algorithms
4. Create interactive visualizations


## Acknowledgments

- Google Patents for patent data
- Hugging Face for transformer models
- UMAP developers for dimensionality reduction 
