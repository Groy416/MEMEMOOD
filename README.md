# Meme-Based Movie Recommendation System

A simple Python script that uses OpenAI's CLIP model to recommend movies based on meme images. It works entirely locally—no server, no API, just pure Python!

## What It Does

Upload a meme image and get movie recommendations based on visual and semantic similarity. The system uses CLIP embeddings to understand both the content of your meme and movie descriptions, then finds the best matches.

## Installation

1. **Clone or download this project**

2. **Install dependencies:**
\`\`\`bash
pip install -r requirements.txt
\`\`\`

This installs:
- `torch` - PyTorch deep learning framework
- `transformers` - Hugging Face transformers (includes CLIP)
- `Pillow` - Image processing
- `numpy` - Numerical computing

## Quick Start

### Step 1: Prepare your movies
Edit `movies.csv` with your movies. Required columns:
- `title` - Movie title
- `genres` - Genre(s)
- `description` - Movie description
- `tags` - Keywords/tags

Example:
\`\`\`csv
title,genres,description,tags
Interstellar,Sci-Fi Drama,A team of astronauts travel through a wormhole,space time travel
The Dark Knight,Action Crime,Batman faces the Joker,superhero villain
\`\`\`

### Step 2: Run the script
\`\`\`bash
python recommend_movies.py meme.jpg movies.csv
\`\`\`

Or if you want to use the default `movies.csv`:
\`\`\`bash
python recommend_movies.py meme.jpg
\`\`\`

### Step 3: Get recommendations
The script will output the top 5 movie recommendations with similarity scores:

\`\`\`
======================================================================
MOVIE RECOMMENDATIONS BASED ON YOUR MEME
======================================================================

#1 - Interstellar
   Genres: Sci-Fi Drama
   Similarity Score: 72.34%
   Description: A team of astronauts travel through a wormhole...
   Tags: space time travel dimensional

#2 - Inception
   Genres: Sci-Fi Thriller
   Similarity Score: 68.91%
   Description: A thief extracts secrets from dreams...
   Tags: dreams heist reality bending
\`\`\`

## How It Works

1. **CLIP Model**: Uses OpenAI's CLIP (Contrastive Language-Image Pre-training) to generate embeddings
2. **Image Processing**: Converts your meme into a 512-dimensional embedding
3. **Text Processing**: Combines movie titles, genres, descriptions, and tags into embeddings
4. **Similarity Search**: Computes cosine similarity between meme and all movies
5. **Ranking**: Returns top-5 movies sorted by similarity score

## File Structure

\`\`\`
.
├── recommend_movies.py      # Main script
├── requirements.txt         # Python dependencies
├── movies.csv              # Your movie database
└── README.md               # This file
\`\`\`

## Usage Examples

### Example 1: Sci-fi meme
\`\`\`bash
python recommend_movies.py spacememe.jpg
\`\`\`

### Example 2: Action-packed meme
\`\`\`bash
python recommend_movies.py actionmeme.jpg
\`\`\`

### Example 3: Custom movie database
\`\`\`bash
python recommend_movies.py meme.jpg my_movies.csv
\`\`\`

## System Requirements

- Python 3.8+
- 8GB+ RAM recommended (4GB minimum)
- GPU optional (CPU works too, just slower)
- Internet connection for first run (downloads CLIP model)

## First Run

On the first run, the script will download the CLIP model (~350MB). This is a one-time download and will be cached locally.

\`\`\`
[*] Loading CLIP model (this may take a moment on first run)...
[*] Using device: cpu
[*] Model loaded successfully!
\`\`\`

## Performance

| Operation | Time |
|-----------|------|
| Load CLIP model | ~5-10 seconds (first run only) |
| Generate movie embeddings | ~1-2 seconds for 20 movies |
| Process meme image | ~1-2 seconds |
| Get recommendations | Instant |

## Tips & Tricks

1. **Better Descriptions**: Detailed movie descriptions lead to better recommendations
2. **Rich Tags**: Add relevant keywords/tags to help the model understand context
3. **Image Quality**: Clearer, more representative meme images work better
4. **Batch Processing**: Edit the script to loop through multiple memes in a folder

## Customization

### Change top-k results
Modify the script call:
\`\`\`python
recommendations = recommender.recommend_from_meme(meme_path, top_k=10)  # Get top 10 instead of 5
\`\`\`

### Use different CLIP model
Edit the model name in `MemeMovieRecommender()`:
\`\`\`python
# Options: openai/clip-vit-base-patch32 (default), openai/clip-vit-large-patch14, etc.
recommender = MemeMovieRecommender(model_name="openai/clip-vit-large-patch14")
\`\`\`

## Troubleshooting

**Q: "CUDA out of memory" error**
- A: Run on CPU by modifying the script (device will auto-detect, but you can force CPU)

**Q: Model takes too long to download**
- A: Normal on first run. Subsequent runs use cached model.

**Q: Poor recommendations**
- A: Try adding more detailed descriptions and tags to your movies.csv

## Advantages of This Approach

- No server setup required
- Completely offline after first run
- No API costs
- Privacy-friendly (everything stays on your machine)
- Simple, easy to understand code
- No complex deployment

## Future Enhancements

1. Add batch processing for multiple memes
2. Cache movie embeddings between runs
3. Interactive CLI menu
4. Export recommendations to JSON
5. Web UI with Flask/Streamlit
6. Database integration for larger movie catalogs

## License

Free to use and modify!



