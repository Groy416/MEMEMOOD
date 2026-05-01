"""
Meme-based Movie Recommendation System
Uses CLIP embeddings to find movies similar to your meme images.
"""

import os
import sys
import csv
import numpy as np
from pathlib import Path
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


class MemeMovieRecommender:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        """Initialize CLIP model and processor."""
        print("[*] Loading CLIP model (this may take a moment on first run)...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[*] Using device: {self.device}")
        
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.movies = []
        self.movie_embeddings = None
        print("[✓] Model loaded successfully!")
    
    def load_movies_from_csv(self, csv_path):
        """Load movies from CSV file and generate embeddings."""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        print(f"\n[*] Loading movies from {csv_path}...")
        self.movies = []
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                self.movies.append(row)
        
        print(f"[✓] Loaded {len(self.movies)} movies")
        
        # Generate embeddings for all movies
        print("[*] Generating embeddings for all movies...")
        self._generate_movie_embeddings()
        print("[✓] Movie embeddings generated!")
    
    def _generate_movie_embeddings(self):
        """Generate embeddings for all movies."""
        movie_texts = []
        for movie in self.movies:
            # Combine all text fields for richer context
            text = f"{movie.get('title', '')} {movie.get('genres', '')} {movie.get('description', '')} {movie.get('tags', '')}"
            movie_texts.append(text)
        
        embeddings = []
        batch_size = 32
        
        for i in range(0, len(movie_texts), batch_size):
            batch = movie_texts[i:i + batch_size]
            inputs = self.processor(text=batch, return_tensors="pt", padding=True).to(self.device)
            
            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                embeddings.extend(text_features.cpu().numpy())
        
        self.movie_embeddings = np.array(embeddings)
    
    def recommend_from_meme(self, meme_path, top_k=5):
        """
        Recommend movies based on a meme image.
        
        Args:
            meme_path: Path to the meme image
            top_k: Number of recommendations to return
            
        Returns:
            List of recommendations with scores
        """
        if not self.movies:
            raise ValueError("No movies loaded. Call load_movies_from_csv() first.")
        
        if not os.path.exists(meme_path):
            raise FileNotFoundError(f"Meme image not found: {meme_path}")
        
        print(f"\n[*] Processing meme: {meme_path}")
        
        # Load and process image
        image = Image.open(meme_path)
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        # Generate embedding for meme
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        meme_embedding = image_features.cpu().numpy()[0]
        
        # Compute similarity scores
        similarities = np.dot(self.movie_embeddings, meme_embedding)
        
        # Get top-k recommendations
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        recommendations = []
        for rank, idx in enumerate(top_indices, 1):
            movie = self.movies[idx]
            score = float(similarities[idx])
            
            recommendations.append({
                'rank': rank,
                'title': movie.get('title', 'Unknown'),
                'genres': movie.get('genres', 'Unknown'),
                'description': movie.get('description', 'No description'),
                'tags': movie.get('tags', ''),
                'similarity_score': score
            })
        
        return recommendations
    
    def print_recommendations(self, recommendations):
        """Pretty print recommendations."""
        print("\n" + "="*70)
        
        print("🎬 MOVIE RECOMMENDATIONS BASED ON YOUR MEME")
        print("="*70)
        
        for rec in recommendations:
            print(f"\n#{rec['rank']} - {rec['title']}")
            print(f"   Genres: {rec['genres']}")
            print(f"   Similarity Score: {rec['similarity_score']:.2%}")
            print(f"   Description: {rec['description']}")
            if rec['tags']:
                print(f"   Tags: {rec['tags']}")
        
        print("\n" + "="*70 + "\n")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python recommend_movies.py <meme_image_path> [movies_csv_path]")
        print("\nExample:")
        print("  python recommend_movies.py meme.jpg movies.csv")
        print("\nIf movies_csv_path is not provided, it defaults to 'movies.csv'")
        sys.exit(1)
    
    meme_path = sys.argv[1]
    csv_path = sys.argv[2] if len(sys.argv) > 2 else "movies.csv"
    
    try:
        # Initialize recommender
        recommender = MemeMovieRecommender()
        
        # Load movies
        recommender.load_movies_from_csv(csv_path)
        
        # Get recommendations
        recommendations = recommender.recommend_from_meme(meme_path, top_k=5)
        
        # Display results
        recommender.print_recommendations(recommendations)
        
    except Exception as e:
        print(f"\n[✗] Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
    
    