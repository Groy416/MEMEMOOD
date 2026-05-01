"""
Meme-based Movie Recommendation System using CNN (fixed v2)
Compatible with modern torchvision (0.17+), uses correct ResNet50 transforms.
"""

import os
import sys
import csv
from pathlib import Path
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights

from sklearn.metrics.pairwise import cosine_similarity


class MemeMovieRecommenderCNN:
    def __init__(self, seed=42):
        print("[*] Loading pre-trained ResNet50 model...")

        # device selection
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[*] Using device: {self.device}")

        # Load weights (correct API)
        self.weights = ResNet50_Weights.DEFAULT
        self.model = models.resnet50(weights=self.weights).to(self.device)

        # Remove final FC layer to get 2048-dim feature vectors
        self.feature_extractor = nn.Sequential(*list(self.model.children())[:-1]).to(self.device)
        self.feature_extractor.eval()

        # Correct modern torchvision transform pipeline
        self.transform = self.weights.transforms()

        # ResNet50 final feature size
        self.img_feature_dim = 2048

        # Keyword vector setup
        self.keywords = [
            'action', 'adventure', 'comedy', 'drama', 'horror', 'romance',
            'thriller', 'sci-fi', 'fantasy', 'animation', 'crime', 'mystery',
            'war', 'western', 'love', 'death', 'power', 'journey', 'hero',
            'villain', 'magic', 'space', 'time', 'future', 'past', 'friendship',
            'family', 'survival', 'revenge', 'dark', 'light', 'evil', 'good',
            'fight', 'explore'
        ]
        self.text_dim = len(self.keywords)
        self.seed = seed
        self._text_proj_matrix = None

        self.movies = []
        self.movie_features = None

        print("[✓] ResNet50 model loaded successfully.\n")

    # --------------------------------------------------------------

    def load_movies_from_csv(self, csv_path):
        csv_path = Path(csv_path)

        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        print(f"[*] Loading movies from {csv_path}...")

        self.movies = []
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.movies.append({k.strip(): v.strip() for k, v in row.items()})

        print(f"[✓] Loaded {len(self.movies)} movies")
        print("[*] Generating movie feature vectors...")

        self._generate_movie_features()

        print("[✓] Movie features ready.\n")

    # --------------------------------------------------------------

    def _generate_movie_features(self):
        features = []
        poster_used = 0

        for m in self.movies:
            poster_path = m.get("poster_path") or m.get("poster") or ""

            used_poster = False

            # Try poster features
            if poster_path:
                path = Path(poster_path).resolve()
                if path.exists():
                    try:
                        feat = self._extract_image_features(str(path))
                        features.append(feat)
                        poster_used += 1
                        used_poster = True
                    except Exception:
                        pass

            if not used_poster:
                # fallback: text → projection
                txt = " ".join([
                    m.get("title", ""),
                    m.get("genres", ""),
                    m.get("description", ""),
                    m.get("tags", "")
                ]).lower()

                txt_vec = self._text_to_feature_vector(txt)
                img_like_vec = self._project_text_to_image_space(txt_vec)
                features.append(img_like_vec)

        self.movie_features = np.vstack(features)
        # normalize
        norms = np.linalg.norm(self.movie_features, axis=1, keepdims=True)
        norms[norms == 0] = 1
        self.movie_features = self.movie_features / norms

        print(f"[i] Poster images used for {poster_used} movies")

    # --------------------------------------------------------------

    def _text_to_feature_vector(self, text):
        vec = np.zeros(self.text_dim, dtype=np.float32)

        for i, kw in enumerate(self.keywords):
            vec[i] = text.count(kw)

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm

        return vec

    # --------------------------------------------------------------

    def _init_text_proj(self):
        if self._text_proj_matrix is None:
            rng = np.random.default_rng(self.seed)
            M = rng.normal(0, 0.5, (self.text_dim, self.img_feature_dim))
            # normalize columns
            col_norms = np.linalg.norm(M, axis=0, keepdims=True)
            col_norms[col_norms == 0] = 1
            M /= col_norms
            self._text_proj_matrix = M.astype(np.float32)

        return self._text_proj_matrix

    def _project_text_to_image_space(self, txt_vec):
        M = self._init_text_proj()
        img_vec = txt_vec @ M

        norm = np.linalg.norm(img_vec)
        if norm > 0:
            img_vec /= norm

        return img_vec

    # --------------------------------------------------------------

    def _extract_image_features(self, image_path):
        with Image.open(image_path).convert("RGB") as img:
            t = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            f = self.feature_extractor(t)
            f = f.view(f.size(0), -1).cpu().numpy()[0]

        # normalize
        norm = np.linalg.norm(f)
        if norm > 0:
            f /= norm

        return f.astype(np.float32)

    # --------------------------------------------------------------

    def recommend_from_meme(self, meme_path, top_k=5):
        meme_path = Path(meme_path)
        if not meme_path.exists():
            raise FileNotFoundError(f"Meme image not found: {meme_path}")

        print(f"[*] Extracting features from meme: {meme_path}")
        meme_vec = self._extract_image_features(str(meme_path))

        sims = cosine_similarity(meme_vec.reshape(1, -1), self.movie_features)[0]
        top_idx = np.argsort(sims)[::-1][:top_k]

        recs = []
        for rank, idx in enumerate(top_idx, start=1):
            m = self.movies[idx]
            recs.append({
                "rank": rank,
                "title": m.get("title", "Unknown"),
                "genres": m.get("genres", "Unknown"),
                "description": m.get("description", ""),
                "tags": m.get("tags", ""),
                "similarity_score": float(sims[idx])
            })

        return recs

    # --------------------------------------------------------------

    def print_recommendations(self, recs):
        print("\n" + "="*70)
        print("🎬 MOVIE RECOMMENDATIONS BASED ON YOUR MEME")
        print("="*70)

        for r in recs:
            print(f"\n#{r['rank']} - {r['title']}")
            print(f"   Genres: {r['genres']}")
            print(f"   Similarity: {r['similarity_score']:.2%}")
            if r['description']:
                print(f"   Description: {r['description'][:200]}...")

        print("\n" + "="*70 + "\n")


# =================================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python cnn.py <meme_image> [movies.csv]")
        sys.exit(1)

    meme = sys.argv[1]
    csv_path = sys.argv[2] if len(sys.argv) > 2 else "movies.csv"

    try:
        r = MemeMovieRecommenderCNN()
        r.load_movies_from_csv(csv_path)
        recs = r.recommend_from_meme(meme, top_k=5)
        r.print_recommendations(recs)

    except Exception as e:
        print(f"\n[✗] Error: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
