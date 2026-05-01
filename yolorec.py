"""
YOLO-based Meme Movie Recommendation System
Uses YOLOv8 object detection to find movies similar to objects detected in your meme images.
"""

import os
import sys
import csv
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
from collections import Counter
import difflib


class YoloMemeMovieRecommender:
    def __init__(self, model_name="yolov8n.pt"):
        """Initialize YOLOv8 model."""
        print("[*] Loading YOLOv8 model (this may take a moment on first run)...")
        
        try:
            self.model = YOLO(model_name)
            print(f"[✓] YOLOv8 model '{model_name}' loaded successfully!")
        except Exception as e:
            print(f"[✗] Error loading model: {e}")
            sys.exit(1)
        
        self.movies = []
        self.movie_objects = {}
        print(f"[✓] Ready to detect objects in memes!")
    
    def load_movies_from_csv(self, csv_path):
        """Load movies from CSV file and extract keywords."""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        print(f"\n[*] Loading movies from {csv_path}...")
        self.movies = []
        self.movie_objects = {}
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                self.movies.append(row)
                
                # Extract all text for keyword matching
                combined_text = f"{row.get('title', '')} {row.get('genres', '')} {row.get('description', '')} {row.get('tags', '')}".lower()
                self.movie_objects[idx] = combined_text
        
        print(f"[✓] Loaded {len(self.movies)} movies")
    
    def detect_objects_in_meme(self, meme_path):
        """
        Detect objects in the meme image using YOLOv8.
        
        Args:
            meme_path: Path to the meme image
            
        Returns:
            List of detected object classes and their confidence scores
        """
        if not os.path.exists(meme_path):
            raise FileNotFoundError(f"Meme image not found: {meme_path}")
        
        print(f"\n[*] Processing meme with YOLOv8: {meme_path}")
        
        # Run YOLO detection
        results = self.model(meme_path, verbose=False)
        
        detected_objects = []
        object_counts = Counter()
        
        # Extract detected classes and confidence scores
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = result.names[class_id]
                
                detected_objects.append({
                    'class': class_name,
                    'confidence': confidence
                })
                object_counts[class_name] += 1
        
        if detected_objects:
            print(f"[✓] Detected {len(detected_objects)} objects:")
            for obj in sorted(detected_objects, key=lambda x: x['confidence'], reverse=True):
                print(f"   - {obj['class']}: {obj['confidence']:.2%} confidence")
        else:
            print("[!] No objects detected in the meme")
        
        return detected_objects
    
    def recommend_from_meme(self, meme_path, top_k=5):
        """
        Recommend movies based on objects detected in a meme image.
        
        Args:
            meme_path: Path to the meme image
            top_k: Number of recommendations to return
            
        Returns:
            List of recommendations with match scores
        """
        if not self.movies:
            raise ValueError("No movies loaded. Call load_movies_from_csv() first.")
        
        # Detect objects in meme
        detected_objects = self.detect_objects_in_meme(meme_path)
        
        if not detected_objects:
            print("[!] No objects detected - returning random movies")
            return self.movies[:top_k]
        
        # Extract unique object classes
        detected_classes = list(set(obj['class'] for obj in detected_objects))
        detected_text = " ".join(detected_classes).lower()
        
        print(f"\n[*] Searching for movies matching: {detected_classes}")
        
        # Score each movie based on keyword matches
        movie_scores = []
        
        for idx, movie in enumerate(self.movies):
            movie_text = self.movie_objects[idx]
            score = 0
            matched_keywords = []
            
            # Check for direct matches
            for obj_class in detected_classes:
                if obj_class in movie_text:
                    score += 2  # Higher weight for direct matches
                    matched_keywords.append(obj_class)
            
            # Use difflib for fuzzy matching on keywords
            for obj_class in detected_classes:
                for word in movie_text.split():
                    if len(word) > 3:  # Only match words longer than 3 chars
                        similarity = difflib.SequenceMatcher(None, obj_class, word).ratio()
                        if similarity > 0.7:
                            score += similarity
            
            if score > 0:
                movie_scores.append({
                    'idx': idx,
                    'score': score,
                    'matched_keywords': matched_keywords
                })
        
        # Sort by score and get top-k
        movie_scores.sort(key=lambda x: x['score'], reverse=True)
        top_matches = movie_scores[:top_k]
        
        recommendations = []
        for rank, match in enumerate(top_matches, 1):
            idx = match['idx']
            movie = self.movies[idx]
            
            recommendations.append({
                'rank': rank,
                'title': movie.get('title', 'Unknown'),
                'genres': movie.get('genres', 'Unknown'),
                'description': movie.get('description', 'No description'),
                'tags': movie.get('tags', ''),
                'match_score': match['score'],
                'matched_keywords': match['matched_keywords']
            })
        
        return recommendations
    
    def print_recommendations(self, recommendations):
        """Pretty print recommendations."""
        print("\n" + "="*70)
        print("🎬 MOVIE RECOMMENDATIONS BASED ON YOLO DETECTION")
        print("="*70)
        
        for rec in recommendations:
            print(f"\n#{rec['rank']} - {rec['title']}")
            print(f"   Genres: {rec['genres']}")
            print(f"   Match Score: {rec['match_score']:.2f}")
            if rec['matched_keywords']:
                print(f"   Matched Objects: {', '.join(rec['matched_keywords'])}")
            print(f"   Description: {rec['description']}")
            if rec['tags']:
                print(f"   Tags: {rec['tags']}")
        
        print("\n" + "="*70 + "\n")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python meme_movie_recommender_yolo.py <meme_image_path> [movies_csv_path]")
        print("\nExample:")
        print("  python meme_movie_recommender_yolo.py meme.jpg movies.csv")
        print("\nIf movies_csv_path is not provided, it defaults to 'movies.csv'")
        sys.exit(1)
    
    meme_path = sys.argv[1]
    csv_path = sys.argv[2] if len(sys.argv) > 2 else "movies.csv"
    
    try:
        # Initialize recommender
        recommender = YoloMemeMovieRecommender()
        
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
