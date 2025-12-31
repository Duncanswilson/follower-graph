"""Semantic clustering using tweet content and bio with sentence embeddings."""

import re
from typing import List, Dict, Tuple, Optional
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer


class SemanticClustering:
    """Cluster users by their tweets and bio using semantic embeddings."""
    
    def __init__(self, min_clusters: int = 5, max_clusters: int = 12):
        """Initialize clustering with cluster range."""
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.kmeans = None
        self.cluster_keywords = {}
        self.embedder = None
        self._tfidf_vectorizer = None
        self._tfidf_feature_names = None
    
    def _get_embedder(self):
        """Lazy load the sentence transformer model."""
        if self.embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
                print("Loading sentence transformer model...")
                # Use a lightweight but effective model
                self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
                print("✓ Model loaded")
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for semantic clustering. "
                    "Install with: pip install sentence-transformers"
                )
        return self.embedder
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text: remove URLs, mentions, extra whitespace."""
        if not text:
            return ""
        
        # Remove URLs
        text = re.sub(r'http\S+|www\.\S+', '', text)
        # Remove @mentions
        text = re.sub(r'@\w+', '', text)
        # Remove RT prefix
        text = re.sub(r'^RT\s+', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def combine_user_content(
        self, 
        mutual: Dict, 
        tweets: List[str],
        max_tweets: int = 10
    ) -> str:
        """Combine bio and tweets into a single text for embedding."""
        parts = []
        
        # Add bio (weight it by including twice)
        bio = self.preprocess_text(mutual.get('description', ''))
        if bio:
            parts.append(bio)
            parts.append(bio)  # Double weight for bio
        
        # Add tweets (limited to avoid overwhelming)
        for tweet in tweets[:max_tweets]:
            cleaned = self.preprocess_text(tweet)
            if cleaned and len(cleaned) > 10:  # Skip very short tweets
                parts.append(cleaned)
        
        return ' '.join(parts)
    
    def prepare_texts(
        self, 
        mutuals: List[Dict],
        tweets_by_user: Dict[str, List[str]]
    ) -> Tuple[List[str], List[int]]:
        """Prepare combined text for each user, tracking empty ones."""
        texts = []
        empty_indices = []
        
        for i, mutual in enumerate(mutuals):
            user_id = mutual['user_id']
            tweets = tweets_by_user.get(user_id, [])
            
            combined = self.combine_user_content(mutual, tweets)
            texts.append(combined)
            
            if not combined or len(combined.strip()) < 10:
                empty_indices.append(i)
        
        return texts, empty_indices
    
    def compute_embeddings(self, texts: List[str]) -> np.ndarray:
        """Compute semantic embeddings for texts."""
        embedder = self._get_embedder()
        
        print(f"Computing embeddings for {len(texts)} users...")
        embeddings = embedder.encode(
            texts, 
            show_progress_bar=True,
            convert_to_numpy=True
        )
        print("✓ Embeddings computed")
        
        return embeddings
    
    def find_optimal_k(self, vectors: np.ndarray, min_k: int, max_k: int) -> int:
        """Find optimal number of clusters using silhouette score."""
        if len(vectors) < min_k:
            return max(2, len(vectors) // 2)
        
        best_k = min_k
        best_score = -1
        
        print(f"Finding optimal k between {min_k} and {max_k}...")
        
        for k in range(min_k, min(max_k + 1, len(vectors))):
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(vectors)
                
                if len(set(labels)) < 2:
                    continue
                
                score = silhouette_score(vectors, labels)
                print(f"  k={k}: silhouette_score={score:.3f}")
                
                if score > best_score:
                    best_score = score
                    best_k = k
            except Exception as e:
                print(f"  k={k}: Error - {e}")
                continue
        
        print(f"✓ Optimal k={best_k} (silhouette_score={best_score:.3f})")
        return best_k
    
    def extract_cluster_keywords(
        self, 
        texts: List[str], 
        labels: np.ndarray,
        n_keywords: int = 5
    ) -> Dict[int, Dict]:
        """Extract representative keywords for each cluster using TF-IDF."""
        # Fit TF-IDF on all texts
        self._tfidf_vectorizer = TfidfVectorizer(
            max_features=200,
            stop_words='english',
            min_df=2,
            ngram_range=(1, 2)
        )
        
        try:
            tfidf_matrix = self._tfidf_vectorizer.fit_transform(texts)
            self._tfidf_feature_names = self._tfidf_vectorizer.get_feature_names_out()
        except ValueError:
            # Not enough text to extract keywords
            return {}
        
        cluster_keywords = {}
        unique_labels = set(labels)
        
        for cluster_id in unique_labels:
            if cluster_id == -1:  # Skip noise cluster if using DBSCAN
                continue
            
            # Get indices of documents in this cluster
            cluster_mask = labels == cluster_id
            cluster_tfidf = tfidf_matrix[cluster_mask]
            
            if cluster_tfidf.shape[0] == 0:
                continue
            
            # Average TF-IDF scores for this cluster
            mean_tfidf = np.asarray(cluster_tfidf.mean(axis=0)).flatten()
            
            # Get top keywords
            top_indices = mean_tfidf.argsort()[-n_keywords:][::-1]
            keywords = [
                self._tfidf_feature_names[i] 
                for i in top_indices 
                if mean_tfidf[i] > 0
            ]
            
            cluster_name = ', '.join(keywords[:3]) if keywords else f'Cluster {int(cluster_id) + 1}'
            cluster_keywords[int(cluster_id)] = {
                'name': cluster_name,
                'keywords': keywords
            }
        
        return cluster_keywords
    
    def cluster(
        self, 
        mutuals: List[Dict],
        tweets_by_user: Dict[str, List[str]],
        n_clusters: Optional[int] = None,
        auto_detect: bool = False
    ) -> List[Dict]:
        """Cluster mutuals by tweet content and bio using semantic embeddings."""
        if len(mutuals) == 0:
            return mutuals
        
        # Prepare combined texts
        texts, empty_indices = self.prepare_texts(mutuals, tweets_by_user)
        
        # Filter out empty texts for embedding
        non_empty_texts = [t for t in texts if t and len(t.strip()) >= 10]
        non_empty_indices = [i for i, t in enumerate(texts) if t and len(t.strip()) >= 10]
        
        if len(non_empty_texts) == 0:
            # No usable content - assign all to uncategorized
            for mutual in mutuals:
                mutual['cluster_id'] = 0
                mutual['cluster_name'] = 'Uncategorized'
            return mutuals
        
        # Compute semantic embeddings
        embeddings = self.compute_embeddings(non_empty_texts)
        
        # Determine number of clusters
        if auto_detect:
            optimal_k = self.find_optimal_k(
                embeddings, 
                self.min_clusters, 
                self.max_clusters
            )
            n_clusters = optimal_k
        elif n_clusters is None:
            n_clusters = min(self.max_clusters, max(self.min_clusters, len(non_empty_texts) // 10))
        
        # Ensure n_clusters doesn't exceed number of samples
        n_clusters = min(n_clusters, len(non_empty_texts))
        
        # Perform clustering
        print(f"Clustering {len(non_empty_texts)} users into {n_clusters} clusters...")
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = self.kmeans.fit_predict(embeddings)
        
        # Extract keywords for cluster naming
        self.cluster_keywords = self.extract_cluster_keywords(
            non_empty_texts, 
            cluster_labels
        )
        
        # Ensure all cluster IDs are native Python ints (not numpy int32)
        self.cluster_keywords = {
            int(k): v for k, v in self.cluster_keywords.items()
        }
        
        # Fill in any missing cluster names
        for cluster_id in range(n_clusters):
            if cluster_id not in self.cluster_keywords:
                self.cluster_keywords[cluster_id] = {
                    'name': f'Cluster {cluster_id + 1}',
                    'keywords': []
                }
        
        # Assign cluster labels to mutuals
        for idx, mutual_idx in enumerate(non_empty_indices):
            cluster_id = int(cluster_labels[idx])  # Convert numpy int to Python int
            mutuals[mutual_idx]['cluster_id'] = cluster_id
            mutuals[mutual_idx]['cluster_name'] = self.cluster_keywords[cluster_id]['name']
        
        # Assign empty content users to uncategorized cluster
        uncategorized_id = int(n_clusters)
        for idx in empty_indices:
            mutuals[idx]['cluster_id'] = uncategorized_id
            mutuals[idx]['cluster_name'] = 'Uncategorized'
        
        # Add uncategorized to cluster_keywords if needed
        if empty_indices:
            self.cluster_keywords[uncategorized_id] = {
                'name': 'Uncategorized',
                'keywords': []
            }
        
        print(f"✓ Clustered {len(mutuals)} users into {len(self.cluster_keywords)} clusters")
        
        return mutuals
    
    def get_cluster_info(self) -> Dict[int, Dict]:
        """Get cluster information including keywords."""
        return self.cluster_keywords.copy()


# Backwards compatibility alias
BioClustering = SemanticClustering
