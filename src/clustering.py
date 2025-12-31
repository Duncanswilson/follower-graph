"""Bio-based clustering using TF-IDF and K-Means."""

import re
from typing import List, Dict, Tuple, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


class BioClustering:
    """Cluster users by their bio content using TF-IDF and K-Means."""
    
    def __init__(self, min_clusters: int = 5, max_clusters: int = 12):
        """Initialize clustering with cluster range."""
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.vectorizer = None
        self.kmeans = None
        self.cluster_keywords = {}
        self.feature_names = None
    
    def preprocess_bio(self, bio: str) -> str:
        """Preprocess bio text: lowercase, remove URLs, mentions, extra whitespace."""
        if not bio:
            return ""
        
        # Remove URLs
        bio = re.sub(r'http\S+|www\.\S+', '', bio)
        # Remove @mentions
        bio = re.sub(r'@\w+', '', bio)
        # Remove hashtags (optional - keeping them might be useful)
        # bio = re.sub(r'#\w+', '', bio)
        # Remove extra whitespace
        bio = re.sub(r'\s+', ' ', bio)
        # Lowercase
        bio = bio.lower().strip()
        
        return bio
    
    def prepare_bios(self, mutuals: List[Dict]) -> Tuple[List[str], List[int]]:
        """Extract and preprocess bios, tracking indices for empty bios."""
        bios = []
        empty_indices = []
        
        for i, mutual in enumerate(mutuals):
            bio = self.preprocess_bio(mutual.get('description', ''))
            bios.append(bio)
            if not bio or len(bio.strip()) == 0:
                empty_indices.append(i)
        
        return bios, empty_indices
    
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
                
                # Silhouette score requires at least 2 clusters and 2 samples per cluster
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
    
    def extract_keywords(self, cluster_id: int, n_keywords: int = 5) -> List[str]:
        """Extract top keywords for a cluster."""
        if self.kmeans is None or self.feature_names is None:
            return []
        
        # Get cluster center
        center = self.kmeans.cluster_centers_[cluster_id]
        
        # Get top feature indices
        top_indices = center.argsort()[-n_keywords:][::-1]
        
        # Get corresponding feature names
        keywords = [self.feature_names[i] for i in top_indices if center[i] > 0]
        
        return keywords
    
    def cluster(
        self, 
        mutuals: List[Dict], 
        n_clusters: Optional[int] = None,
        auto_detect: bool = False
    ) -> List[Dict]:
        """Cluster mutuals by bio content."""
        if len(mutuals) == 0:
            return mutuals
        
        # Prepare bios
        bios, empty_indices = self.prepare_bios(mutuals)
        
        # Filter out empty bios for clustering
        non_empty_bios = [bio for bio in bios if bio]
        non_empty_indices = [i for i, bio in enumerate(bios) if bio]
        
        if len(non_empty_bios) == 0:
            # All bios empty - assign all to uncategorized
            for mutual in mutuals:
                mutual['cluster_id'] = 0
                mutual['cluster_name'] = 'Uncategorized'
            return mutuals
        
        # Vectorize bios
        self.vectorizer = TfidfVectorizer(
            max_features=100,
            stop_words='english',
            min_df=2,  # Ignore terms that appear in fewer than 2 documents
            ngram_range=(1, 2)  # Unigrams and bigrams
        )
        
        vectors = self.vectorizer.fit_transform(non_empty_bios)
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        # Determine number of clusters
        if auto_detect:
            optimal_k = self.find_optimal_k(
                vectors.toarray(), 
                self.min_clusters, 
                self.max_clusters
            )
            n_clusters = optimal_k
        elif n_clusters is None:
            n_clusters = min(self.max_clusters, max(self.min_clusters, len(non_empty_bios) // 10))
        
        # Ensure n_clusters doesn't exceed number of samples
        n_clusters = min(n_clusters, len(non_empty_bios))
        
        # Perform clustering
        print(f"Clustering {len(non_empty_bios)} users into {n_clusters} clusters...")
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = self.kmeans.fit_predict(vectors)
        
        # Extract keywords for each cluster
        self.cluster_keywords = {}
        for cluster_id in range(n_clusters):
            keywords = self.extract_keywords(cluster_id)
            cluster_name = ', '.join(keywords[:3]) if keywords else f'Cluster {cluster_id + 1}'
            self.cluster_keywords[cluster_id] = {
                'name': cluster_name,
                'keywords': keywords
            }
        
        # Assign cluster labels to mutuals
        for idx, mutual_idx in enumerate(non_empty_indices):
            cluster_id = cluster_labels[idx]
            mutuals[mutual_idx]['cluster_id'] = int(cluster_id)
            mutuals[mutual_idx]['cluster_name'] = self.cluster_keywords[cluster_id]['name']
        
        # Assign empty bios to uncategorized cluster
        uncategorized_id = n_clusters  # Use next available ID
        for idx in empty_indices:
            mutuals[idx]['cluster_id'] = uncategorized_id
            mutuals[idx]['cluster_name'] = 'Uncategorized'
        
        # Update cluster_keywords to include uncategorized
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

