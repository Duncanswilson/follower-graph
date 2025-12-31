"""Semantic clustering using tweet content and bio with sentence embeddings."""

import re
import os
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
    
    def generate_cluster_names_with_llm(
        self,
        mutuals: List[Dict],
        tweets_by_user: Dict[str, List[str]],
        openai_api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        max_samples_per_cluster: int = 8
    ) -> Dict[int, Dict]:
        """Generate cluster names using OpenAI by analyzing sample users from each cluster.
        
        Args:
            mutuals: List of mutual user dictionaries (must have cluster_id assigned)
            tweets_by_user: Dictionary mapping user_id to list of tweets
            openai_api_key: OpenAI API key (if None, uses OPENAI_API_KEY env var)
            model: OpenAI model to use (default: gpt-4o-mini)
            max_samples_per_cluster: Maximum number of users to sample per cluster
            
        Returns:
            Updated cluster_keywords dictionary with LLM-generated names
        """
        # Build cluster_labels dictionary from mutuals
        cluster_labels: Dict[int, List[int]] = {}
        for idx, mutual in enumerate(mutuals):
            cluster_id = mutual.get('cluster_id')
            if cluster_id is not None:
                if cluster_id not in cluster_labels:
                    cluster_labels[cluster_id] = []
                cluster_labels[cluster_id].append(idx)
        # Try to get API key
        api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("⚠ No OpenAI API key found. Falling back to TF-IDF naming.")
            return self.cluster_keywords
        
        try:
            from openai import OpenAI
        except ImportError:
            print("⚠ openai package not installed. Install with: pip install openai")
            print("  Falling back to TF-IDF naming.")
            return self.cluster_keywords
        
        client = OpenAI(api_key=api_key)
        
        print(f"Generating cluster names with {model}...")
        
        updated_keywords = self.cluster_keywords.copy()
        
        for cluster_id, user_indices in cluster_labels.items():
            if cluster_id == -1:  # Skip noise cluster
                continue
            
            if not user_indices:
                continue
            
            # Sample users from this cluster (prioritize those with more content)
            sampled_indices = self._sample_cluster_users(
                mutuals, 
                tweets_by_user, 
                user_indices, 
                max_samples_per_cluster
            )
            
            if not sampled_indices:
                continue
            
            # Build prompt with sample users
            prompt = self._build_cluster_naming_prompt(
                mutuals, 
                tweets_by_user, 
                sampled_indices
            )
            
            try:
                # Call OpenAI API
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful assistant that generates concise, descriptive names for groups of Twitter users based on their bios and tweets. Respond with only the cluster name (2-4 words), no explanation."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.3,
                    max_tokens=20
                )
                
                cluster_name = response.choices[0].message.content.strip()
                # Clean up the response (remove quotes if present, limit length)
                cluster_name = cluster_name.strip('"\'')
                if len(cluster_name) > 50:
                    cluster_name = cluster_name[:47] + "..."
                
                # Update cluster keywords
                if cluster_id in updated_keywords:
                    updated_keywords[cluster_id]['name'] = cluster_name
                else:
                    updated_keywords[cluster_id] = {
                        'name': cluster_name,
                        'keywords': []
                    }
                
                print(f"  Cluster {cluster_id}: {cluster_name}")
                
            except Exception as e:
                print(f"  ⚠ Error generating name for cluster {cluster_id}: {e}")
                print(f"    Falling back to TF-IDF name: {updated_keywords.get(cluster_id, {}).get('name', f'Cluster {cluster_id + 1}')}")
                continue
        
        print("✓ Cluster names generated")
        
        # Update cluster_keywords
        self.cluster_keywords = updated_keywords
        
        # Update cluster names in mutuals
        for mutual in mutuals:
            cluster_id = mutual.get('cluster_id')
            if cluster_id is not None and cluster_id in updated_keywords:
                mutual['cluster_name'] = updated_keywords[cluster_id]['name']
        
        return updated_keywords
    
    def _sample_cluster_users(
        self,
        mutuals: List[Dict],
        tweets_by_user: Dict[str, List[str]],
        user_indices: List[int],
        max_samples: int
    ) -> List[int]:
        """Sample users from a cluster, prioritizing those with more content."""
        # Score users by content richness (bio length + tweet count)
        scored_users = []
        for idx in user_indices:
            if idx >= len(mutuals):
                continue
            mutual = mutuals[idx]
            user_id = mutual['user_id']
            tweets = tweets_by_user.get(user_id, [])
            
            bio_length = len(mutual.get('description', '') or '')
            tweet_count = len(tweets)
            content_score = bio_length + (tweet_count * 50)  # Weight tweets more
            
            scored_users.append((idx, content_score))
        
        # Sort by score descending and take top N
        scored_users.sort(key=lambda x: x[1], reverse=True)
        sampled = [idx for idx, _ in scored_users[:max_samples]]
        
        return sampled
    
    def _build_cluster_naming_prompt(
        self,
        mutuals: List[Dict],
        tweets_by_user: Dict[str, List[str]],
        user_indices: List[int]
    ) -> str:
        """Build a prompt for OpenAI to generate a cluster name."""
        prompt_parts = [
            "Given these Twitter users who form a cluster, generate a 2-4 word descriptive name:",
            ""
        ]
        
        for idx in user_indices:
            if idx >= len(mutuals):
                continue
            
            mutual = mutuals[idx]
            user_id = mutual['user_id']
            screen_name = mutual.get('screen_name', user_id)
            bio = mutual.get('description', '') or ''
            tweets = tweets_by_user.get(user_id, [])
            
            # Take 2-3 most representative tweets (first few, skipping RTs)
            sample_tweets = []
            for tweet in tweets[:5]:  # Check first 5 tweets
                if tweet and not tweet.strip().startswith('RT @'):
                    cleaned = self.preprocess_text(tweet)
                    if cleaned and len(cleaned) > 10:
                        sample_tweets.append(cleaned)
                        if len(sample_tweets) >= 3:
                            break
            
            user_info = f"User (@{screen_name}):"
            if bio:
                user_info += f'\nBio: "{bio}"'
            if sample_tweets:
                tweet_text = ' | '.join([f'"{t}"' for t in sample_tweets])
                user_info += f'\nTweets: {tweet_text}'
            
            prompt_parts.append(user_info)
            prompt_parts.append("")
        
        prompt_parts.append("Respond with just the cluster name, no explanation.")
        
        return '\n'.join(prompt_parts)
    
    def get_cluster_info(self) -> Dict[int, Dict]:
        """Get cluster information including keywords."""
        return self.cluster_keywords.copy()


# Backwards compatibility alias
BioClustering = SemanticClustering
