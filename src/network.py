"""NetworkX graph construction for ego network."""

import networkx as nx
from typing import List, Dict


class EgoNetworkBuilder:
    """Builds an ego network graph from mutual connections."""
    
    def __init__(self, ego_username: str):
        """Initialize with ego (central) user."""
        self.ego_username = ego_username
        self.graph = nx.Graph()
    
    def build(self, mutuals: List[Dict]) -> nx.Graph:
        """Build ego network graph from mutuals data."""
        # Add ego node (central node)
        self.graph.add_node(
            self.ego_username,
            node_type='ego',
            label=self.ego_username,
            title=f'Ego: {self.ego_username}',
            size=50,  # Larger size for ego
            color='#FF6B6B'  # Distinct color for ego
        )
        
        # Add mutual nodes and edges
        for mutual in mutuals:
            user_id = mutual.get('user_id') or mutual.get('screen_name')
            screen_name = mutual.get('screen_name', '')
            name = mutual.get('name', screen_name)
            bio = mutual.get('description', '')
            cluster_id = mutual.get('cluster_id', 0)
            cluster_name = mutual.get('cluster_name', 'Uncategorized')
            followers_count = mutual.get('followers_count', 0)
            verified = mutual.get('verified', False)
            
            # Calculate node size based on follower count (capped)
            # Scale between 10 and 30
            if followers_count > 0:
                # Log scale for better distribution
                import math
                log_followers = math.log10(max(followers_count, 1))
                size = min(30, max(10, 10 + log_followers * 3))
            else:
                size = 15
            
            # Create tooltip
            tooltip = f"@{screen_name}\n{name}"
            if bio:
                bio_preview = bio[:100] + "..." if len(bio) > 100 else bio
                tooltip += f"\n\n{bio_preview}"
            if verified:
                tooltip += "\n✓ Verified"
            
            # Add node with attributes
            self.graph.add_node(
                user_id,
                node_type='mutual',
                label=screen_name,
                title=tooltip,
                screen_name=screen_name,
                name=name,
                bio=bio,
                cluster_id=cluster_id,
                cluster_name=cluster_name,
                followers_count=followers_count,
                verified=verified,
                size=size
            )
            
            # Add edge from ego to mutual (star topology)
            self.graph.add_edge(self.ego_username, user_id)
        
        print(f"✓ Built ego network: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
        
        return self.graph
    
    def get_cluster_distribution(self) -> Dict[int, int]:
        """Get distribution of nodes across clusters."""
        cluster_counts = {}
        for node, data in self.graph.nodes(data=True):
            if data.get('node_type') == 'mutual':
                cluster_id = data.get('cluster_id', 0)
                cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1
        return cluster_counts

