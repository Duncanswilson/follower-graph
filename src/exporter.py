"""JSON graph exporter for d3.js visualization."""

import json
import math
from pathlib import Path
from typing import Dict, List
import networkx as nx


class GraphExporter:
    """Exports NetworkX graph to JSON format for d3.js."""
    
    def __init__(self, cluster_info: Dict[int, Dict]):
        """Initialize exporter with cluster information."""
        self.cluster_info = cluster_info
        self.cluster_colors = self._generate_cluster_colors()
    
    def _generate_cluster_colors(self) -> Dict[int, str]:
        """Generate color mapping for clusters."""
        colors = [
            '#FF6B6B',  # Red
            '#4ECDC4',  # Teal
            '#45B7D1',  # Blue
            '#FFA07A',  # Light Salmon
            '#98D8C8',  # Mint
            '#F7DC6F',  # Yellow
            '#BB8FCE',  # Purple
            '#85C1E2',  # Sky Blue
            '#F8B88B',  # Peach
            '#82E0AA',  # Green
            '#F1948A',  # Pink
            '#F4D03F',  # Gold
        ]
        
        color_map = {}
        for cluster_id in self.cluster_info.keys():
            color_index = cluster_id % len(colors)
            color_map[cluster_id] = colors[color_index]
        
        return color_map
    
    def _calculate_node_size(self, followers_count: int, is_ego: bool = False) -> float:
        """Calculate node size based on follower count."""
        if is_ego:
            return 50.0
        
        if followers_count <= 0:
            return 8.0
        
        # Logarithmic scaling: 8-25px range
        log_followers = math.log10(max(followers_count, 1))
        size = 8.0 + log_followers * 2.5
        return min(25.0, max(8.0, size))
    
    def export(self, graph: nx.Graph, ego_username: str, output_file: str) -> str:
        """Export graph to JSON format for d3.js."""
        nodes = []
        links = []
        clusters = []
        
        # Build cluster metadata
        for cluster_id, info in sorted(self.cluster_info.items()):
            clusters.append({
                'id': cluster_id,
                'name': info.get('name', f'Cluster {cluster_id}'),
                'keywords': info.get('keywords', []),
                'color': self.cluster_colors.get(cluster_id, '#CCCCCC')
            })
        
        # Extract nodes
        for node_id, data in graph.nodes(data=True):
            node_type = data.get('node_type', 'mutual')
            is_ego = (node_type == 'ego')
            
            # Get cluster info
            cluster_id = data.get('cluster_id', 0) if not is_ego else -1
            cluster_name = data.get('cluster_name', 'Uncategorized') if not is_ego else 'Ego'
            
            # Calculate size
            followers_count = data.get('followers_count', 0)
            size = self._calculate_node_size(followers_count, is_ego)
            
            # Build node object
            node_obj = {
                'id': str(node_id),
                'label': data.get('screen_name', data.get('label', str(node_id))),
                'name': data.get('name', ''),
                'group': cluster_id,
                'size': size,
                'isEgo': is_ego,
                'bio': data.get('bio', data.get('description', ''))[:200],  # Truncate bio
                'followers': followers_count,
                'verified': data.get('verified', False),
                'clusterName': cluster_name
            }
            
            nodes.append(node_obj)
        
        # Extract links
        for source, target in graph.edges():
            links.append({
                'source': str(source),
                'target': str(target)
            })
        
        # Build final JSON structure
        graph_data = {
            'nodes': nodes,
            'links': links,
            'clusters': clusters,
            'metadata': {
                'ego': ego_username,
                'nodeCount': len(nodes),
                'linkCount': len(links)
            }
        }
        
        # Write to file
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Exported graph to {output_file}")
        print(f"  Nodes: {len(nodes)}, Links: {len(links)}, Clusters: {len(clusters)}")
        
        return str(output_path)

