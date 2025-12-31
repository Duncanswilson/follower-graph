"""D3.js interactive visualization for ego network."""

import shutil
from pathlib import Path
from typing import Dict
import networkx as nx

from src.exporter import GraphExporter


class NetworkVisualizer:
    """Creates interactive HTML visualization using d3.js."""
    
    def __init__(self):
        """Initialize visualizer."""
        pass
    
    def visualize(
        self, 
        graph: nx.Graph, 
        cluster_info: Dict[int, Dict],
        output_file: str = "output/ego_network.html",
        ego_username: str = "ego"
    ) -> str:
        """Create interactive visualization from NetworkX graph."""
        output_path = Path(output_file)
        output_dir = output_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Export graph to JSON
        exporter = GraphExporter(cluster_info)
        json_file = output_dir / "graph.json"
        exporter.export(graph, ego_username, str(json_file))
        
        # Copy HTML template
        template_path = Path(__file__).parent.parent / "templates" / "graph.html"
        if not template_path.exists():
            raise FileNotFoundError(f"Template not found: {template_path}")
        
        shutil.copy(template_path, output_path)
        
        print(f"✓ Visualization saved to {output_file}")
        print(f"  Graph data: {json_file}")
        
        return str(output_path)

