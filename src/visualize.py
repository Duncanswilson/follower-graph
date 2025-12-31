"""D3.js interactive visualization for ego network."""

import json
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
        
        # Export graph to JSON file (for reference/debugging)
        exporter = GraphExporter(cluster_info)
        json_file = output_dir / "graph.json"
        exporter.export(graph, ego_username, str(json_file))
        
        # Read the exported JSON data
        with open(json_file, 'r', encoding='utf-8') as f:
            graph_json = f.read()
        
        # Read HTML template
        template_path = Path(__file__).parent.parent / "templates" / "graph.html"
        if not template_path.exists():
            raise FileNotFoundError(f"Template not found: {template_path}")
        
        with open(template_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Embed graph data into HTML by replacing the placeholder
        # The template has: const EMBEDDED_GRAPH_DATA = /* GRAPH_DATA_PLACEHOLDER */ null;
        html_content = html_content.replace(
            '/* GRAPH_DATA_PLACEHOLDER */ null',
            graph_json
        )
        
        # Write the final HTML with embedded data
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✓ Visualization saved to {output_file}")
        print(f"  Graph data embedded in HTML (also saved to {json_file})")
        
        return str(output_path)

