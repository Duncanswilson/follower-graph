#!/usr/bin/env python3
"""Main CLI entry point for Twitter ego network visualization."""

# Apply httpx compatibility patch FIRST, before any twikit imports
import httpx
from functools import wraps

# Patch AsyncClient to handle proxy parameter that Twikit passes
_original_async_client_init = httpx.AsyncClient.__init__

@wraps(_original_async_client_init)
def _patched_async_client_init(self, *args, proxy=None, **kwargs):
    """Patched init that ignores proxy parameter."""
    # Remove proxy from kwargs since httpx AsyncClient doesn't accept it
    kwargs.pop('proxy', None)
    return _original_async_client_init(self, *args, **kwargs)

httpx.AsyncClient.__init__ = _patched_async_client_init

import asyncio
import argparse
import getpass
import os
import sys
import webbrowser
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
from threading import Thread

from src.collector import TwitterCollector
from src.clustering import SemanticClustering
from src.network import EgoNetworkBuilder
from src.visualize import NetworkVisualizer


async def main():
    """Main async entry point."""
    parser = argparse.ArgumentParser(
        description="Build an interactive ego network visualization of your Twitter mutuals"
    )
    parser.add_argument(
        '--username',
        type=str,
        required=True,
        help='Your Twitter username (without @)'
    )
    parser.add_argument(
        '--email',
        type=str,
        help='Your Twitter email (required for first login)'
    )
    parser.add_argument(
        '--password',
        type=str,
        help='Your Twitter password (will prompt if not provided)'
    )
    parser.add_argument(
        '--clusters',
        type=int,
        help='Number of clusters (default: auto-detect)'
    )
    parser.add_argument(
        '--auto-clusters',
        action='store_true',
        help='Auto-detect optimal number of clusters'
    )
    parser.add_argument(
        '--min-clusters',
        type=int,
        default=5,
        help='Minimum number of clusters for auto-detection (default: 5)'
    )
    parser.add_argument(
        '--max-clusters',
        type=int,
        default=12,
        help='Maximum number of clusters for auto-detection (default: 12)'
    )
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Force fresh data fetch (ignore cache)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='output/ego_network.html',
        help='Output HTML file path (default: output/ego_network.html)'
    )
    parser.add_argument(
        '--serve',
        action='store_true',
        help='Start HTTP server to view visualization (opens browser automatically)'
    )
    parser.add_argument(
        '--mutual-edges',
        action='store_true',
        help='Collect mutual-to-mutual connections (enriches graph but takes longer)'
    )
    parser.add_argument(
        '--mutual-edges-sample',
        type=int,
        help='Limit mutual edge collection to top N users by followers (reduces API calls)'
    )
    parser.add_argument(
        '--mutual-edges-delay',
        type=float,
        default=2.0,
        help='Delay in seconds between API calls for mutual edges (default: 2.0, increase if hitting rate limits)'
    )
    parser.add_argument(
        '--no-adaptive-rate-limit',
        action='store_true',
        help='Disable adaptive rate limiting (use fixed delay instead)'
    )
    parser.add_argument(
        '--openai-key',
        type=str,
        help='OpenAI API key (or set OPENAI_API_KEY environment variable)'
    )
    parser.add_argument(
        '--no-llm-naming',
        action='store_true',
        help='Disable LLM-powered cluster naming (use TF-IDF keywords instead)'
    )
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Twitter Ego Network Visualization")
    print("=" * 60)
    print()
    
    # Initialize collector
    collector = TwitterCollector()
    
    # Handle authentication
    if not collector.cookies_file.exists() or args.no_cache:
        if not args.email:
            args.email = input("Enter your Twitter email: ").strip()
        
        if not args.password:
            args.password = getpass.getpass("Enter your Twitter password: ")
        
        if not args.email or not args.password:
            print("Error: Email and password are required for first login")
            sys.exit(1)
    
    # Login
    try:
        await collector.login(args.username, args.email or "", args.password or "")
    except Exception as e:
        print(f"Error logging in: {e}")
        print("Make sure your credentials are correct and that 2FA is disabled or handled.")
        sys.exit(1)
    
    # Collect mutuals
    try:
        print()
        mutuals = await collector.collect_mutuals(
            args.username,
            use_cache=not args.no_cache
        )
        
        if len(mutuals) == 0:
            print("No mutual connections found!")
            sys.exit(0)
        
        print(f"\nFound {len(mutuals)} mutual connections")
    except Exception as e:
        print(f"Error collecting mutuals: {e}")
        sys.exit(1)
    
    # Fetch tweets for semantic clustering
    try:
        print()
        tweets_by_user = await collector.fetch_tweets_for_mutuals(
            mutuals,
            tweets_per_user=20,
            use_cache=not args.no_cache,
            rate_limit_delay=0.5
        )
    except Exception as e:
        print(f"Warning: Error fetching tweets: {e}")
        print("Falling back to bio-only clustering...")
        tweets_by_user = {}
    
    # Cluster mutuals using semantic analysis
    try:
        print()
        clustering = SemanticClustering(
            min_clusters=args.min_clusters,
            max_clusters=args.max_clusters
        )
        
        auto_detect = args.auto_clusters or args.clusters is None
        mutuals = clustering.cluster(
            mutuals,
            tweets_by_user=tweets_by_user,
            n_clusters=args.clusters,
            auto_detect=auto_detect
        )
        
        # Generate LLM-powered cluster names if enabled
        if not args.no_llm_naming:
            try:
                print()
                openai_key = args.openai_key or os.getenv('OPENAI_API_KEY')
                if openai_key:
                    clustering.generate_cluster_names_with_llm(
                        mutuals,
                        tweets_by_user,
                        openai_api_key=openai_key
                    )
                else:
                    print("⚠ No OpenAI API key provided. Using TF-IDF cluster names.")
                    print("  Set OPENAI_API_KEY env var or use --openai-key to enable LLM naming.")
            except Exception as e:
                print(f"⚠ Error generating LLM cluster names: {e}")
                print("  Falling back to TF-IDF cluster names.")
        
        cluster_info = clustering.get_cluster_info()
        print(f"\nCluster distribution:")
        for cluster_id, info in sorted(cluster_info.items()):
            count = sum(1 for m in mutuals if m.get('cluster_id') == cluster_id)
            print(f"  {info['name']}: {count} users")
    except Exception as e:
        print(f"Error clustering: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Collect mutual-to-mutual edges (optional)
    mutual_edges = []
    if args.mutual_edges:
        try:
            print()
            # Get ego user ID to exclude from mutual edges
            ego_user_id = await collector.get_user_id(args.username)
            mutual_edges = await collector.fetch_mutual_connections(
                mutuals,
                ego_user_id=ego_user_id,
                use_cache=not args.no_cache,
                rate_limit_delay=args.mutual_edges_delay,
                sample_size=args.mutual_edges_sample,
                use_adaptive=not args.no_adaptive_rate_limit
            )
        except Exception as e:
            print(f"Warning: Error collecting mutual edges: {e}")
            print("Continuing without mutual edges...")
            mutual_edges = []
    
    # Build network
    try:
        print()
        network_builder = EgoNetworkBuilder(args.username)
        graph = network_builder.build(mutuals)
        
        # Add mutual-to-mutual edges if collected
        if mutual_edges:
            network_builder.add_mutual_edges(mutual_edges)
            # Graph is updated in-place, so no need to reassign
    except Exception as e:
        print(f"Error building network: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Visualize
    try:
        print()
        visualizer = NetworkVisualizer()
        output_file = visualizer.visualize(
            graph,
            cluster_info,
            output_file=args.output,
            ego_username=args.username
        )
        
        print()
        print("=" * 60)
        print("✓ Visualization complete!")
        
        if args.serve:
            # Start HTTP server
            output_dir = Path(output_file).parent.absolute()
            html_file = Path(output_file).name
            
            class Handler(SimpleHTTPRequestHandler):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, directory=str(output_dir), **kwargs)
            
            port = 8000
            server = HTTPServer(('localhost', port), Handler)
            
            url = f"http://localhost:{port}/{html_file}"
            print(f"  Starting HTTP server on http://localhost:{port}")
            print(f"  Opening {url} in your browser...")
            
            # Open browser
            webbrowser.open(url)
            
            # Start server in background thread
            def run_server():
                server.serve_forever()
            
            server_thread = Thread(target=run_server, daemon=True)
            server_thread.start()
            
            print("  Press Ctrl+C to stop the server")
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                print("\n  Shutting down server...")
                server.shutdown()
        else:
            print(f"  Open {output_file} in your browser to explore your network.")
            print(f"  Or run with --serve to start a local HTTP server.")
        
        print("=" * 60)
    except Exception as e:
        print(f"Error visualizing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

