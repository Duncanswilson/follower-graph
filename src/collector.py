"""Twitter data collector using Twikit library."""

import asyncio
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from twikit import Client


class AdaptiveRateLimiter:
    """Adaptive rate limiter that adjusts delays based on success/failure patterns."""
    
    def __init__(self, base_delay=2.0, min_delay=0.5, max_delay=30.0):
        """Initialize adaptive rate limiter.
        
        Args:
            base_delay: Starting delay between requests in seconds
            min_delay: Minimum delay allowed (won't go below this)
            max_delay: Maximum delay allowed (won't go above this)
        """
        self.current_delay = base_delay
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.consecutive_successes = 0
        self.speedup_threshold = 10  # successes before speeding up
        
    def on_success(self):
        """Call after a successful request - may reduce delay if doing well."""
        self.consecutive_successes += 1
        if self.consecutive_successes >= self.speedup_threshold:
            # Reduce delay by 20% (multiply by 0.8)
            new_delay = self.current_delay * 0.8
            self.current_delay = max(self.min_delay, new_delay)
            self.consecutive_successes = 0
            
    def on_rate_limit(self):
        """Call when hitting a rate limit - increases delay and resets success counter."""
        self.consecutive_successes = 0
        # Double the delay (up to max_delay)
        self.current_delay = min(self.max_delay, self.current_delay * 2)


class TwitterCollector:
    """Collects Twitter followers, following, and profile data using Twikit."""
    
    def __init__(self, cache_dir: str = "cache"):
        """Initialize collector with cache directory."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cookies_file = self.cache_dir / "cookies.json"
        self.mutuals_file = self.cache_dir / "mutuals.json"
        self.tweets_cache_file = self.cache_dir / "tweets_cache.json"
        self.mutual_edges_file = self.cache_dir / "mutual_edges.json"
        self.client = Client('en-US')
        self._logged_in = False
        self._tweets_cache = self._load_tweets_cache()
    
    def _extract_user_data(self, user_obj) -> Dict:
        """Extract user data from Twikit User object."""
        return {
            'user_id': str(user_obj.id),
            'screen_name': user_obj.screen_name,
            'name': user_obj.name,
            'description': getattr(user_obj, 'description', '') or '',
            'verified': getattr(user_obj, 'verified', False),
            'followers_count': getattr(user_obj, 'followers_count', 0),
        }
    
    def _load_tweets_cache(self) -> Dict[str, List[str]]:
        """Load tweets cache from file."""
        if self.tweets_cache_file.exists():
            try:
                with open(self.tweets_cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        return {}
    
    def _save_tweets_cache(self) -> None:
        """Save tweets cache to file."""
        with open(self.tweets_cache_file, 'w', encoding='utf-8') as f:
            json.dump(self._tweets_cache, f, indent=2, ensure_ascii=False)
    
    async def fetch_user_tweets(
        self, 
        user_id: str, 
        count: int = 20,
        use_cache: bool = True
    ) -> List[str]:
        """Fetch recent tweets for a user, returning just the text content."""
        if not self._logged_in:
            raise RuntimeError("Must login first")
        
        # Check cache first
        if use_cache and user_id in self._tweets_cache:
            return self._tweets_cache[user_id]
        
        tweets_text = []
        try:
            # Fetch user's tweets
            tweets = await self.client.get_user_tweets(user_id, 'Tweets', count=count)
            
            for tweet in tweets:
                text = getattr(tweet, 'text', '') or ''
                if text:
                    tweets_text.append(text)
            
            # Cache the results
            self._tweets_cache[user_id] = tweets_text
            
        except Exception as e:
            # Silently fail for individual users - they may have protected accounts
            self._tweets_cache[user_id] = []
        
        return tweets_text
    
    async def fetch_tweets_for_mutuals(
        self, 
        mutuals: List[Dict],
        tweets_per_user: int = 20,
        use_cache: bool = True,
        rate_limit_delay: float = 0.5
    ) -> Dict[str, List[str]]:
        """Fetch tweets for all mutuals with rate limiting and progress tracking."""
        if not self._logged_in:
            raise RuntimeError("Must login first")
        
        print(f"Fetching tweets for {len(mutuals)} users...")
        
        tweets_by_user = {}
        fetched_count = 0
        cached_count = 0
        
        for i, mutual in enumerate(mutuals):
            user_id = mutual['user_id']
            screen_name = mutual.get('screen_name', user_id)
            
            # Check if already cached
            if use_cache and user_id in self._tweets_cache:
                tweets_by_user[user_id] = self._tweets_cache[user_id]
                cached_count += 1
                continue
            
            # Fetch tweets
            tweets = await self.fetch_user_tweets(
                user_id, 
                count=tweets_per_user,
                use_cache=use_cache
            )
            tweets_by_user[user_id] = tweets
            fetched_count += 1
            
            # Progress update every 20 users
            if (fetched_count) % 20 == 0:
                print(f"  Fetched tweets for {fetched_count} users ({cached_count} cached)...")
            
            # Rate limiting to avoid getting blocked
            if rate_limit_delay > 0:
                await asyncio.sleep(rate_limit_delay)
        
        # Save cache after all fetches
        self._save_tweets_cache()
        
        print(f"✓ Fetched tweets for {fetched_count} users ({cached_count} from cache)")
        return tweets_by_user
    
    def _convert_cookies_format(self, cookies_data):
        """Convert Cookie-Editor format (list of objects) to Twikit format (dict)."""
        if isinstance(cookies_data, dict):
            # Already in dict format - might be name:value pairs or already converted
            # Check if it's a simple name:value dict
            if all(isinstance(v, str) for v in cookies_data.values()):
                return cookies_data
            # Otherwise assume it needs conversion
            return {item.get('name'): item.get('value') for item in cookies_data.values() if 'name' in item and 'value' in item}
        elif isinstance(cookies_data, list):
            # Cookie-Editor format: list of cookie objects
            return {cookie.get('name'): cookie.get('value') for cookie in cookies_data if 'name' in cookie and 'value' in cookie}
        else:
            raise ValueError(f"Unexpected cookie format: {type(cookies_data)}")
    
    async def login(self, username: str, email: str, password: str, retries: int = 3) -> None:
        """Login to Twitter and save session cookies."""
        if self.cookies_file.exists():
            try:
                # Load and convert cookies format
                with open(self.cookies_file, 'r', encoding='utf-8') as f:
                    cookies_data = json.load(f)
                
                # Convert to Twikit format (dict of name:value)
                cookies_dict = self._convert_cookies_format(cookies_data)
                
                # Set cookies using Twikit's method
                if hasattr(self.client, 'set_cookies'):
                    self.client.set_cookies(cookies_dict)
                    print(f"✓ Loaded {len(cookies_dict)} cookies from cache")
                
                # Verify session is still valid
                try:
                    await self.client.get_user_by_screen_name(username)
                    self._logged_in = True
                    print("✓ Using cached session")
                    return
                except Exception as e:
                    print(f"Cached session expired or invalid: {e}")
                    print("Logging in again...")
            except Exception as e:
                print(f"Failed to load cookies: {e}")
                print("Logging in again...")
        
        # Try login with retries and delays
        last_error = None
        for attempt in range(1, retries + 1):
            try:
                if attempt > 1:
                    delay = 2 ** attempt  # Exponential backoff: 4s, 8s
                    print(f"Retrying login (attempt {attempt}/{retries}) after {delay}s delay...")
                    await asyncio.sleep(delay)
                
                # Login returns a dict, but we don't need to use it
                result = await self.client.login(
                    auth_info_1=username,
                    auth_info_2=email,
                    password=password,
                    cookies_file=str(self.cookies_file)
                )
                
                # Save cookies after successful login
                if hasattr(self.client, 'save_cookies'):
                    self.client.save_cookies(str(self.cookies_file))
                
                self._logged_in = True
                print("✓ Logged in successfully")
                return
                
            except Exception as e:
                last_error = e
                error_str = str(e)
                
                # Check if it's a Cloudflare block
                if "403" in error_str or "cloudflare" in error_str.lower() or "blocked" in error_str.lower():
                    if attempt < retries:
                        print(f"⚠ Cloudflare protection detected (attempt {attempt}/{retries})")
                        continue
                    else:
                        print("\n" + "=" * 60)
                        print("❌ Cloudflare Protection Blocked Login")
                        print("=" * 60)
                        print("\nTwitter/X is blocking automated login attempts.")
                        print("\nSOLUTIONS:")
                        print("1. Wait 10-15 minutes and try again")
                        print("2. Use a different network/VPN")
                        print("3. Manually export cookies from your browser:")
                        print("   - Log into Twitter/X in your browser")
                        print("   - Use a browser extension to export cookies")
                        print("   - Save as JSON to:", self.cookies_file)
                        print("   - Then run the script again (it will use cached cookies)")
                        print("\n4. Try using the Twitter data export instead:")
                        print("   - Settings > Your Account > Download an archive")
                        print("   - Wait 24 hours for the export")
                        print("   - Use the following.js and follower.js files")
                        print("=" * 60)
                        raise RuntimeError("Cloudflare blocked login. See suggestions above.") from e
                
                # Other errors
                if attempt < retries:
                    print(f"⚠ Login failed (attempt {attempt}/{retries}): {error_str[:100]}")
                else:
                    raise
        
        # If we get here, all retries failed
        raise RuntimeError(f"Login failed after {retries} attempts") from last_error
    
    async def get_user_id(self, username: str) -> str:
        """Get user ID from username."""
        if not self._logged_in:
            raise RuntimeError("Must login first")
        user = await self.client.get_user_by_screen_name(username)
        return user.id
    
    async def fetch_all_followers(self, username: str) -> List[Dict]:
        """Fetch all followers for a user with pagination."""
        if not self._logged_in:
            raise RuntimeError("Must login first")
        
        print(f"Fetching followers for @{username}...")
        followers = []
        
        try:
            # Get user object first to get user_id
            user = await self.client.get_user_by_screen_name(username)
            user_id = user.id
            
            # Fetch followers with pagination
            result = await self.client.get_user_followers(user_id, count=200)
            
            # Process first page
            for user_obj in result:
                followers.append(self._extract_user_data(user_obj))
            
            # Handle pagination - fetch remaining pages
            page_count = 1
            while result.next_cursor:
                page_count += 1
                if page_count % 10 == 0:
                    print(f"  Fetched {len(followers)} followers so far...")
                
                result = await result.next()
                if len(result) == 0:
                    break
                
                for user_obj in result:
                    followers.append(self._extract_user_data(user_obj))
            
            print(f"✓ Fetched {len(followers)} followers")
            return followers
            
        except Exception as e:
            print(f"Error fetching followers: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    async def fetch_all_following(self, username: str) -> List[Dict]:
        """Fetch all following for a user with pagination."""
        if not self._logged_in:
            raise RuntimeError("Must login first")
        
        print(f"Fetching following for @{username}...")
        following = []
        
        try:
            # Get user object first to get user_id
            user = await self.client.get_user_by_screen_name(username)
            user_id = user.id
            
            # Fetch following with pagination
            result = await self.client.get_user_following(user_id, count=200)
            
            # Process first page
            for user_obj in result:
                following.append(self._extract_user_data(user_obj))
            
            # Handle pagination - fetch remaining pages
            page_count = 1
            while result.next_cursor:
                page_count += 1
                if page_count % 10 == 0:
                    print(f"  Fetched {len(following)} following so far...")
                
                result = await result.next()
                if len(result) == 0:
                    break
                
                for user_obj in result:
                    following.append(self._extract_user_data(user_obj))
            
            print(f"✓ Fetched {len(following)} following")
            return following
            
        except Exception as e:
            print(f"Error fetching following: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def find_mutuals(self, followers: List[Dict], following: List[Dict]) -> List[Dict]:
        """Find mutual connections between followers and following."""
        followers_by_id = {f['user_id']: f for f in followers}
        following_by_id = {f['user_id']: f for f in following}
        
        mutual_ids = set(followers_by_id.keys()) & set(following_by_id.keys())
        
        # Use data from followers list (prefer that source)
        mutuals = [followers_by_id[uid] for uid in mutual_ids]
        
        print(f"✓ Found {len(mutuals)} mutual connections")
        return mutuals
    
    def save_mutuals(self, mutuals: List[Dict]) -> None:
        """Save mutuals data to cache file."""
        with open(self.mutuals_file, 'w', encoding='utf-8') as f:
            json.dump(mutuals, f, indent=2, ensure_ascii=False)
        print(f"✓ Cached {len(mutuals)} mutuals to {self.mutuals_file}")
    
    def load_mutuals(self) -> Optional[List[Dict]]:
        """Load mutuals data from cache file."""
        if not self.mutuals_file.exists():
            return None
        
        try:
            with open(self.mutuals_file, 'r', encoding='utf-8') as f:
                mutuals = json.load(f)
            print(f"✓ Loaded {len(mutuals)} mutuals from cache")
            return mutuals
        except Exception as e:
            print(f"Error loading cache: {e}")
            return None
    
    async def collect_mutuals(
        self, 
        username: str, 
        use_cache: bool = True
    ) -> List[Dict]:
        """Main method to collect mutual connections."""
        # Try loading from cache first
        if use_cache:
            cached = self.load_mutuals()
            if cached is not None:
                return cached
        
        # Fetch fresh data
        followers = await self.fetch_all_followers(username)
        following = await self.fetch_all_following(username)
        
        # Find mutuals
        mutuals = self.find_mutuals(followers, following)
        
        # Cache results
        self.save_mutuals(mutuals)
        
        return mutuals
    
    def load_mutual_edges(self) -> Optional[List[Tuple[str, str]]]:
        """Load mutual edges from cache file."""
        if not self.mutual_edges_file.exists():
            return None
        
        try:
            with open(self.mutual_edges_file, 'r', encoding='utf-8') as f:
                edges_data = json.load(f)
            # Convert list of lists to list of tuples
            edges = [(edge[0], edge[1]) for edge in edges_data]
            print(f"✓ Loaded {len(edges)} mutual edges from cache")
            return edges
        except Exception as e:
            print(f"Error loading mutual edges cache: {e}")
            return None
    
    def save_mutual_edges(self, edges: List[Tuple[str, str]]) -> None:
        """Save mutual edges to cache file."""
        # Convert tuples to lists for JSON serialization
        edges_data = [[source, target] for source, target in edges]
        with open(self.mutual_edges_file, 'w', encoding='utf-8') as f:
            json.dump(edges_data, f, indent=2, ensure_ascii=False)
        print(f"✓ Cached {len(edges)} mutual edges to {self.mutual_edges_file}")
    
    def _is_rate_limit_error(self, error: Exception) -> bool:
        """Check if error is a rate limit (429) error."""
        error_str = str(error).lower()
        error_repr = repr(error).lower()
        return (
            "429" in error_str or 
            "429" in error_repr or
            "rate limit" in error_str or
            "ratelimit" in error_str or
            "too many requests" in error_str
        )
    
    def _save_progress(self, edges: List[Tuple[str, str]], checked_indices: List[int]) -> None:
        """Save incremental progress to avoid losing work on rate limits."""
        # Remove duplicates before saving
        unique_edges = set()
        for source, target in edges:
            if source < target:
                unique_edges.add((source, target))
            else:
                unique_edges.add((target, source))
        
        edges_list = list(unique_edges)
        
        # Save edges
        self.save_mutual_edges(edges_list)
        
        # Save progress checkpoint
        progress_file = self.cache_dir / "mutual_edges_progress.json"
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump({
                'checked_indices': checked_indices,
                'total_edges': len(edges_list)
            }, f, indent=2)
    
    def _load_progress(self) -> Optional[Dict]:
        """Load progress checkpoint to resume from where we left off.
        
        Returns:
            Dict with 'checked_indices' and optionally 'edges' if available, or None
        """
        progress_file = self.cache_dir / "mutual_edges_progress.json"
        if not progress_file.exists():
            return None
        
        try:
            with open(progress_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except Exception:
            return None
    
    def _has_incomplete_progress(self) -> bool:
        """Check if there's an incomplete progress checkpoint."""
        progress_file = self.cache_dir / "mutual_edges_progress.json"
        return progress_file.exists()
    
    async def fetch_mutual_connections(
        self, 
        mutuals: List[Dict],
        ego_user_id: Optional[str] = None,
        use_cache: bool = True,
        rate_limit_delay: float = 2.0,
        sample_size: Optional[int] = None,
        max_retries: int = 3,
        use_adaptive: bool = True
    ) -> List[Tuple[str, str]]:
        """Find which mutuals follow each other.
        
        Args:
            mutuals: List of mutual user dictionaries
            ego_user_id: User ID of the ego account (will be excluded from edges)
            use_cache: Whether to use cached results if available
            rate_limit_delay: Base delay between API calls in seconds (default: 2.0)
            sample_size: Optional limit on number of mutuals to check (None = check all)
            max_retries: Maximum retries for rate limit errors (default: 3)
            use_adaptive: Whether to use adaptive rate limiting (default: True)
        
        Returns:
            List of (source_user_id, target_user_id) tuples representing edges
        """
        if not self._logged_in:
            raise RuntimeError("Must login first")
        
        # Try loading from cache first
        # But if there's a progress checkpoint, we should resume instead
        has_progress = self._has_incomplete_progress() if use_cache else False
        
        if use_cache and not has_progress:
            cached_edges = self.load_mutual_edges()
            if cached_edges is not None:
                return cached_edges
        
        # Create set of mutual user IDs for fast lookup
        mutual_ids = {mutual['user_id'] for mutual in mutuals}
        
        # Exclude ego account from mutual_ids to prevent recursion and redundant edges
        if ego_user_id:
            mutual_ids.discard(ego_user_id)
        
        # Optionally sample mutuals (e.g., top N by followers)
        mutuals_to_check = mutuals
        if sample_size and sample_size < len(mutuals):
            # Sort by followers_count descending and take top N
            mutuals_to_check = sorted(
                mutuals, 
                key=lambda m: m.get('followers_count', 0), 
                reverse=True
            )[:sample_size]
            print(f"Sampling top {sample_size} mutuals by follower count")
        
        # Load progress checkpoint and existing edges if available
        checked_indices = set()
        edges = []
        
        if use_cache:
            progress_data = self._load_progress()
            if progress_data:
                checked_indices = set(progress_data.get('checked_indices', []))
                print(f"Resuming from checkpoint: {len(checked_indices)} users already checked")
                
                # Load existing edges from cache to merge with new ones
                cached_edges = self.load_mutual_edges()
                if cached_edges:
                    edges = list(cached_edges)
                    print(f"  Loaded {len(edges)} existing edges from cache")
        
        # Initialize adaptive rate limiter if enabled
        if use_adaptive:
            adaptive_limiter = AdaptiveRateLimiter(base_delay=rate_limit_delay)
            print(f"Checking mutual connections among {len(mutuals_to_check)} users...")
            print(f"  Adaptive rate limiting enabled (starting at {rate_limit_delay}s)")
            print(f"  (This may take a while due to rate limiting)")
        else:
            adaptive_limiter = None
            print(f"Checking mutual connections among {len(mutuals_to_check)} users...")
            print(f"  Base delay: {rate_limit_delay}s between requests")
            print(f"  (This may take a while due to rate limiting)")
        
        checked_count = len(checked_indices)
        error_count = 0
        rate_limit_count = 0
        edges_list = []  # Initialize for use in finally block
        
        # Use try/finally to ensure we save progress even on interruption
        try:
            for i, mutual in enumerate(mutuals_to_check):
                # Skip if already checked
                if i in checked_indices:
                    continue
                
                source_id = mutual['user_id']
                source_name = mutual.get('screen_name', source_id)
                
                retry_count = 0
                success = False
                
                while retry_count <= max_retries and not success:
                    try:
                        # Fetch this mutual's following list
                        result = await self.client.get_user_following(source_id, count=200)
                        
                        # Process first page
                        following_ids = set()
                        for user_obj in result:
                            following_ids.add(str(user_obj.id))
                        
                        # Handle pagination with max page limit to prevent infinite recursion
                        page_count = 0
                        max_pages = 50  # 50 pages * 200 = 10,000 following max
                        while result.next_cursor and page_count < max_pages:
                            page_count += 1
                            result = await result.next()
                            if len(result) == 0:
                                break
                            for user_obj in result:
                                following_ids.add(str(user_obj.id))
                        
                        # Find overlaps with other mutuals (excluding ego account)
                        for target_id in following_ids:
                            if (target_id in mutual_ids and 
                                target_id != source_id and 
                                target_id != ego_user_id):
                                edges.append((source_id, target_id))
                        
                        checked_count += 1
                        checked_indices.add(i)
                        success = True
                        
                        # Update adaptive limiter on success
                        if adaptive_limiter:
                            adaptive_limiter.on_success()
                        
                        # Save progress every 10 users
                        if checked_count % 10 == 0:
                            self._save_progress(edges, list(checked_indices))
                        
                        # Progress update every 20 users
                        if checked_count % 20 == 0:
                            current_delay = adaptive_limiter.current_delay if adaptive_limiter else rate_limit_delay
                            print(f"  Checked {checked_count}/{len(mutuals_to_check)} users, found {len(edges)} connections so far... (delay: {current_delay:.1f}s)")
                        
                        # Reset rate limit counter on success
                        rate_limit_count = 0
                        
                        # Rate limiting - use adaptive delay if enabled, otherwise fixed delay
                        if adaptive_limiter:
                            if adaptive_limiter.current_delay > 0:
                                await asyncio.sleep(adaptive_limiter.current_delay)
                        elif rate_limit_delay > 0:
                            await asyncio.sleep(rate_limit_delay)
                            
                    except RecursionError:
                        # Some users trigger infinite recursion in Twikit pagination
                        error_count += 1
                        if error_count <= 5:
                            print(f"  Warning: Pagination recursion error for @{source_name}, skipping")
                        # Save progress even on errors to avoid losing work
                        if checked_count % 5 == 0:
                            self._save_progress(edges, list(checked_indices))
                        success = True  # Skip this user
                        continue
                    except Exception as e:
                        if self._is_rate_limit_error(e):
                            rate_limit_count += 1
                            retry_count += 1
                            
                            # Update adaptive limiter on rate limit
                            if adaptive_limiter:
                                adaptive_limiter.on_rate_limit()
                            
                            if retry_count <= max_retries:
                                # Exponential backoff: 60s, 120s, 240s
                                wait_time = 60 * (2 ** (retry_count - 1))
                                current_delay = adaptive_limiter.current_delay if adaptive_limiter else rate_limit_delay
                                print(f"\n⚠ Rate limit hit (attempt {retry_count}/{max_retries})")
                                print(f"  Current delay adjusted to {current_delay:.1f}s")
                                print(f"  Waiting {wait_time}s before retrying @{source_name}...")
                                await asyncio.sleep(wait_time)
                            else:
                                print(f"\n❌ Rate limit exceeded after {max_retries} retries")
                                print(f"  Saving progress and stopping...")
                                print(f"  You can resume later - progress is saved to cache")
                                self._save_progress(edges, list(checked_indices))
                                raise RuntimeError(
                                    f"Rate limit exceeded. Progress saved. "
                                    f"Checked {checked_count}/{len(mutuals_to_check)} users. "
                                    f"Run again to resume from checkpoint."
                                ) from e
                        else:
                            # Non-rate-limit error - log, save progress, and continue
                            error_count += 1
                            if error_count <= 5:  # Only print first few errors
                                print(f"  Warning: Could not fetch following for @{source_name}: {e}")
                            
                            # Save progress even on errors to avoid losing work
                            if checked_count % 5 == 0:  # Save more frequently on errors
                                self._save_progress(edges, list(checked_indices))
                            
                            success = True  # Mark as "handled" to break retry loop
                            continue
        
        except KeyboardInterrupt:
            print(f"\n\n⚠ Interrupted by user")
            print(f"  Saving progress...")
            self._save_progress(edges, list(checked_indices))
            print(f"  Progress saved. Run again to resume.")
            raise
        
        except Exception as e:
            # Save progress on any unexpected error
            print(f"\n⚠ Unexpected error: {e}")
            print(f"  Saving progress...")
            self._save_progress(edges, list(checked_indices))
            raise
        
        finally:
            # Always save final state
            # Remove duplicates (since A->B and B->A might both exist, but we want undirected edges)
            # Keep only one direction (smaller ID first)
            unique_edges = set()
            for source, target in edges:
                if source < target:
                    unique_edges.add((source, target))
                else:
                    unique_edges.add((target, source))
            
            edges_list = list(unique_edges)
            
            # Final save - always happens
            self.save_mutual_edges(edges_list)
        
        # Clean up progress file on successful completion (only if we got here)
        progress_file = self.cache_dir / "mutual_edges_progress.json"
        if progress_file.exists():
            progress_file.unlink()
        
        print(f"✓ Found {len(edges_list)} unique mutual connections")
        if error_count > 0:
            print(f"  ({error_count} users skipped due to errors)")
        
        return edges_list

