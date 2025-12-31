"""Twitter data collector using Twikit library."""

import asyncio
import json
from pathlib import Path
from typing import List, Dict, Optional
from twikit import Client


class TwitterCollector:
    """Collects Twitter followers, following, and profile data using Twikit."""
    
    def __init__(self, cache_dir: str = "cache"):
        """Initialize collector with cache directory."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cookies_file = self.cache_dir / "cookies.json"
        self.mutuals_file = self.cache_dir / "mutuals.json"
        self.client = Client('en-US')
        self._logged_in = False
    
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

