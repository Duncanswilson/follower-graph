# Twitter Ego Network Visualization

Build an interactive ego network visualization of your Twitter mutuals with automatic clustering based on bio content.

## Features

- Fetches your Twitter followers and following using Twikit
- Identifies mutual connections
- Clusters mutuals by bio keywords using TF-IDF and K-Means
- Generates interactive HTML visualization with PyVis
- Color-coded clusters with hover tooltips

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
# First run - will prompt for Twitter credentials
python main.py --username your_handle

# Subsequent runs use cached session
python main.py --username your_handle --clusters 8

# Auto-detect optimal number of clusters
python main.py --username your_handle --auto-clusters
```

## Output

The visualization is saved to `output/ego_network.html`. Open it in your browser to explore your mutual network!

## Troubleshooting

### Cloudflare 403 Error

If you see a "403 Forbidden" or "Cloudflare blocked" error:

1. **Wait and Retry**: Cloudflare blocks may be temporary. Wait 10-15 minutes and try again.

2. **Use Browser Cookies** (Recommended):
   - Log into Twitter/X in your browser
   - Install a cookie export extension (e.g., "Cookie-Editor" for Chrome/Firefox)
   - Export cookies for `x.com` as JSON
   - Save to `cache/cookies.json` in the project directory
   - Run the script again - it will use the cached cookies

3. **Try Different Network**: Use a different IP/VPN to avoid IP-based blocks

4. **Use Twitter Data Export**:
   - Go to Twitter Settings > Your Account > Download an archive
   - Wait 24 hours for the export
   - Extract `following.js` and `follower.js` from the archive
   - Place them in a `data/` directory
   - (Note: This requires modifying the code to use the export parser instead)

### Other Issues

- **2FA Enabled**: Twikit may not work with 2FA. Temporarily disable it or use browser cookies.
- **Rate Limits**: If you hit rate limits, wait before retrying. The script includes automatic retries.

## Security

- Credentials are only used for initial login
- Session cookies are cached locally (not committed to git)
- Never share your `cache/` directory

