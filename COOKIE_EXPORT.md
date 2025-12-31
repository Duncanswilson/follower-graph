# Manual Cookie Export Guide

If automated login is blocked by Cloudflare, you can manually export cookies from your browser.

## Method 1: Using Cookie-Editor Extension (Easiest)

1. **Install Cookie-Editor**:
   - Chrome: https://chrome.google.com/webstore/detail/cookie-editor/hlkenndednhfkekhgcdicdfddnkalmdm
   - Firefox: https://addons.mozilla.org/en-US/firefox/addon/cookie-editor/

2. **Export Cookies**:
   - Log into Twitter/X in your browser (x.com)
   - Click the Cookie-Editor extension icon
   - Click "Export" button
   - Select "JSON" format
   - Copy the JSON content

3. **Save Cookies**:
   - Create `cache/` directory if it doesn't exist
   - Save the JSON to `cache/cookies.json`
   - Make sure the file is valid JSON

4. **Run Script**:
   ```bash
   python main.py --username your_handle
   ```
   The script will detect and use the cached cookies automatically.

## Method 2: Using Browser DevTools

1. **Open DevTools**:
   - Log into Twitter/X (x.com)
   - Press F12 or right-click > Inspect
   - Go to "Application" tab (Chrome) or "Storage" tab (Firefox)

2. **Export Cookies**:
   - In the left sidebar, click "Cookies" > "https://x.com"
   - Right-click and select "Copy" or manually copy cookie values
   - You'll need to format them as JSON

3. **Format as JSON**:
   Create a JSON file with this structure:
   ```json
   [
     {
       "name": "cookie_name",
       "value": "cookie_value",
       "domain": ".x.com",
       "path": "/",
       "secure": true,
       "httpOnly": false
     }
   ]
   ```

4. **Save to** `cache/cookies.json`

## Method 3: Using Python Script

If you have Python installed, you can use this helper script:

```python
import json
from http.cookies import SimpleCookie

# Paste your cookie string from browser DevTools (Network tab > Request Headers > Cookie)
cookie_string = "paste_cookie_string_here"

# Parse cookies
cookie = SimpleCookie()
cookie.load(cookie_string)

# Convert to JSON format
cookies_list = []
for key, morsel in cookie.items():
    cookies_list.append({
        "name": key,
        "value": morsel.value,
        "domain": ".x.com",
        "path": "/",
        "secure": True,
        "httpOnly": False
    })

# Save to file
with open("cache/cookies.json", "w") as f:
    json.dump(cookies_list, f, indent=2)

print("Cookies saved to cache/cookies.json")
```

## Verification

After saving cookies, verify they work:
```bash
python -c "import json; json.load(open('cache/cookies.json')); print('✓ Cookies file is valid JSON')"
```

## Notes

- Cookies expire after some time. You may need to re-export them periodically.
- Make sure you're logged into Twitter/X when exporting cookies.
- The cookies file should contain cookies for `x.com` domain.

