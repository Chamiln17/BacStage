# Troubleshooting Guide

## Common Errors and Solutions

### Error 1: Permission Denied When Writing CSV

**Error Message:**
```
PermissionError: [Errno 13] Permission denied: 'data\\raw\\videos_metadata.csv'
```

**Cause:** The CSV file is open in another program (Excel, VS Code, etc.)

**Solution:**
1. **Close the file** in your IDE/editor
2. **Close Excel** if you have the file open
3. **Run the command again**

**Prevention:**
- Don't keep data files open while running collection scripts
- Use `--output` flag to write to a different file for testing

---

### Error 2: UnicodeEncodeError (Arabic Characters)

**Error Message:**
```
UnicodeEncodeError: 'charmap' codec can't encode character '✓' in position...
UnicodeEncodeError: 'charmap' codec can't encode characters 'الاستاذ'...
```

**Cause:** Windows console doesn't support UTF-8 by default

**Solution:** ✅ **Already Fixed!**
- All collection scripts now handle UTF-8 encoding automatically
- Logs are written with UTF-8 encoding
- Arabic characters and special symbols (✓) now work correctly

**If still occurring:**
Run PowerShell with UTF-8:
```powershell
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"
```

---

### Error 3: Duplicate Channel IDs

**Error Message:**
```
Retrieved 272 video IDs from channel...
Retrieved 225 video IDs from channel...
# But both are from the same channel ID
```

**Cause:** Same channel ID used for different channels in `channels.csv`

**Example of WRONG channels.csv:**
```csv
channel_id,channel_name,subjects
UCuJqXbrblfKeO9fa82a7KHQ,Channel 1,Math
UCuJqXbrblfKeO9fa82a7KHQ,Channel 2,Physics  ← DUPLICATE ID!
```

**Solution:**
1. Find the correct channel ID for each channel
2. Each row must have a **unique** channel ID

**How to find channel ID:** Open browser console (F12) on the channel page:
```javascript
document.querySelector('link[rel="canonical"]').href.split('/channel/')[1].split('/')[0]
```

---

### Error 4: Logs Scattered Everywhere

**Problem:** Log files created in project root

**Solution:** ✅ **Already Fixed!**
- All logs now go to `logs/` directory
- Log files:
  - `logs/data_collection.log`
  - `logs/data_collection_enhanced.log`
  - `logs/data_collection_incremental.log`
  - `logs/feature_engineering.log`

**View logs:**
```bash
# Latest 50 lines
tail -50 logs/data_collection.log

# Or PowerShell
Get-Content logs/data_collection.log -Tail 50

# All logs
Get-ChildItem logs/
```

---

### Error 5: API Quota Exceeded

**Error Message:**
```
HttpError 403: quotaExceeded
```

**Cause:** Used all 10,000 daily API quota units

**Solution:**
1. **Wait until next day** (quota resets midnight PST)
2. **Check usage:**
   ```bash
   grep "quota" logs/data_collection.log
   ```
3. **Limit future collections:**
   ```bash
   uv run python -m src.data.collect \
       --channels data/raw/channels.csv \
       --max-videos 50 \
       --max-quota 4000
   ```

---

### Error 6: Channel Not Found

**Error Message:**
```
⚠️ No videos found for this channel
```

**Possible Causes:**
1. Channel ID is incorrect
2. Channel is private/deleted
3. Channel has no public videos

**Solution:**
1. **Verify channel exists:**
   - Visit: `https://www.youtube.com/channel/YOUR_CHANNEL_ID`
2. **Check if channel is public**
3. **Get the correct channel ID** using browser console

---

### Error 7: Missing Videos (Got 125, Expected 300)

**Not an Error!** This is expected behavior.

**Explanation:**
- YouTube Search API returns max ~500 most recent videos
- Private/unlisted videos excluded
- Very old videos may not be indexed
- This is a YouTube API limitation, not a bug

**If you really need more:**
- The API already returned all available videos
- 125 videos is what's publicly accessible via the API

---

## Quick Fixes Checklist

Before running collection:
- ✅ Close all CSV files in IDE/Excel
- ✅ Check channel IDs are unique (no duplicates)
- ✅ Verify API key is in `.env` file
- ✅ Check quota usage if running multiple times

After errors:
- ✅ Check `logs/` directory for error details
- ✅ Verify channels.csv format is correct
- ✅ Close any open data files
- ✅ Try with `--max-videos 10` for testing

---

## Getting Help

1. **Check logs first:**
   ```bash
   cat logs/data_collection_incremental.log
   ```

2. **Test with small sample:**
   ```bash
   uv run python -m src.data.collect \
       --channels data/raw/channels.csv \
       --max-videos 5
   ```

3. **Verify your setup:**
   ```bash
   # Check API key
   cat .env

   # Check channels
   cat data/raw/channels.csv

   # Test imports
   uv run python -c "from src.data import YouTubeCollector; print('OK')"
   ```

---

## Prevention Tips

**Before Collection:**
1. Test with 1-2 channels first
2. Use `--max-videos 10` for initial testing
3. Check quota usage in logs after each run
4. Keep data files closed

**Best Practices:**
1. Always use incremental collection for adding channels
2. Monitor logs directory for errors
3. Backup important data before re-collecting
4. Use meaningful channel names in CSV for tracking

---

## Windows-Specific Issues

### PowerShell Encoding
If you see encoding errors, set UTF-8 at the start of your session:
```powershell
chcp 65001
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"
```

### File Locks
Windows locks files more aggressively than Linux:
- Close files before running scripts
- Use Task Manager to check if Excel/IDE has file open
- Restart terminal if file stays locked

---

## Still Having Issues?

1. Check this guide first
2. Review logs in `logs/` directory  
3. Try the "Quick Fixes Checklist" above
4. Verify your channels.csv has unique IDs
5. Test with a minimal example (1 channel, 5 videos)
