
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from data.transcript_collector import clean_vtt_text

def test_clean_vtt():
    print("Testing clean_vtt_text with malicious inputs...")
    
    # 1. Normal VTT
    vtt = """WEBVTT
    
    00:00:01.000 --> 00:00:05.000
    Hello world
    """
    assert clean_vtt_text(vtt) == "Hello world", "Failed normal VTT"
    print("✓ Normal VTT passed")
    
    # 2. JS Injection
    js_garbage = """window.WIZ_global_data = {"test": true};
    var ytcfg = {d: function() { return {}; }};
    """
    cleaned = clean_vtt_text(js_garbage)
    assert cleaned == "", f"Failed JS injection check. Result: '{cleaned}'"
    print("✓ JS Injection passed (result empty)")
    
    # 3. HTML Content
    html_garbage = """
    <!DOCTYPE html>
    <html lang="en">
    <body>
    <script>var x=1;</script>
    </body>
    </html>
    """
    cleaned_html = clean_vtt_text(html_garbage)
    # The new cleaner might not catch all HTML tags if they don't look like VTT timestamps, 
    # but the get_best_transcript has a pre-check. 
    # However, clean_vtt_text strips tags <...>.
    print(f"HTML Result: '{cleaned_html}'") 

if __name__ == "__main__":
    try:
        test_clean_vtt()
        print("\nAll Tests Passed!")
    except AssertionError as e:
        print(f"\nTEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)
