#!/usr/bin/env python3
import os, sys, urllib.request
from pathlib import Path

def download_with_retry(url, filepath, expected_size=None, max_retries=3):
    for attempt in range(max_retries):
        try:
            print(f"Download attempt {attempt+1}/{max_retries}")
            def report_progress(block_num, block_size, total_size):
                downloaded = block_num*block_size
                pct = min(downloaded/total_size*100,100)
                mb_d = downloaded/1024/1024; mb_t = total_size/1024/1024
                sys.stdout.write(f'\rProgress: {pct:.1f}% ({mb_d:.1f}/{mb_t:.1f} MB)')
                sys.stdout.flush()
            urllib.request.urlretrieve(url, filepath, reporthook=report_progress)
            print("\n✅ Download complete!")
            size = os.path.getsize(filepath)
            print(f"File size: {size/1024/1024:.1f} MB")
            if expected_size and size < expected_size*0.9:
                print("⚠️ Too small, retrying...")
                os.remove(filepath); continue
            with open(filepath,'rb') as f:
                if f.read(4) != b'GGUF':
                    print("⚠️ Bad header, retrying…")
                    os.remove(filepath); continue
                else:
                    print("✅ Header OK")
            return True
        except Exception as e:
            print(f"\nError: {e}")
            if os.path.exists(filepath): os.remove(filepath)
    return False

def main():
    Path("models").mkdir(exist_ok=True)
    url = "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    path = "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    exp_size = 669_000_000
    print("Starting download…")
    if os.path.exists(path):
        sz=os.path.getsize(path)
        hdr=open(path,'rb').read(4)
        if hdr==b'GGUF' and sz>exp_size*0.9:
            print("✅ Existing model looks valid"); return
        else:
            print("❌ Corrupted, re-downloading"); os.remove(path)
    ok = download_with_retry(url, path, exp_size)
    if ok: print("✅ Downloaded and verified!")
    else: sys.exit("❌ Failed to download model.")

if __name__=="__main__":
    main()
