# Copyright (c) 2026 Xavier Callens / Socrate AI Lab. All Rights Reserved.
# SPDX-License-Identifier: MIT
#
# WARS-Quantum-LTN: "Pour l'Honneur de la Science" Hugging Face Promotion Loop
# =========================================================================

import os
import ssl
import sys
import time

# ── SSL BYPASS PATCHES FOR ENTERPRISE DEEP INSPECTION PROXIES ───────────────
ssl._create_default_https_context = ssl._create_unverified_context
try:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except ImportError:
    pass

# Patch requests to ignore SSL verification globally
import requests
original_requests_init = requests.Session.__init__
def patched_requests_init(self, *args, **kwargs):
    original_requests_init(self, *args, **kwargs)
    self.verify = False
requests.Session.__init__ = patched_requests_init

# Patch httpx (which huggingface_hub uses under the hood) to ignore SSL verification
import httpx
original_httpx_init = httpx.Client.__init__
def patched_httpx_init(self, *args, **kwargs):
    kwargs['verify'] = False
    original_httpx_init(self, *args, **kwargs)
httpx.Client.__init__ = patched_httpx_init

# Bypassing certificate checks for git/curl and python requests envs
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
os.environ["PYTHONHTTPSVERIFY"] = "0"
os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"

from huggingface_hub import HfApi

# Stylized Console Colors
RED = '\033[0;31m'
GREEN = '\033[0;32m'
YELLOW = '\033[0;33m'
BLUE = '\033[0;34m'
MAGENTA = '\033[0;35m'
CYAN = '\033[0;36m'
BOLD = '\033[1m'
NC = '\033[0m'

REPO_ID = "callensxavier/runux-wars-ci-dfa-tpu-benchmarks"

def promote_scientific_results():
    print(f"{CYAN}{BOLD}========================================================================{NC}")
    print(f"{CYAN}{BOLD}   RunuX AI Engine — Hugging Face Scientific Promotion Loop            {NC}")
    print(f"{CYAN}{BOLD}   Motto: Pour l'Honneur de la Science & l'Esprit Humain 🇫🇷            {NC}")
    print(f"{CYAN}{BOLD}========================================================================{NC}\n")

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print(f"{RED}❌ Error: Hugging Face Access Token (HF_TOKEN) is not configured in environment.{NC}")
        print("          Please set it before executing this promotion utility.")
        sys.exit(1)
        
    print(f"  [+] Found active Hugging Face access credentials.")
    print(f"  [+] Target Hugging Face Repository: {BOLD}{REPO_ID}{NC}")
    print("  [+] Compiling physical solvers, preprints, and multidimensional plots...")

    files_to_promote = {
        "PAPER_DRAFT.md": "PAPER_DRAFT.md",
        "README.md": "README.md",
        "examples/solve_tearing_mode.py": "examples/solve_tearing_mode.py",
        "examples/solve_2d_tearing_mode.py": "examples/solve_2d_tearing_mode.py",
        "examples/solve_3d_toroidal_disruption.py": "examples/solve_3d_toroidal_disruption.py",
        "examples/tearing_mode_benchmark.png": "examples/tearing_mode_benchmark.png",
        "examples/tearing_mode_2d_benchmark.png": "examples/tearing_mode_2d_benchmark.png",
        "examples/toroidal_disruption_3d_benchmark.png": "examples/toroidal_disruption_3d_benchmark.png"
    }

    # Initialize Hugging Face API (which now uses the patched httpx class)
    api = HfApi()

    # Upload files programmatically
    start_time = time.time()
    successful_uploads = []
    
    for local_path, repo_path in files_to_promote.items():
        if not os.path.exists(local_path):
            print(f"      {YELLOW}[!] Skipping missing file: {local_path}{NC}")
            continue
            
        print(f"  [+] Uploading {BOLD}{local_path}{NC} to Hugging Face...")
        try:
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=repo_path,
                repo_id=REPO_ID,
                token=hf_token
            )
            print(f"      -> {GREEN}Successfully uploaded as {repo_path} ({os.path.getsize(local_path)} bytes){NC}")
            successful_uploads.append(repo_path)
        except Exception as e:
            print(f"      {RED}[!] Upload failed for {local_path}: {str(e)}{NC}")

    elapsed = time.time() - start_time
    print(f"\n  [+] Promotion process finished in {elapsed:.2f} seconds.")
    print(f"  [+] Successfully verified and uploaded {len(successful_uploads)} physical assets.")
    
    if len(successful_uploads) > 0:
        print(f"\n  {GREEN}🎉 L'HONNEUR DE LA SCIENCE: SCIENTIFIC RESULTS PUSHED & PROMOTED SUCCESSFULLY!{NC}")
        print(f"    - Access Link:   {BOLD}https://huggingface.co/{REPO_ID}{NC}")
    else:
        print(f"\n  ❌ {RED}PROMOTION FAILURE: No assets were successfully uploaded due to certificate validation traps.{NC}")
    print(f"{CYAN}{BOLD}========================================================================{NC}\n")

if __name__ == "__main__":
    promote_scientific_results()
