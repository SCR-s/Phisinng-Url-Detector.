Phisinng-Url-Detector
Simple, lightweight phishing URL detector written in Python.

Overview
Phisinng-Url-Detector analyzes URLs and flags potentially malicious or phishing links using a set of heuristic checks. It is designed to be easy to run locally and easy to extend with additional rules or models.

Key features
Parse and validate URLs
Heuristic checks (e.g., IP address in domain, long path, suspicious characters or TLDs)
Blacklist / whitelist support (local files or remote sources)
Optional WHOIS/domain-age checks (when configured)
Risk scoring and short explanation for each flagged check
Command-line interface for single URLs or batch scans

Tech stack
Language: Python (100%)

Typical libraries (install as needed):
requests
tldextract
validators
python-whois (optional, for WHOIS/domain age)
tqdm (optional, for progress display)
CLI: argparse (or click if preferred)
Requirements
Python 3.8+

Internet access for remote checks (optional)

Installation
Clone the repo: git clone https://github.com/SCR-s/Phisinng-Url-Detector.git cd Phisinng-Url-Detector

(Optional) Create and activate a virtual environment: python -m venv venv source venv/bin/activate # macOS / Linux venv\Scripts\activate # Windows

Install dependencies: pip install -r requirements.txt If there is no requirements.txt: pip install requests tldextract validators



Usage
Scan a single URL: python detect.py --url "http://example.com"

Scan a file of URLs (one per line) and write JSON results: python detect.py --input urls.txt --output results.json

Example output (JSON): { "url": "http://example.com", "score": 0.2, "flags": ["short_tld", "no_https"], "explanation": "Short TLD and not HTTPS — low risk" }

Adjust commands above to match your script/file names if different.




Contact / Maintainer
Maintainer: SCR-s If you want changes to wording, more examples, or direct commit help, tell me what to update or how to access the repo and I will proceed.
