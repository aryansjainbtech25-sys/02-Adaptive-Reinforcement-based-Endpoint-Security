
import qrcode
from pathlib import Path

output_dir = Path("tests/images")
output_dir.mkdir(parents=True, exist_ok=True)

test_cases = {
    "https_url.png": "https://example.com",
    "http_url.png": "http://example.com",
    "plain_text.png": "Hello ARES",
    "ip_url.png": "http://192.0.2.1/login",
    "query_url.png": "https://example.com/login?id=123",
}

for filename, content in test_cases.items():
    image = qrcode.make(content)
    image.save(output_dir / filename)
    print(f"Created: {output_dir / filename}")
