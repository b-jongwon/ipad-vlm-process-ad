"""Read only the user-authorized key file. Never print its contents or key."""
import os
import re
from pathlib import Path

KEY_FILE = Path(os.environ.get('OPENAI_KEY_FILE', '.env.local'))

def openai_key():
    value = os.environ.get('OPENAI_API_KEY', '').strip()
    if value:
        return value
    for path in [Path(__file__).resolve().parents[1]/'.env.local', KEY_FILE]:
        if not path.is_file():
            continue
        try:
            content = path.read_text(encoding='utf-8-sig').strip()
        except UnicodeDecodeError:
            content = path.read_text(encoding='cp949').strip()
        match = re.search(r'(?m)^\s*OPENAI_API_KEY\s*=\s*[\"\']?([^\s\"\']+)', content)
        if match:
            return match.group(1)
        match = re.search(r'\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}', content)
        if match:
            return match.group(0)
    raise RuntimeError('No usable OpenAI key found in the authorized files')

if __name__ == '__main__':
    try:
        openai_key()
        print('OpenAI key present: yes (value not displayed)')
    except RuntimeError:
        print('OpenAI key present: no')
