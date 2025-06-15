# Scrapes character bios from the Attack on Titan fandom wiki
# Output: a list of dictionaries with title and bio text

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://attackontitan.fandom.com/wiki"

# List of character page slugs
CHARACTERS = [
    "Eren_Yeager",
    "Mikasa_Ackerman",
    "Armin_Arlert",
    "Levi_Ackerman",
    "Historia_Reiss",
    "Reiner_Braun",
    "Annie_Lenhardt",
    "Hange_Zoë",
    "Erwin_Smith",
    "Zeke_Yeager",
]

def get_character_bios(limit_paragraphs=5):
    bios = []

    for name in CHARACTERS:
        url = f"{BASE_URL}/{name}"
        print(f"Scraping: {url}")
        res = requests.get(url)

        if res.status_code != 200:
            print(f"Failed to fetch {name}")
            continue

        soup = BeautifulSoup(res.text, 'html.parser')
        content = soup.find("div", class_="mw-parser-output")

        if not content:
            print(f"No content found for {name}")
            continue

        # Get the first few relevant paragraphs
        paragraphs = content.find_all("p", recursive=False)[:limit_paragraphs]
        text = " ".join(p.get_text(strip=True) for p in paragraphs)

        bios.append({
            "source": "character_bio",
            "title": name.replace("_", " "),
            "text": text
        })

    return bios

# Optional test run (remove if importing elsewhere)
if __name__ == "__main__":
    data = get_character_bios()
    for entry in data:
        print(f"\n## {entry['title']} ##\n{entry['text'][:300]}...\n")
