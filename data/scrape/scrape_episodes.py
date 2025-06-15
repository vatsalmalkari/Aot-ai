import requests
from bs4 import BeautifulSoup
import re

# Fandom URLs for all AoT seasons
SEASON_URLS = {
    "Season 1": "https://attackontitan.fandom.com/wiki/List_of_Attack_on_Titan_episodes",
    "Season 2": "https://attackontitan.fandom.com/wiki/List_of_Attack_on_Titan_episodes/Season_2",
    "Season 3": "https://attackontitan.fandom.com/wiki/List_of_Attack_on_Titan_episodes/Season_3",
    "Season 4": "https://attackontitan.fandom.com/wiki/List_of_Attack_on_Titan_episodes/The_Final_Season"
}

# Curated episode titles (as they appear in the Fandom "Title" column)
TARGET_EPISODES = {
    "Season 1": [
        "To You, in 2000 Years: The Fall of Shiganshina, Part 1",
        "First Battle: The Struggle for Trost, Part 1",
        "Primal Desire: The Struggle for Trost, Part 9",
        "Female Titan: The 57th Exterior Scouting Mission, Part 1",
        "The Defeated: The 57th Exterior Scouting Mission, Part 6"
    ],
    "Season 2": [
        "Beast Titan", "I'm Home", "Warrior"
    ],
    "Season 3": [
        "Sin", "Hero", "Midnight Sun"
    ],
    "Season 4": [
        "Declaration of War"
    ]
}


def normalize_title(title):
    """
    Normalizes episode titles for consistent comparison.
    Handles various punctuation, case differences, and potential extra spaces.
    """
    normalized = title.lower()
    normalized = re.sub(r"[’']", "'", normalized)
    normalized = re.sub(r"[:]", "", normalized)
    normalized = re.sub(r"[,]", "", normalized)
    normalized = re.sub(r"[–—]", "-", normalized)
    # Remove text within parentheses or after "Transliteration"
    normalized = re.sub(r"\(japanese:.*?\)", "", normalized)
    normalized = re.sub(r"transliteration:.*", "", normalized, flags=re.IGNORECASE)
    normalized = normalized.strip('"').strip()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized

def clean_airdate(airdate_text):
    """
    Cleans up airdate string and formats it as "Month date, year".
    """
    # Remove content in parentheses, typically "(theatres)" or "(television)"
    airdate_text = re.sub(r'\s*\(.*?\)', '', airdate_text).strip()
    # Remove reference brackets like [3] or [c]
    airdate_text = re.sub(r'\[.*?\]', '', airdate_text).strip()
    # Remove any stray "airdate" text if it gets picked up
    airdate_text = re.sub(r'^airdate\s*:\s*', '', airdate_text, flags=re.IGNORECASE).strip()

    # Attempt to reformat if needed (e.g., if it's "July 11, 2018")
    # This regex looks for "Month Day, Year" or "Month Day Year"
    match = re.search(r'([A-Za-z]+)\s*(\d{1,2}),?\s*(\d{4})', airdate_text)
    if match:
        month = match.group(1)
        day = match.group(2)
        year = match.group(3)
        return f"{month} {day}, {year}"
    return airdate_text # Return original if regex doesn't match


def get_selected_episode_details_fandom():
    """
    Scrapes Fandom wiki for selected Attack on Titan episode details (title and airdate).
    Returns a list of dictionaries, each containing 'source', 'title', and 'airdate'.
    """
    all_episodes = []

    for season, url in SEASON_URLS.items():
        print(f"\n📘 Scraping {season} from {url}")
        res = requests.get(url)
        if res.status_code != 200:
            print(f"❌ Failed to fetch {url} - Status Code: {res.status_code}")
            continue

        soup = BeautifulSoup(res.text, "html.parser")

        episode_table = None
        
        # Season 1 has a different structure with tabs
        if season == "Season 1":
            season_tab_content = soup.find("div", class_="wds-tab__content", attrs={"data-tab-body": "1"})
            if season_tab_content:
                episode_table = season_tab_content.find("table", class_="wikitable")
        else:
            mw_parser_output = soup.find("div", class_="mw-parser-output")
            if mw_parser_output:
                season_heading = mw_parser_output.find(['h2', 'h3'], string=re.compile(f"{season}", re.IGNORECASE))
                if season_heading:
                    current_element = season_heading.find_next_sibling()
                    while current_element and current_element.name != 'table' and not current_element.find("table", class_="wikitable"):
                        current_element = current_element.find_next_sibling()
                    if current_element and current_element.name == 'table':
                        episode_table = current_element
                    elif current_element:
                         episode_table = current_element.find("table", class_="wikitable")
                
                if not episode_table:
                    episode_table = mw_parser_output.find("table", class_="wikitable")

        if not episode_table:
            print(f"⚠️ No suitable episode table found for {season} on {url}")
            continue
        else:
            print(f"✅ Found episode table for {season}")

        header_row = episode_table.find("tr")
        if not header_row:
            print(f"❌ Could not find header row for {season}'s table. Skipping.")
            continue

        column_headers = [th.get_text(strip=True).lower() for th in header_row.find_all(['th', 'td'])]
        column_map = {}
        for idx, header in enumerate(column_headers):
            header = re.sub(r'\s+', ' ', header).strip()
            if "title" == header:
                column_map['title'] = idx
            elif "air date" == header or "airdate" == header:
                column_map['airdate'] = idx

        print(f"DEBUG: Column map for {season}: {column_map}")
        
        if 'title' not in column_map or 'airdate' not in column_map:
            print(f"❌ Essential columns (Title or Airdate) not found for {season}. Skipping.")
            continue

        rows_with_tds = [row for row in episode_table.find_all("tr") if row.find('td')]
        
        if not rows_with_tds:
            print(f"⚠️ No episode rows with <td> cells found in table for {season}")
            continue
        else:
            print(f"✅ Found {len(rows_with_tds)} episode rows for {season}")

        target_titles_normalized = [normalize_title(t) for t in TARGET_EPISODES.get(season, [])]
        print(f"DEBUG: Target normalized titles for {season}: {target_titles_normalized}")

        for i, row in enumerate(rows_with_tds):
            cells = row.find_all("td")
            
            if column_map['title'] >= len(cells) or column_map['airdate'] >= len(cells):
                print(f"DEBUG: Skipping row {i} due to missing required columns or too few cells ({len(cells)}) for mapped indices.")
                continue

            raw_title = ""
            airdate = ""

            # --- Extract Title ---
            title_cell = cells[column_map['title']]
            title_b_tag = title_cell.find('b')
            if title_b_tag:
                raw_title = title_b_tag.get_text(strip=True)
            else:
                text_content = title_cell.get_text(strip=True)
                match = re.match(r'^(.*?)(?:\n|Transliteration:|\(Japanese:|$)', text_content, re.IGNORECASE | re.DOTALL)
                if match:
                    raw_title = match.group(1).strip()
                else:
                    raw_title = text_content.split('\n')[0].strip()
            
            raw_title = raw_title.strip('"')

            clean_title = normalize_title(raw_title)
            
            # --- Extract Airdate ---
            airdate = clean_airdate(cells[column_map['airdate']].get_text(strip=True))

            print(f"DEBUG: Row {i} - Raw Title: \"{raw_title}\" | Clean Title: \"{clean_title}\"")
            print(f"DEBUG: Row {i} - Extracted Airdate: \"{airdate}\"")

            if clean_title in target_titles_normalized:
                print(f"✅ Matched: {season} - \"{raw_title}\" (Airdate: {airdate})")
                all_episodes.append({
                    "source": "episode_details",
                    "title": f"{season} - \"{raw_title}\"",
                    "airdate": airdate
                })
            else:
                print(f"🔍 Skipped: \"{raw_title}\" (Normalized: \"{clean_title}\") - Not in target list.")
            
    return all_episodes


# Main execution block
if __name__ == "__main__":
    episodes = get_selected_episode_details_fandom()
    print(f"\n✅ Scraped {len(episodes)} target episode details.\n")
    if episodes:
        for ep in episodes:
            print(f"Title: {ep['title']}, Airdate: {ep['airdate']}")
    else:
        print("No episodes were scraped. Check debug messages above for potential issues.")