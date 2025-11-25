from pathlib import Path
import json
import requests
from tqdm import tqdm

OPEN_CITATION_DATA_PATH = Path("results_20251022_161614_limitNone_opencitations.json")
API = "https://api.opencitations.net/meta/v1/metadata/omid:{matched_id}"

with open(OPEN_CITATION_DATA_PATH, "r") as f:
    open_citation_data = json.load(f)

# author name and other field is missing in some records, fix them by querying the API
for item in tqdm(open_citation_data):
    matched = item.get('search_results', {}).get('opencitations', {}).get('metadata_search', {}).get('top_result')
    if matched:
        # Fix the IDs to be in the correct format
        ids = matched.get('ids', {})
        
        # Fix OMID: convert from "https://w3id.org/oc/meta/br/06210459208" to "br/06210459208"
        if 'omid' in ids and isinstance(ids['omid'], str):
            omid = ids['omid']
            if omid.startswith('https://w3id.org/oc/meta/'):
                omid = omid.replace('https://w3id.org/oc/meta/', '')
                parts = omid.rstrip('/').split('/')
                if len(parts) >= 2:
                    ids['omid'] = '/'.join(parts[-2:])  # Get last two parts like 'br/06210459208'
            elif omid.startswith('https://w3id.org/oc/'):
                ids['omid'] = omid.replace('https://w3id.org/oc/', '')
        
        # Fix DOI: convert from "https://doi.org/10.1136/bcr-2015-212291" to "10.1136/bcr-2015-212291"
        if 'doi' in ids and isinstance(ids['doi'], str):
            doi = ids['doi']
            if doi.startswith('https://doi.org/'):
                ids['doi'] = doi.replace('https://doi.org/', '')
            elif doi.startswith('http://dx.doi.org/'):
                ids['doi'] = doi.replace('http://dx.doi.org/', '')
        
        # Set the fixed IDs back
        matched['ids'] = ids
        
        # Get the OMID for API call if needed
        matched_id = ids.get('omid')
        if matched_id:
            api_url = API.format(matched_id=matched_id)
            try:
                response = requests.get(api_url, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    # Update item with missing fields from API response
                    if isinstance(data, list) and len(data) > 0:
                        api_result = data[0]
                        
                        # Update author information
                        if api_result.get('author') and not matched.get('first_author'):
                            # Extract first author from the author string
                            authors = api_result.get('author', '')
                            if ';' in authors:
                                first_author_str = authors.split(';')[0].strip()
                            else:
                                first_author_str = authors.strip()
                            
                            # Extract just the name part (before [omid:...])
                            if '[' in first_author_str:
                                first_author_name = first_author_str.split('[')[0].strip()
                            else:
                                first_author_name = first_author_str
                            
                            matched['first_author'] = first_author_name
                        
                        # Update journal information
                        if api_result.get('venue') and not matched.get('journal'):
                            venue = api_result.get('venue', '')
                            # Remove the [issn:...] part
                            if '[' in venue:
                                journal = venue.split('[')[0].strip()
                            else:
                                journal = venue
                            matched['journal'] = journal
                        
                        # Update publisher
                        if api_result.get('publisher'):
                            matched['publisher'] = api_result.get('publisher')
                        
                        # Update DOI if missing - store just the DOI without https://doi.org/ prefix
                        if api_result.get('id') and 'doi:' in api_result.get('id'):
                            doi = api_result.get('id').split('doi:')[1].split()[0]
                            if doi:
                                # Remove any URL prefixes if present
                                doi = doi.replace('https://doi.org/', '').replace('http://dx.doi.org/', '')
                                if not matched.get('ids', {}).get('doi'):
                                    matched.setdefault('ids', {})['doi'] = doi
                        
                        print(f"Updated: {matched.get('title', 'N/A')[:50]}...")
            except Exception as e:
                print(f"Error fetching data for {matched_id}: {e}")

# Save updated data back to the file
with open(OPEN_CITATION_DATA_PATH, "w") as f:
    json.dump(open_citation_data, f, indent=2)

print(f"\nUpdated {len(open_citation_data)} records and saved to {OPEN_CITATION_DATA_PATH}")

