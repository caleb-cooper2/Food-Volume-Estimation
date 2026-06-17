# Importing subset of Nutrition5k data
## (without gutil or gcloud)

### 1. Get all filenames recursively
```bash 
python3 - << 'EOF'
import requests

params = {
    "prefix": "nutrition5k_dataset/imagery/realsense_overhead/",
    "maxResults": 1000
}
all_names = []
while True:
    resp = requests.get(
        "https://storage.googleapis.com/storage/v1/b/nutrition5k_dataset/o",
        params=params
    )
    data = resp.json()
    all_names.extend(item["name"] for item in data.get("items", []))
    token = data.get("nextPageToken")
    if not token:
        break
    params["pageToken"] = token
    print(f"  Listed {len(all_names)} so far...")

with open("overhead_objects.txt", "w") as f:
    f.write("\n".join(all_names))
print(f"Total: {len(all_names)} objects")
EOF
```

### 2. Create filtered list of filenames to download
```bash
python3 - << 'EOF'
from pathlib import Path

with open("overhead_objects.txt") as f:
    lines = [l.strip() for l in f if l.strip()]

# Extract unique dish IDs preserving order
seen_dishes = []
seen_set = set()
for line in lines:
    parts = line.split("/")
    # path is: nutrition5k_dataset/imagery/realsense_overhead/dish_XXXX/file
    if len(parts) >= 5:
        dish_id = parts[3]
        if dish_id not in seen_set:
            seen_set.add(dish_id)
            seen_dishes.append(dish_id)

# Take first 50 dishes
target_dishes = set(seen_dishes[:50])
print(f"Selected {len(target_dishes)} dishes")

# Filter all objects to only those 50 dishes
filtered = [l for l in lines if any(d in l for d in target_dishes)]
print(f"Total files to download: {len(filtered)}")

with open("download_list.txt", "w") as f:
    f.write("\n".join(filtered))
EOF
```

### 3. Download filtered list with loop
```bash
mkdir -p data/nutrition5k

while IFS= read -r object_path; do
    dest="data/$object_path"
    mkdir -p "$(dirname "$dest")"

    if [ -f "$dest" ] && [ -s "$dest" ]; then
        continue  # skip already downloaded
    fi

    encoded=$(python3 -c "import urllib.parse; print(urllib.parse.quote('$object_path', safe=''))")
    curl -s -o "$dest" "https://storage.googleapis.com/nutrition5k_dataset/$encoded"
done < download_list.txt

echo "Done"
```