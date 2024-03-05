import json

def load_json(file_path):
    """Load JSON data from a file."""
    with open(file_path, 'r') as file:
        return json.load(file)

def save_json(data, file_path):
    """Save data to a JSON file."""
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

def create_entry_key(entry):
    """Create a unique key for each entry based on infra and vehicle file names."""
    infra_file_name = entry['infrastructure_pointcloud_path'].split('/')[-1].split('.')[0]
    vehicle_file_name = entry['vehicle_pointcloud_path'].split('/')[-1].split('.')[0]
    return f"{infra_file_name}-{vehicle_file_name}"

def find_matching_entries_efficient(data_dict, group_entries):
    """Find matching entries in data.json based on easy_group.json using a more efficient method."""
    
    filtered_entries = []
    for group in group_entries:
        key = f"{group['infra_file_name']}-{group['vehicle_file_name']}"
        if key in data_dict:
            filtered_entries.append(data_dict[key])

    return filtered_entries

# Load data from JSON files
data_entries = load_json(f'/home/massimo/vehicle_infrastructure_calibration/data/cooperative-vehicle-infrastructure/cooperative/data_info.json')
easy_group_entries = load_json(f'/home/massimo/vehicle_infrastructure_calibration/intermediate_output/111/easy_group_k15_totalcnt6600.json')
hard_group_entries = load_json(f'/home/massimo/vehicle_infrastructure_calibration/intermediate_output/111/medium_group_k15_totalcnt6600.json')

# Create a dictionary for quick lookup
data_dict = {create_entry_key(entry): entry for entry in data_entries}

# Find and filter matching entries with efficient search
filtered_easy_entries = find_matching_entries_efficient(data_dict, easy_group_entries)
filtered_hard_entries = find_matching_entries_efficient(data_dict, hard_group_entries)

# Save the filtered entries to a new JSON file
save_json(filtered_easy_entries, 'easy_data.json')
save_json(filtered_hard_entries, 'hard_data.json')

print("Filtered data has been saved to easy_data.json and hard_data.json")


