import csv
import os

def save_feedback(tweet_text, corrected_label):
    """Saves the tweet text and corrected label to a CSV file."""
    directory = "Dashboard/HITL"
    file_path = os.path.join(directory, "../../Data_Set/Processed_Data_Set/human_feedback.csv")
    
    # Ensure directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    file_exists = os.path.isfile(file_path)
    
    try:
        with open(file_path, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # Write header if new file
            if not file_exists:
                writer.writerow(['tweet_text', 'label'])
            writer.writerow([tweet_text, corrected_label])
        return True
    except Exception as e:
        print(f"Error saving feedback: {e}")
        return False