import csv
import os

def save_feedback(tweet_text, corrected_label):
    """Saves the tweet text and corrected label to a CSV file."""
    # Define the absolute path based on the project structure
    # This uses a safer approach to find the directory
    file_path = os.path.join("Data_Set", "Processed_Data_Set", "human_feedback.csv")
    
    # Get the directory part of the file path
    directory = os.path.dirname(file_path)
    
    try:
        # NEW: This creates 'Data_Set/Processed_Data_Set' if it doesn't exist
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            
        file_exists = os.path.isfile(file_path)
        
        # Open in append mode
        with open(file_path, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # Write header if it's a brand new file
            if not file_exists:
                writer.writerow(['tweet_text', 'label'])
            
            # Clean the text (remove newlines that break CSV rows)
            clean_text = tweet_text.replace('\n', ' ').replace('\r', '')
            writer.writerow([clean_text, corrected_label])
            
        print(f"✓ Feedback saved to: {file_path}")
        return True
    except Exception as e:
        print(f"❌ Error saving feedback to CSV: {e}")
        return False