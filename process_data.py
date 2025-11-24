import pandas as pd
import json
import os
import glob

def smart_data_processor():
    """Smart data processor that handles multiple data sources"""
    print("🤖 SMART DATA PROCESSOR STARTED...")
    
    # Check for different data sources
    data_sources = [
        'training_data_processed.json',      # Existing processed data
        'large_training_dataset.json',       # Generated large dataset
        'downloaded_training_data.json',     # Downloaded datasets
        'DataSet_Misinfo_FAKE.csv',          # Your original CSV files
        'DataSet_Misinfo_TRUE.csv'           # Your original CSV files
    ]
    
    all_training_data = []
    
    # Process each available data source
    for source in data_sources:
        if os.path.exists(source):
            print(f" Processing: {source}")
            data = process_data_source(source)
            if data:
                all_training_data.extend(data)
                print(f"    Added {len(data)} samples")
            else:
                print(f"    Failed to process {source}")
        else:
            print(f"    Not found: {source}")
    
    if not all_training_data:
        print(" No training data found! Please provide data files.")
        return None
    
    # Remove duplicates (based on question text)
    unique_data = remove_duplicates(all_training_data)
    print(f" Removed {len(all_training_data) - len(unique_data)} duplicates")
    
    # Save final processed data
    output_file = 'training_data_processed.json'
    with open(output_file, 'w') as f:
        json.dump(unique_data, f, indent=2)
    
    # Print statistics
    print_stats(unique_data)
    
    print(f" Final dataset saved: {output_file}")
    print(f" Total unique samples: {len(unique_data)}")
    
    return unique_data

def process_data_source(source_path):
    """Process different types of data sources"""
    if source_path.endswith('.json'):
        return process_json_file(source_path)
    elif source_path.endswith('.csv'):
        return process_csv_file(source_path)
    else:
        print(f" Unsupported file format: {source_path}")
        return []

def process_json_file(json_path):
    """Process JSON training data files"""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Validate and standardize JSON data
        processed_data = []
        for item in data:
            if validate_training_item(item):
                processed_data.append(standardize_item(item))
        
        return processed_data
        
    except Exception as e:
        print(f" Error processing JSON {json_path}: {e}")
        return []

def process_csv_file(csv_path):
    """Process CSV files (your original format)"""
    try:
        df = pd.read_csv(csv_path)
        processed_data = []
        
        if 'DataSet_Misinfo_FAKE.csv' in csv_path:
            # Process fake data
            for _, row in df.iterrows():
                fake_text = row.get('text (fake)', row.get('text', ''))
                if fake_text and fake_text != 'No text found':
                    processed_data.append(create_misinfo_item(fake_text))
        
        elif 'DataSet_Misinfo_TRUE.csv' in csv_path:
            # Process true data
            for _, row in df.iterrows():
                true_text = row.get('text (true)', row.get('text', ''))
                if true_text and true_text != 'No text found':
                    processed_data.append(create_factual_item(true_text))
        
        return processed_data
        
    except Exception as e:
        print(f" Error processing CSV {csv_path}: {e}")
        return []

def validate_training_item(item):
    """Validate that a training item has required fields"""
    required_fields = ['question', 'expert_response', 'is_misinformation']
    return all(field in item for field in required_fields)

def standardize_item(item):
    """Ensure all items have consistent structure"""
    standardized = {
        'question': item['question'],
        'expert_response': item.get('expert_response', ''),
        'casual_response': item.get('casual_response', item.get('expert_response', '')),
        'business_response': item.get('business_response', item.get('expert_response', '')),
        'is_misinformation': item['is_misinformation'],
        'category': item.get('category', 'general'),
        'original_text': item.get('original_text', item['question'])
    }
    return standardized

def create_misinfo_item(text):
    """Create a misinformation training item"""
    return {
        "question": f"Is this accurate: {text}",
        "expert_response": f" MISINFORMATION: {text}. This claim is false and not supported by evidence.",
        "casual_response": f" FALSE: {text}. This isn't true - it's been proven wrong by experts.",
        "business_response": f" INACCURATE: {text}. This claim lacks verification and should not be used for decisions.",
        "is_misinformation": True,
        "category": "general",
        "original_text": text
    }

def create_factual_item(text):
    """Create a factual training item"""
    return {
        "question": f"What is true about: {text}",
        "expert_response": f" FACTUAL: {text}. This is supported by scientific evidence and reliable sources.",
        "casual_response": f" TRUE: {text}. This is backed by evidence and experts agree on this.",
        "business_response": f" ACCURATE: {text}. This information is verified and reliable for decision-making.",
        "is_misinformation": False,
        "category": "general",
        "original_text": text
    }

def remove_duplicates(data):
    """Remove duplicate items based on question text"""
    seen_questions = set()
    unique_data = []
    
    for item in data:
        question_key = item['question'].lower().strip()
        if question_key not in seen_questions:
            seen_questions.add(question_key)
            unique_data.append(item)
    
    return unique_data

def print_stats(data):
    """Print statistics about the dataset"""
    misinfo_count = sum(1 for item in data if item['is_misinformation'])
    factual_count = len(data) - misinfo_count
    
    categories = {}
    for item in data:
        cat = item.get('category', 'unknown')
        categories[cat] = categories.get(cat, 0) + 1
    
    print("\n DATASET STATISTICS:")
    print("=" * 40)
    print(f" Total Samples: {len(data)}")
    print(f" Misinformation: {misinfo_count}")
    print(f" Factual: {factual_count}")
    print(f" Balance: {misinfo_count/len(data)*100:.1f}% misinfo")
    print("\n Categories:")
    for category, count in categories.items():
        print(f"   - {category}: {count} samples")

if __name__ == "__main__":
    print(" SMART TRAINING DATA PROCESSOR")
    print("=" * 50)
    
    print("\n Processing all available data...")
    processed_data = smart_data_processor()
    
    if processed_data:
        print("\n DATA PROCESSING COMPLETED!")
        print(" Next: Run 'python 03_train_model_pro.py' to train with new data")
    else:
        print("\n No valid data found. Please check your data files.")
