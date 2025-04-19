import pandas as pd
from faker import Faker
import random
from data_class import Dataset, Data

# Generate fake training data
fake = Faker()

def generate_fake_dataset(num_samples=10000):
    datasets = []
    data_items = []
    
    # Create some fake datasets
    for org_id in range(1, 6):
        dataset_id = f"ds_{org_id}"
        datasets.append({
            'org_id': org_id,
            'id': dataset_id,
            'name': fake.company(),
            'type': 'text'
        })
        
        # Create data items for each dataset
        for item_id in range(1, num_samples//5 + 1):
            # Randomly include PII or not
            has_pii = random.random() < 0.3
            
            if has_pii:
                # Generate text with PII
                pii_type = random.choice(['email', 'phone', 'name', 'address', 'ssn'])
                if pii_type == 'email':
                    value = fake.email()
                elif pii_type == 'phone':
                    value = fake.phone_number()
                elif pii_type == 'name':
                    value = fake.name()
                elif pii_type == 'address':
                    value = fake.address()
                else:  # ssn
                    value = fake.ssn()
                
                # Mix with normal text
                value = f"{fake.sentence()} {value} {fake.sentence()}"
                flag = True
            else:
                # Generate normal text
                value = fake.paragraph()
                flag = False
                
            data_items.append({
                'dataset_id': dataset_id,
                'id': f"item_{item_id}",
                'value': value,
                'flag': flag
            })
    
    return pd.DataFrame(datasets), pd.DataFrame(data_items)

# Generate and save fake data
datasets_df, data_df = generate_fake_dataset()