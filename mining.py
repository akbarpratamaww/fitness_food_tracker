import pandas as pd
from collections import defaultdict
from itertools import combinations
import random

def run_apriori(transactions, min_support=0.05, min_confidence=0.2):
    """
    Implements a clean Apriori algorithm from scratch for 1-itemsets and 2-itemsets (pairs).
    This handles typical food combinations and is easy to follow for academic projects.
    
    Parameters:
    - transactions: List of lists/sets containing food names.
    - min_support: Minimum support threshold (float between 0 and 1).
    - min_confidence: Minimum confidence threshold (float between 0 and 1).
    
    Returns:
    - rules: List of dicts representing rules: A -> B.
    - frequent_1: Dict of frequent 1-itemsets and their support.
    - frequent_2: Dict of frequent 2-itemsets and their support.
    """
    total_tx = len(transactions)
    if total_tx == 0:
        return [], {}, {}
    
    # 1. Count single items (C1)
    c1 = defaultdict(int)
    for tx in transactions:
        # Deduplicate items per transaction
        for item in set(tx):
            c1[item] += 1
            
    # Filter by support (L1)
    frequent_1 = {}
    for item, count in c1.items():
        support = count / total_tx
        if support >= min_support:
            frequent_1[item] = support
            
    # 2. Count pairs (C2) using only items in L1
    c2 = defaultdict(int)
    frequent_1_items = set(frequent_1.keys())
    
    for tx in transactions:
        tx_filtered = [item for item in set(tx) if item in frequent_1_items]
        if len(tx_filtered) >= 2:
            # Generate pairs (sorted to avoid duplicate pairs in different order)
            for pair in combinations(sorted(tx_filtered), 2):
                c2[pair] += 1
                
    # Filter by support (L2)
    frequent_2 = {}
    for pair, count in c2.items():
        support = count / total_tx
        if support >= min_support:
            frequent_2[pair] = support
            
    # 3. Generate Association Rules (A -> B)
    rules = []
    for (item_a, item_b), support_ab in frequent_2.items():
        support_a = frequent_1[item_a]
        support_b = frequent_1[item_b]
        
        # Rule 1: item_a -> item_b
        confidence_a_to_b = support_ab / support_a
        lift_a_to_b = confidence_a_to_b / support_b
        if confidence_a_to_b >= min_confidence:
            rules.append({
                'antecedent': item_a,
                'consequent': item_b,
                'support': support_ab,
                'confidence': confidence_a_to_b,
                'lift': lift_a_to_b
            })
            
        # Rule 2: item_b -> item_a
        confidence_b_to_a = support_ab / support_b
        lift_b_to_a = confidence_b_to_a / support_a
        if confidence_b_to_a >= min_confidence:
            rules.append({
                'antecedent': item_b,
                'consequent': item_a,
                'support': support_ab,
                'confidence': confidence_b_to_a,
                'lift': lift_b_to_a
            })
            
    # Sort rules by confidence descending, then lift descending
    rules = sorted(rules, key=lambda x: (x['confidence'], x['lift']), reverse=True)
    
    return rules, frequent_1, frequent_2

def get_sample_transactions():
    """
    Generates a realistic synthetic multi-user dataset of 120 food transactions.
    It contains clear intentional pattern associations for demonstration.
    """
    # Intentional pattern rules:
    # 1. Chicken Breast -> Rice (Gym diet)
    # 2. Egg -> Bread (Classic Breakfast)
    # 3. Salad -> Yogurt (Healthy diet)
    # 4. Pizza -> Cheese (Cheesy cheat meal)
    # 5. Burger -> French Fries (Fast food combo)
    # 6. Salmon -> Broccoli (Omega-3 lunch)
    # 7. Chocolate Cake -> Ice Cream (Sweet dessert)
    
    templates = [
        ['Chicken Breast', 'Rice', 'Broccoli'],
        ['Chicken Breast', 'Rice', 'Spinach'],
        ['Egg', 'Bread', 'Milk'],
        ['Egg', 'Bread', 'Orange'],
        ['Salad', 'Yogurt', 'Apple'],
        ['Salad', 'Yogurt', 'Orange'],
        ['Pizza', 'Cheese'],
        ['Pizza', 'Cheese', 'Ice Cream'],
        ['Burger', 'French Fries'],
        ['Burger', 'French Fries', 'Cheese'],
        ['Salmon', 'Broccoli', 'Carrot'],
        ['Salmon', 'Spinach', 'Rice'],
        ['Chocolate Cake', 'Ice Cream'],
        ['Apple', 'Banana', 'Yogurt'],
        ['Banana', 'Milk']
    ]
    
    transactions = []
    random.seed(42) # For reproducible results
    
    # Generate 120 transactions by sampling templates with some random noise/extra items
    all_fruits = ['Apple', 'Banana', 'Orange']
    
    for _ in range(120):
        base = list(random.choice(templates))
        # 20% chance to add a random fruit
        if random.random() < 0.2:
            fruit = random.choice(all_fruits)
            if fruit not in base:
                base.append(fruit)
        # 10% chance to add Milk
        if random.random() < 0.1:
            if 'Milk' not in base:
                base.append('Milk')
        transactions.append(base)
        
    return transactions
