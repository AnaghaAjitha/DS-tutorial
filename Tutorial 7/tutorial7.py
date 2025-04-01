from itertools import combinations

# Function to generate frequent itemsets using the Apriori algorithm
def generate_frequent_itemsets(transactions, min_support):
    itemsets = {}  # Dictionary to store frequent itemsets and their support values
    single_items = set()  # Set to store individual items
    
    # Collect all unique items from transactions
    for transaction in transactions:
        for item in transaction:
            single_items.add(frozenset([item]))
    
    # Count occurrences of each item
    item_counts = {item: 0 for item in single_items}
    for transaction in transactions:
        for item in single_items:
            if item.issubset(transaction):  # Check if the item is present in the transaction
                item_counts[item] += 1
    
    # Filter out items that do not meet the minimum support threshold
    num_transactions = len(transactions)
    frequent_itemsets = {item: count / num_transactions for item, count in item_counts.items() if count / num_transactions >= min_support}
    
    k = 2  # Start with item pairs
    current_itemsets = set(frequent_itemsets.keys())
    
    while current_itemsets:
        new_itemsets = set()
        # Generate new candidate itemsets by merging existing frequent itemsets
        for itemset1 in current_itemsets:
            for itemset2 in current_itemsets:
                union_set = itemset1 | itemset2  # Union of two itemsets
                if len(union_set) == k and union_set not in new_itemsets:
                    new_itemsets.add(union_set)
        
        # Count occurrences of new itemsets
        item_counts = {item: 0 for item in new_itemsets}
        for transaction in transactions:
            for itemset in new_itemsets:
                if itemset.issubset(transaction):  # Check if itemset is present in transaction
                    item_counts[itemset] += 1
        
        # Filter itemsets based on support threshold
        new_frequent_itemsets = {item: count / num_transactions for item, count in item_counts.items() if count / num_transactions >= min_support}
        frequent_itemsets.update(new_frequent_itemsets)  # Add new frequent itemsets to the dictionary
        
        current_itemsets = set(new_frequent_itemsets.keys())  # Update the current itemsets for the next iteration
        k += 1  # Increase itemset size
    
    return frequent_itemsets

# Function to generate association rules from frequent itemsets
def generate_association_rules(frequent_itemsets, min_confidence):
    rules = []  # List to store association rules
    for itemset in frequent_itemsets.keys():
        if len(itemset) > 1:
            # Generate all possible antecedent-consequent pairs
            for i in range(1, len(itemset)):
                for antecedent in combinations(itemset, i):
                    antecedent = frozenset(antecedent)
                    consequent = itemset - antecedent
                    confidence = frequent_itemsets[itemset] / frequent_itemsets[antecedent]  # Calculate confidence
                    if confidence >= min_confidence:
                        rules.append((antecedent, consequent, confidence))  # Store the rule
    return rules

# Example dataset (transactions)
transactions = [
    {'milk', 'bread', 'butter'},
    {'beer', 'bread'},
    {'milk', 'bread', 'butter', 'beer'},
    {'bread', 'butter'},
    {'milk', 'butter'},
]

# Set minimum support and confidence thresholds
min_support = 0.5  # Minimum fraction of transactions an itemset must appear in
min_confidence = 0.7  # Minimum confidence for association rules

# Run Apriori algorithm
frequent_itemsets = generate_frequent_itemsets(transactions, min_support)
rules = generate_association_rules(frequent_itemsets, min_confidence)

# Print frequent itemsets
print("Frequent Itemsets:")
for itemset, support in frequent_itemsets.items():
    print(f"{set(itemset)}: {support:.2f}")

# Print generated association rules
print("\nAssociation Rules:")
for antecedent, consequent, confidence in rules:
    print(f"{set(antecedent)} -> {set(consequent)} (Confidence: {confidence:.2f})")
