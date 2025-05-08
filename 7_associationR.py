# Import necessary libraries
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import matplotlib.pyplot as plt
import networkx as nx

# Sample Groceries dataset
groceries = [
    ['milk', 'bread', 'nuts', 'apple'],
    ['milk', 'bread', 'nuts'],
    ['milk', 'bread'],
    ['milk', 'bread', 'apple'],
    ['milk', 'bread', 'apple'],
    ['milk', 'bread'],
    ['milk', 'bread', 'apple'],
    ['milk', 'bread', 'apple'],
    ['milk', 'bread', 'apple'],
    ['milk', 'bread', 'apple'],
    # Add more transactions to exceed 500 if needed
]

# Convert transactions to a one-hot encoded DataFrame
te = TransactionEncoder()
te_ary = te.fit(groceries).transform(groceries)
df = pd.DataFrame(te_ary, columns=te.columns_)

# Function to apply Apriori and display rules
def mine_rules(df, min_support, min_confidence, min_len=1):
    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    rules = rules[rules['antecedents'].apply(lambda x: len(x) >= min_len)]
    return rules

# (a) Minimum support = 1% and confidence = 30%
rules_a = mine_rules(df, min_support=0.01, min_confidence=0.3)
print("Rules with support=1% and confidence=30%:")
print(rules_a.head())

# (b) Minimum support = 2% and confidence = 40%
rules_b = mine_rules(df, min_support=0.02, min_confidence=0.4)
print("\nRules with support=2% and confidence=40%:")
print(rules_b.head())

# (c) Minimum support = 3% and confidence = 50%
rules_c = mine_rules(df, min_support=0.03, min_confidence=0.5)
print("\nRules with support=3% and confidence=50%:")
print(rules_c.head())

# Sort all rules based on lift and display top 5
all_rules = pd.concat([rules_a, rules_b, rules_c]).drop_duplicates()
sorted_rules = all_rules.sort_values(by='lift', ascending=False)
print("\nTop 5 rules sorted by lift:")
print(sorted_rules.head())

# Interpret the confidence of two rules
print("\nInterpretation of confidence:")
for index, row in sorted_rules.head(2).iterrows():
    antecedents = ', '.join(list(row['antecedents']))
    consequents = ', '.join(list(row['consequents']))
    confidence = row['confidence']
    print(f"If a customer buys {antecedents}, there's a {confidence:.2%} chance they also buy {consequents}.")

# Function to plot rules using networkx
def plot_rules(rules, title):
    G = nx.DiGraph()
    for _, row in rules.iterrows():
        for antecedent in row['antecedents']:
            for consequent in row['consequents']:
                G.add_edge(antecedent, consequent, weight=row['lift'])
    pos = nx.spring_layout(G, k=1)
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color='lightblue', font_size=10, font_weight='bold', arrows=True)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels={k: f"{v:.2f}" for k, v in edge_labels.items()})
    plt.title(title)
    plt.show()

# Plot the rules
plot_rules(sorted_rules.head(5), "Top 5 Association Rules")

# Display first 5 rules with minimum length 5
rules_min_len_5 = all_rules[all_rules['antecedents'].apply(lambda x: len(x) >= 5)]
print("\nFirst 5 rules with minimum length 5:")
print(rules_min_len_5.head())