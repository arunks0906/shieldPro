import matplotlib.pyplot as plt

# Feature importance data
features = ["Profile Pic", "#Posts", "Fullname Words",
            "Nums/Length Fullname", "Name==Username", "Description Length",
            "External URL", "Private", "#Followers", "Nums/Length Username", "#Follows"]

importance_values = [0.1026487815139758, 0.1337524219551481, 0.0547975230291738,
                     0.00908241997370342, 0.001940272780401139, 0.08336105555301652,
                     0.008473806091822817, 0.012506042789557293, 0.22749934111251505,
                     0.27890787961246893, 0.08703045558821712]

# Sort features based on importance values
sorted_indices = sorted(range(len(importance_values)), key=lambda k: importance_values[k])
features = [features[i] for i in sorted_indices]
importance_values = [importance_values[i] for i in sorted_indices]

# Plotting the bar chart
plt.figure(figsize=(10, 6))
plt.barh(features, importance_values, color='darkred')
plt.xlabel('Feature Importance')
plt.title('Feature Importance for Instagram Profile')
plt.show()
