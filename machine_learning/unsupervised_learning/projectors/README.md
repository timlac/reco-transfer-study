# Tensorflow Projectors

# Contents 

- `tf_projector_all`: Projector for specific conditions: metahuman, furhat and original 
- `tf_projector_condition`: Projector for specific condition
- `tf_projector_external`: Projector for external data (3000 items from database)

# Observations 

Without group normalization we see clear actor specific clusters. 

For the internal data it is difficult to see any emotion specific clusters.

# Plots

## PCA internal Conditions 

![PCA Conditions](../../../data/plots/PCA_conditions.png)

## T-SNE external emotions

Using T-SNE, we can see some clusters based on emotions, e.g. `interest_curiosity` and `gratitude`.

![T-SNE External](../../../data/plots/TSNE_external.png)