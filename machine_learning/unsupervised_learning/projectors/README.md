# Tensorflow Projectors

# Contents 

- `tf_projector_all`: Projector for specific conditions: metahuman, furhat and original 
- `tf_projector_condition`: Projector for specific condition
- `tf_projector_external`: Projector for external data (3000 items from database)

# Observations 

Using T-SNE, we can see some clusters based on emotions, e.g. `interest_curiosity` and `gratitude`. 
But also clear actor specific clusters. 

PCA likewise clusters actors rather than emotions. Although there seems to be somewhat of a pattern, where positive emotions
tend towards one side of axis. 

**TODO:** Try to use actor specific normalization, either low level normalization, or functional normalization of the items.

