# Descriptive Statistics

It is evident that certain action units are completely blacked out by furhat, specifically
action unit 6 and 10 which have zero values throughout. 

## Observation of absolute values

Furhat and Metahuman does not seem to pick up on action unit 04: **Brow Lowerer**

Furhat specifically seems to overgenerate action unit 05: **Upper Lid Raiser** 
which is evident both in the variance and mean of the action unit in descriptive statistics.
The variance seen in this action unit is likely somewhat random, confusing the classifier.

It is important to note that even if the mean values are smaller, 
we currently normalize the data separately for each condition, meaning 
that small differences in mean values can still have a large impact on the classification.

## Observation of separately normalized values

It might make sense to normalize the data for each condition separately, to make the 
descriptive statistics more comparable across conditions.



