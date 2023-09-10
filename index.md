---
 layout: default
 title: University of Pennsylvania Bias Bounties 2023
 description: CIS 4230/5230
---

Welcome to the first iteration of Bias Bounties brought to you by CIS 4230/5230!

The original "Bias Bounty" structure outlined in Globus-Harris etal. \[[GHKR2019](https://arxiv.org/pdf/2201.10408.pdf)\] creates a structure to reduce machine learning bias or discrimination through crowdsourcing. A competitor is rewarded when they are able to not only identify a region in which the current model performs poorly, but also creates a new predictive model which does strictly better than the biased model on the region. The amount in which a competitor is rewarded is proportional to the reduction in overall model error as well as a constant factor for finding an update.

For the 2023 iteration of Bias Bounties, the competition task is predicting an individual's annual income from demographic data collected in the US Census. The data is drawn from the Folktables package with Southern states as the intended impact area of the model. The loss function used for this regression task is mean square error, and statistics are published using root mean squared error to keep from exploding errors (there is no rescaling). Incomes in this task are capped at $100,000 in order to create a more fine-grain prediction task. The data dictionary can be found [here](https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMS_Data_Dictionary_2021.pdf).

Since this competition is occuring in a class setting, each team has two Pointer Decision List models to which their potential updates will be submitted to. The first PDL is the standard global model for which all submitted pairs attempt an update. The second PDL is a private team model where only the respective team's submitted updates are attempted. As outlined in the project description distributed via Slack, each team will be graded on the number of updates and error of their private model, and will be given extra credit for global updates. Each team's emphasis should be on the global model since an update on the global model will likely lead to a reduction in error on their private model.

Happy Hunting!