# EvidenceNetworksDemo
[![arXiv](https://img.shields.io/badge/arXiv-2305.11241-b31b1b.svg)](https://arxiv.org/abs/2305.11241) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Simple demo using Evidence Networks
(Jeffrey & Wandelt: https://arxiv.org/abs/2305.11241)

Evidence Networks can enable Bayesian model comparison when state-of-the-art methods (e.g. nested sampling) fail and even when likelihoods or priors are intractable or unknown. Bayesian model comparison, i.e. the computation of Bayes factors or evidence ratios, can be cast as an optimization problem. Though the Bayesian interpretation of optimal classification is well-known, here we change perspective and present classes of loss functions that result in fast, amortized neural estimators that directly estimate convenient functions of the Bayes factor. We have also introduced the leaky parity-odd power (l-POP) transform, leading to the novel ``l-POP-Exponential'' loss function.

### Time series demo from paper: [time_series_demo.ipynb](  https://github.com/NiallJeffrey/EvidenceNetworksDemo/blob/main/time_series_demo.ipynb)

##
![time_series_readme_fig](https://github.com/NiallJeffrey/EvidenceNetworksDemo/blob/main/time_series_data.png)
