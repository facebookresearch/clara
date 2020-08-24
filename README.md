# CLARA: Confidence of Labels and Raters

An implementation of the Gibbs sampler for the model (together with simulators to generate synthetic data) used in the paper "CLARA: Confidence of Labels and Raters" (KDD'20).

```
@inproceedings{clara-kdd-20,
    author = {Viet-An Nguyen and Peibei Shi and Jagdish Ramakrishnan and Udi Weinsberg and Henry C. Lin and Steve Metz and Neil Chandra and Jane Jing and Dimitris Kalimeris},
    title = {{CLARA: Confidence of Labels and Raters}},
    booktitle = {Proceedings of the 26th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD ’20)},
    year = {2020},
}
```

## Simulating Data

### Generate data without classifier scores

We can generate a dataset with 1000 items with the true prevalence `theta = [0.8, 0.2]` and all labelers share the same confusion matrix `psi = [[0.9, 0.1], [0.05, 0.95]]` as follow:

``` python
from simulator import generate_dataset_tiebreaking
df = generate_dataset_tiebreaking(
    dataset_id=0,
    theta=np.array([0.8, 0.2]),
    psi=np.array([[0.9, 0.1], [0.05, 0.95]]),
    num_items=1000,
)
```

The simulated data will look like:

| dataset | id    |  labelers |   ratings | true_rating |
|:-------:|-------|----------:|----------:|:-----------:|
|    0    | 0_995 |    [0, 0] |    [0, 0] |      0      |
|    0    | 0_996 |    [0, 0] |    [0, 0] |      0      |
|    0    | 0_997 | [0, 0, 0] | [0, 1, 0] |      0      |
|    0    | 0_998 |    [0, 0] |    [0, 0] |      0      |
|    0    | 0_999 |    [0, 0] |    [0, 0] |      0      |

### Generate data with classifier scores

``` python
from simulator import generate_dataset_tiebreaking_with_scores
df = generate_dataset_tiebreaking_with_scores(
    dataset_id=1,
    theta=np.array([0.8, 0.2]),
    psi=np.array([[0.9, 0.1], [0.05, 0.95]]),
    num_items=1000,
)
```

## Using CLARA

### Fit the model

To fit a CLARA model with a single confusion matrix shared across all labelers

``` python
model = ClaraGibbs(burn_in=2000, num_samples=1000, sample_lag=3)
model.fit(A=1, R=2, ratings=np.array(df.ratings))
```

### Estimate the prevalence

``` python
model.get_prevalence()
```

### Estimate the confusion matrix

``` python
model.get_confusion_matrix(labeler_id=0)
```

## Installation

**Installation Requirements**

* Python >= 3.6
* numpy
* pandas
* scipy

## License

You may find out more about the license [here](LICENSE).
