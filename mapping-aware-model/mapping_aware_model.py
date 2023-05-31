stan_code="""
            data {
                int<lower=1> A; // number of annotators
                int<lower=2> K; // number of categories
                int<lower=1> N; // number of annotations
                int<lower=1> I; // number of items
                int<lower=1> L; // total number of flat labels (L in the overleaf)
                int<lower=1> D[K]; // number of ratings for each decision (D_k in the overleaf)
                int<lower=1> D_max; // max number of ratings for any decision (not in overleaf, implementation only)

                // the label decision of the l-th flat label
                int<lower=1, upper=K> c[L];

                // the index within the label decision of the l-th flat label
                int<lower=1, upper=D_max> ell[L];

                // the item the n-th annotation belongs to
                int<lower=1, upper=I> ii[N];

                // the annotator which produced the n-th annotation
                int<lower=1, upper=A> aa[N];

                // the flat index of the label of the n-th annotation
                int<lower=1, upper=L> x[N];

                vector<lower=0>[K] alpha;  // class prevalence prior

                // weight for each item
                vector<lower=0>[I] weights;

                // lower bound on the diagonal of the decision confusion matrix
                real<lower=0.5, upper=1.0> gamma;
            }

            parameters {
                simplex[K] theta;  // prevalence in the categories

                vector<lower=0, upper=1>[K] psi_diag_unconstrained[A];
                simplex[K-1] psi_cond_error[A, K];

                // shared parameters across all reviewers
                simplex[D_max] eta[K];
                vector<lower=1, upper=gamma/(1-gamma)>[D_max] rho[K, K];
            }

            transformed parameters {
                // shared parameters
                vector[D_max] log_pi[K, K];
                vector<lower=0>[D_max] pi_unnormalized[K, K];

                // per reviewer decision confusion matrix (K by K)
                vector[K] log_psi_diag[A];
                vector[K] log1m_psi_diag[A];
                vector[K-1] log_psi_cond_error[A, K];
                vector[K] log_psi[A, K];

                // rectangular confusion matrix (K by L)
                vector[L] log_psi_rectangular[A, K];

                vector[K] log_item_probs[I];

                // constructing log_psi (K x K) - copied from CLARAStanConstrainedConfusion
                for (a in 1:A) {
                    for (i in 1:K) {
                        log_psi_diag[a, i] = log(gamma + (1 - gamma) * psi_diag_unconstrained[a, i]);
                    }
                }

                log1m_psi_diag = log1m_exp(log_psi_diag);
                log_psi_cond_error = log(psi_cond_error);

                for (a in 1:A) {
                    for (i in 1:K) {
                        log_psi[a, i, i] = log_psi_diag[a, i];
                        for (j in 1:(i-1)) {
                            log_psi[a, i, j] = log1m_psi_diag[a, i] + log_psi_cond_error[a, i, j];
                        }
                        for (j in (i+1):K) {
                            log_psi[a, i, j] = log1m_psi_diag[a, i] + log_psi_cond_error[a, i, j-1];
                        }
                    }
                }

                // construct beta from eta and rho
                for (k_true in 1:K) {
                    for (k_predicted in 1:K) {
                        for (l in 1:D[k_predicted]) {
                            if (k_true == k_predicted) {
                                pi_unnormalized[k_true, k_predicted, l] = eta[k_predicted, l];
                            } else {
                                pi_unnormalized[k_true, k_predicted, l] = eta[k_predicted, l] * rho[k_true, k_predicted, l];
                            }
                        }
                        for (l in (D[k_predicted] + 1):D_max) {
                            pi_unnormalized[k_true, k_predicted, l] = 0.000001;
                        }

                        log_pi[k_true, k_predicted] = log(pi_unnormalized[k_true, k_predicted] / sum(pi_unnormalized[k_true, k_predicted]) );
                    }
                }

                // constructing log_psi_rectangular (K x L)
                for (a in 1:A) {
                    for (k in 1:K) {
                        for (l in 1:L) {
                            log_psi_rectangular[a, k, l] = log_psi[a, k, c[l]] + log_pi[k, c[l], ell[l]];
                        }
                    }
                }

                for (i in 1:I) {
                    for (k in 1:K) {
                        log_item_probs[i, k] = log(theta[k]);
                    }
                }

                for (n in 1:N) {
                    for (k in 1:K) {
                        log_item_probs[ii[n], k] = log_item_probs[ii[n], k] + log_psi_rectangular[aa[n], k, x[n]];
                    }
                }
            }

            model {
                theta ~ dirichlet(alpha);

                for (a in 1:A) {
                    for (k in 1:K) {
                        psi_diag_unconstrained[a, k] ~ beta(5, 5);
                        psi_cond_error[a, k] ~ dirichlet(rep_vector(1, K-1));
                    }
                }

                for (k_true in 1:K) {
                    eta[k_true] ~ dirichlet(append_row(rep_vector(10, D[k_true]), rep_vector(0.001, D_max-D[k_true])));

                    for (k_predicted in 1:K) {
                        for (l in 1:D_max) {
                            rho[k_true, k_predicted, l] ~ uniform(1, gamma/(1-gamma));
                        }
                    }
                }

                for (i in 1:I) {
                    target += weights[i] * log_sum_exp(log_item_probs[i]);
                }
            }

            generated quantities {
                vector[K] item_probs[I]; // the true class distribution of each item

                for(i in 1:I)
                    item_probs[i] = softmax(log_item_probs[i]);
            }
            """
