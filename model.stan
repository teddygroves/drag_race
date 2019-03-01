functions {
  real exploded_logit_lpmf_subterm(vector abilities, int first_tie, int last_tie){
    /* NB abilities are assumed to be in true rank order*/
    real out = 0;
    for(t in first_tie:last_tie) {
      out += abilities[t] - log_sum_exp(abilities[t:]);
    }
    return out;
  }
  real exploded_logit_lpmf_term(vector abilities,
                                int first_tied_ability_ix,
                                int last_tied_ability_ix){
    real out = 0;
    int n_ties = 1 + last_tied_ability_ix - first_tied_ability_ix;
    int n_permutations = get_n_permutations(n_ties);
    vector[n_ties] tied = abilities[first_tied_ability_ix:last_tied_ability_ix];
    vector[n_permutations] permutations = get_permutations(tied);
    for (p_ix in 1:n_permutations){
      tied_abilities = permutations[p_ix];
      vector[n_ability] permutation_abilities =
        append_row(abilities[:first_tied_ability_ix-1],
                   append_row(tied_abilities, abilities[last_tied_ability+1:]));
      subterm = exploded_logit_lpmf_subterm(permutation_abilities,
                                            first_tied_ability_ix,
                                            last_tied_ability_ix);
      out += exp(subterm);
    }
    return log(out);

  }
  real exploded_logit_with_ties_lpmf(int[] ranks, vector abilities){
    /* nb ranks don't need to be unique*/
    int J = len(ranks);
    int K = max(ranks);

    // array of rank counts
    int d[K] = rep_array(0, K);
    for (j in 1:J){
      d[ranks[j]] += 1;
    }

    real lprob = 0;
    int pos = 1;
    for (k in 1:K){
      real lprob_k = 0;
      int dk = d[k];
      n_perm = tgamma(dk);
      vector[dk] tied_abilities = segment(abilities, pos, dk);
      pos += dk;

      int[dk] true_rank_permutations[n_perm] = get_permutations(d[k]);

      for (i in 1:nperm){
        p = true_rank_permutations[i];
        lprob_p = 0;
        for (r in 1:dk){
          lprob_p += tied_abilities[r]
            - (log_sum_exp(tied_abilities[r:dk])
               + log_sum_exp(abilities[k:]));

        }
        lprob_k +=
          }
    }
    
    }
}
data {
  int<lower=1> N;
  int<lower=1> N_episode;
  int<lower=1> N_contestant;
  int<lower=1,upper=N_contestant> contestant[N];  // nb they are ranked per episode
  int<lower=1> N_episode_contestant[N_episode];
  int<lower=1> episode_rank[N];
}
parameters {
  vector[N_contestant] ability;
  vector[N_contestant] ability_lipsync;
  real sigma_ability;
}
model {
  // priors
  ability ~ normal(0, sigma_ability);
  // likelihood
  pos = 1;
  for (e in 1:N_episode){
    int n = N_episode_contestant[e];
    vector[n] episode_abilities = abilities[contestant[pos:pos+n]];
    target += get_log_prob_winner(episode_abilities);
    target += get_log_prob_bottom_two(episode_abilities);
    pos += n;
  }
}
