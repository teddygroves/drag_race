functions {
  real maxi_challenge_lp_one_winner(vector abilities){
    // winner
    real winner_lp = abilities[1] - log_sum_exp(abilities);
    // assume bottom two are maxi-exchangeable
    vector[2] possibilities;
    int n = rows(abilities);
    possibilities[1] =
      // bottom is worse than second bottom
      -abilities[n] - log_sum_exp(-abilities[2:n])
      -abilities[n-1] - log_sum_exp(-abilities[2:n-1]);
    possibilities[2] =
      // second bottom is worse than bottom
      -abilities[n-1] - log_sum_exp(-abilities[2:n])
      -abilities[n] - log_sum_exp(append_row(-abilities[2:n-2], -abilities[n]));
    return winner_lp + log_sum_exp(possibilities);
  }
  real maxi_challenge_lp_two_winners(vector abilities){
    vector[rows(abilities)] swapped_abilities = append_row([abilities[2], abilities[1]]', abilities[3:]);
    vector[2] lp_cases = [maxi_challenge_lp_one_winner(abilities),
                          maxi_challenge_lp_one_winner(swapped_abilities)]';
    return log_sum_exp(lp_cases);
  }
  real maxi_challenge_lp(vector abilities, int double_winner){
    return
      double_winner == 1 ?
      maxi_challenge_lp_two_winners(abilities) :
      maxi_challenge_lp_one_winner(abilities);
  }
}
data {
  int<lower=1> N;
  int<lower=1> K;
  int<lower=1> N_episode;
  int<lower=1> N_contestant;
  int<lower=1> N_contestant_next;
  int<lower=1,upper=N_contestant> contestant[N];  // nb ranked per episode first > safe > bottom2
  int<lower=1,upper=N_contestant> contestant_next[N_contestant_next];
  int<lower=1> N_episode_contestant[N_episode];
  int<lower=0,upper=1> double_winner[N_episode];
  matrix[N_contestant, K] X;
}
parameters {
  vector[N_contestant] ability_maxi_z;
  real<lower=0> sigma_ability_maxi;
  vector[N_contestant] ability_lipsync_z;
  real<lower=0> sigma_ability_lipsync;
  vector[K] beta_maxi;
  vector[K] beta_lipsync;
}
transformed parameters {
  vector[N_contestant] ability_maxi = X * beta_maxi + ability_maxi_z * sigma_ability_maxi;
  vector[N_contestant] ability_lipsync = X * beta_lipsync + ability_lipsync_z * sigma_ability_lipsync;
}
model {
  int pos = 1;
  // priors
  ability_maxi_z ~ student_t(4, 0, 1);
  ability_lipsync_z ~ student_t(4, 0, 1);
  beta_maxi ~ normal(0, 1);
  beta_lipsync ~ normal(0, 1);
  sigma_ability_maxi ~ normal(0, 1.5);
  sigma_ability_lipsync ~ normal(0, 1.5);
  // likelihood
  for (e in 1:N_episode){
    int n = N_episode_contestant[e];
    vector[n] episode_abilities_maxi = segment(ability_maxi[contestant], pos, n);
    vector[n] episode_abilities_lipsync = segment(ability_lipsync[contestant], pos, n);
    target += maxi_challenge_lp(episode_abilities_maxi, double_winner[e]);
    target += bernoulli_logit_lpmf(1 | episode_abilities_lipsync[n-1] - episode_abilities_lipsync[n]);
    pos += n;
  }
}
generated quantities {
  int is_eliminated[N_contestant] = rep_array(0, N_contestant);
  {
    int bottom_two[2] = contestant_next[sort_indices_asc(ability_maxi[contestant_next])[1:2]];
    int eliminated_contestant = bottom_two[sort_indices_asc(ability_lipsync[bottom_two])[1]];
    is_eliminated[eliminated_contestant] = 1;
  }
}
