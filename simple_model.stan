functions {
  real maxi_challenge_lp(vector abilities){
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
}
data {
  int<lower=1> N;
  int<lower=1> K;
  int<lower=1> N_episode;
  int<lower=1> N_contestant;
  int<lower=1,upper=N_contestant> contestant[N];  // nb ranked per episode first > safe > bottom2
  int<lower=1> N_episode_contestant[N_episode];
  matrix[N_contestant, K] X;
}
parameters {
  vector[N_contestant] ability_maxi_z;
  real<lower=0> sigma_ability_maxi;
  vector[N_contestant] ability_lipsync_z;
  real<lower=0> sigma_ability_lipsync;
  vector[K] beta;
}
transformed parameters {
  vector[N_contestant] ability_maxi = X * beta + ability_maxi_z * sigma_ability_maxi;
  vector[N_contestant] ability_lipsync = ability_lipsync_z * sigma_ability_lipsync;
}
model {
  int pos = 1;
  // priors
  ability_maxi_z ~ student_t(4, 0, 1);
  ability_lipsync_z ~ student_t(4, 0, 1);
  beta ~ normal(0, 1);
  sigma_ability_maxi ~ normal(0, 1.5);
  sigma_ability_lipsync ~ normal(0, 1.5);
  // likelihood
  for (e in 1:N_episode){
    int n = N_episode_contestant[e];
    vector[n] episode_abilities_maxi = segment(ability_maxi[contestant], pos, n);
    vector[n] episode_abilities_lipsync = segment(ability_lipsync[contestant], pos, n);
  target += maxi_challenge_lp(episode_abilities_maxi);
  target += bernoulli_logit_lpmf(1 | episode_abilities_lipsync[n-1] - episode_abilities_lipsync[n]);
  pos += n;
  }
}
