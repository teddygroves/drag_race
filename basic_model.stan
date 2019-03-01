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
