functions {
real rpdr_outcome_lp(vector ability, int W, int S, int B){
  int N_contestant = rows(ability);
  real out = 0;
  if ((W > 0) && (W < N_contestant)){
    for (w in 1:W){
      out += ability[w] - log_sum_exp(append_row(ability[w], ability[W+1:]));
    }
  }
  if ((S > 0) && (W+S < N_contestant)){
    for (s in W+1:W+S){
      out += ability[s] - log_sum_exp(append_row(ability[s], ability[W+S+1:]));
    }
  }
  if ((B > 0) && (W+S+B < N_contestant)){
    for (b in W+S+1:W+S+B){
      out += ability[b] - log_sum_exp(append_row(ability[b], ability[W+S+B+1:]));
    }
  }
  return out;
}
}
data {
  int<lower=1> N;  // Number of episode participations
  int<lower=1> K;  // Number of predictors
  int<lower=1> E;  // Number of episodes
  int<lower=1> C;  // Number of contestants
  matrix[C, K] X;
  int<lower=1> N_episode_contestant[E];
  int<lower=0> N_episode_winner[E];
  int<lower=0> N_episode_safe[E];
  int<lower=0> N_episode_bottom[E];
  int<lower=1,upper=C> contestant[N];
}
parameters {
  vector[C] ability_z;
  real<lower=0> sigma_ability;
  vector[K] beta;
}
transformed parameters {
  vector[C] ability = X * beta + ability_z * sigma_ability;
}
model {
  int pos = 1;
  // priors
  ability_z ~ normal(0, 1);
  beta ~ normal(0, 1);
  sigma_ability ~ normal(0, 1);
  // likelihood
  for (e in 1:E){
    target += rpdr_outcome_lp(ability[segment(contestant, pos, N_episode_contestant[e])],
                              N_episode_winner[e],
                              N_episode_safe[e],
                              N_episode_bottom[e]);
    pos += N_episode_contestant[e];
  }
}
