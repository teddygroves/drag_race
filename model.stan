functions {
  real rpdr_outcome_lp(vector ability, int[] episode_rank){
    real out = 0;
    int first_in_group = 1;
    for (contestant in 1:rows(ability)){
      if ((contestant > 1)
          && (episode_rank[contestant] > episode_rank[contestant-1])){
        first_in_group = contestant;
      }
      if (episode_rank[contestant] < max(episode_rank)){
        out += ability[contestant] - log_sum_exp(ability[first_in_group:]);
      }
  }
  return out;
  }
}
data {
  int<lower=1> N;         // Number of episode participations
  int<lower=1> K;         // Number of predictors
  int<lower=1> E;         // Number of episodes
  int<lower=1> C;         // Number of contestants
  int<lower=1> N_survey;  // Number of surveys
  matrix[C, K] X;         // Contestant level predictors
  // episode data
  int<lower=1> N_episode_contestant[E];
  int<lower=1,upper=6> episode_rank[N];
  int<lower=1,upper=C> contestant[N];
  // survey data
  int<lower=1,upper=C> survey_contestant[N_survey];
  int<lower=1,upper=C> survey_opponent[N_survey];
  int<lower=1> survey_count[N_survey];
  int<lower=0> survey_wins[N_survey];
  // config 
  int<lower=0,upper=1> use_survey;
  int<lower=0,upper=1> use_episodes;
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
  if (use_survey == 1){
    survey_wins ~ binomial_logit(survey_count, ability[survey_contestant] - ability[survey_opponent]);
  }
  if (use_episodes == 1){
    for (e in 1:E){
      int contestants[N_episode_contestant[e]] = segment(contestant, pos, N_episode_contestant[e]);
      int episode_ranks[N_episode_contestant[e]] = segment(episode_rank, pos, N_episode_contestant[e]);
      target += rpdr_outcome_lp(ability[contestants], episode_ranks);
      pos += N_episode_contestant[e];
    }
  }
}
