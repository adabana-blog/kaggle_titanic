//
// This Stan program defines a simple model, with a
// vector of values 'y' modeled as normally distributed
// with mean 'mu' and standard deviation 'sigma'.
//
// Learn more about model development with Stan at:
//
//    http://mc-stan.org/users/interfaces/rstan.html
//    https://github.com/stan-dev/rstan/wiki/RStan-Getting-Started
//

// The input data is a vector 'y' of length 'N'.
data {
  int I;
  int D;
  matrix[I,D] X;
  int<lower=0,upper=1> Survived[I];
  int I_new;
  matrix[I_new,D] X_new;
}

// The parameters accepted by the model. Our model
// accepts two parameters 'mu' and 'sigma'.
parameters {
  real b0;
  vector[D] b;
}

transformed parameters{
  real q[I];
  for (i in 1:I){
    q[i] = inv_logit(b0 + dot_product(X[i], b));
  }
}
// The model to be estimated. We model the output
// 'y' to be normally distributed with mean 'mu'
// and standard deviation 'sigma'.
model {
  for (i in 1:I){
    Survived[i] ~ bernoulli(q[i]);
  }
}

generated quantities{
  vector[I_new] q_new;
  for (i in 1:I_new){
    q_new[i] = inv_logit(b0 + dot_product(X_new[i], b));
  }
}

