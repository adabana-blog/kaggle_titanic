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
  matrix[I_new,D];
}

// The parameters accepted by the model. Our model
// accepts two parameters 'mu' and 'sigma'.
parameters {
  real b0;
  vector[D] b;
}

transformed parameters{
  real q[I];
  q[i] = inv_logit(b0 + X[i]);
}
// The model to be estimated. We model the output
// 'y' to be normally distributed with mean 'mu'
// and standard deviation 'sigma'.
model {
  for (i in 1:I){
    Survived[i] ~ bernoulli(q[i]);
  }
}

