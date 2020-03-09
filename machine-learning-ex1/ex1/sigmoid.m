function g = sigmoid(z)
  
  
  %sig = @(z) 1./(1 + exp(-z));
  g = 1 ./ (1 + exp(-z));
  %g = sig(z);
  
endfunction
