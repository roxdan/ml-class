function J = computeCostLogistic(X, y, theta)
  
  m = length(y);
  
  hyphotesis = sigmoid(theta' * X);
  
  disp("Essa é a hipótese: ");
  disp(hyphotesis)
  
  for i = 1:size(X,2)
    logt(i) = y(i).*log(hyphotesis(:,i) + (1 - y(i)).*log(1-hyphotesis(:,i));
  endfor
  
  J = -(1/m) * sum(logt);
  disp(J)
  
endfunction
