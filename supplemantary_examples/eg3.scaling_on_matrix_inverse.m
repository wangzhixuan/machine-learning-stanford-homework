b_origin = [[1 1 2]; [3 1 0]; [0 -2 0 ]]

original_b_inv = inv(b_origin)

num_point = 10;

factors = zeros(num_point, 1);
sum_of_errors =zeros(num_point, 1);

for i=1:num_point
  factor = 100**i;
  factors(i) = factor;
  b = b_origin * diag([1 factor 1]);
  b_inv = inv(b);
  identity_matrix = b_inv * b;
  
  sum_of_error = 0;
  
  for r=1:3,
     for l=1:3,
        if (r!=l)
          sum_of_error += abs(identity_matrix(r,l));
        endif
     endfor
  endfor
   
  sum_of_errors(i) = log(sum_of_error);


  if (sum_of_error>0.001)
     factor
     sum_of_error
     identity_matrix
  endif
  
  
  
endfor

loglog(factors, e.^sum_of_errors)
xlabel('roughly the conditional number of matrix b')
ylabel('sum of abs(diagonal terms) in inv(b) * b')

