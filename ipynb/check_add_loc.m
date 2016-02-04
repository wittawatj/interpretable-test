J = 5;
A = wishrnd(eye(J+1), J+20);
At = A(1:end-1, 1:end-1);

for i=1:1000
    x = randn(J+1, 1)*10;
    xt = x(1:end-1);
    diff = x'*(A\x) - xt'*(At\xt);
    assert(diff>0);
end
 
