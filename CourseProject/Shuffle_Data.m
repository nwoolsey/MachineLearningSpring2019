function [ perm ] = Shuffle_Data( perm, w, n )

D = zeros(n,n);

P = length(perm);

s = P/n;

for i = 1:n
    for j = i+1:n
        
        D(i,j) = norm(w(:,i) - w(:,j))^2;
        
        D(j,i) =D(i,j) ;
    end
    
end

while any(D(:) > 0)
    
    [i,j] = find(D == max(D(:)));
    
    D([i(1) j(1)],:) = 0;
    
    D(:,[i(1) j(1)]) = 0;
    
    subperm1 = randperm(s);
    
    subperm2 = randperm(s);
    
    temp1 = perm((i(1)-1)*s+subperm1);
    
    temp2 = perm((j(1)-1)*s+subperm2);
    
    perm((i(1)-1)*s+1:(i(1)-1/2)*s) = temp1(1:s/2);
    
    perm((i(1)-1/2)*s+1:i(1)*s) = temp2(1:s/2);
    
    perm((j(1)-1)*s+1:(j(1)-1/2)*s) = temp1(s/2+1:s);
    
    perm((j(1)-1/2)*s+1:j(1)*s) = temp2(s/2+1:s);
       
end