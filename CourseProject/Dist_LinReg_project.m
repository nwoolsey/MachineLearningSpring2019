clear

clc

close all

n = 10;

P = n*10;

N = 2;

w = [2; 2];

A = [1*randn(P,1), ones(P,1)];

y = w(1)*A(:,1) + w(2) + N*randn(P,1)/w(1);

A(:,1) = A(:,1) ;%+ N*randn(P,1);

w_star = (A'*A)\(A'*y);

x = -5:0.1:5;

y_star = w(1)*x + w(2);

figure(1)
plot(A(:,1),y,'.',x,y_star)
legend({['training data'],['best linear fit']})
xlabel('x_i')
ylabel('y_i')


w_st_loc = zeros(2,n);
c = zeros(2,n+1);
c_shuff = zeros(2,n);
A_comp = zeros(2,2,n);
A_comp_shuff = zeros(2,2,n+1);
A_comp(:,:,n+1) = A'*A;
c(:,n+1) = A'*y;

perm = randperm(P);
for i = 1:n
    
    c(:,i) = A(1+(i-1)*(P/n):i*P/n,:)'*y(1+(i-1)*(P/n):i*P/n);
    
    A_comp(:,:,i) = A(1+(i-1)*(P/n):i*P/n,:)'*A(1+(i-1)*(P/n):i*P/n,:);
    
    w_st_loc(:,i) = A_comp(:,:,i)\c(:,i);
    
    c_shuff(:,i) = A(perm(1+(i-1)*(P/n):i*P/n),:)'*y(perm(1+(i-1)*(P/n):i*P/n));
    
    A_comp_shuff(:,:,i) = A(perm(1+(i-1)*(P/n):i*P/n),:)'*A(perm(1+(i-1)*(P/n):i*P/n),:);
    
end

T=10;
tau = 100;
r = 0.0002;

Tot_iter = tau*T;

w_train = zeros(2,tau*T,n);
w_train_shuff = zeros(2,tau*T,n);
w_train_centr = zeros(2,tau*T);

LF_train = zeros(1,tau*T);
LF_train_shuff = zeros(1,tau*T);
LF_train_centr = zeros(1,tau*T);

for t = 2:Tot_iter
    gr = 2*A_comp(:,:,n+1)*w_train_centr(:,t-1) - 2*c(:,n+1);
    
    w_train_centr(:,t) = w_train_centr(:,t-1) - r*gr;
    if rem(t,200) == 1
                 tau = ceil(tau/2);
    end
    
        if rem(t,200)==0 && t <=300
%     if t==200
        %Shuffle
        perm = Shuffle_Data(perm,w_train_shuff,n);
%         perm = randperm(P);
        for k = 1:n
            c_shuff(:,k) = A(perm(1+(k-1)*(P/n):k*P/n),:)'*y(perm(1+(k-1)*(P/n):k*P/n));
            
            A_comp_shuff(:,:,k) = A(perm(1+(k-1)*(P/n):k*P/n),:)'*A(perm(1+(k-1)*(P/n):k*P/n),:);
        end
    end
    %Distributed Updates
    for k = 1:n
        
        if rem(t,tau)==0
            % Global Update
            gr = 2*A_comp(:,:,k)*mean(w_train(:,t-1,:),3) - 2*c(:,k);
            
            w_train(:,t,k) = mean(w_train(:,t-1,:),3) - n*r*gr;
            
            gr = 2*A_comp_shuff(:,:,k)*mean(w_train_shuff(:,t-1,:),3) - 2*c_shuff(:,k);
            
            w_train_shuff(:,t,k) = mean(w_train_shuff(:,t-1,:),3) - n*r*gr;
            
        else
            %Local Update
            gr = 2*A_comp(:,:,k)*w_train(:,t-1,k) - 2*c(:,k);
            
            w_train(:,t,k) = w_train(:,t-1,k) - n*r*gr;
            
            gr = 2*A_comp_shuff(:,:,k)*w_train_shuff(:,t-1,k) - 2*c_shuff(:,k);
            
            w_train_shuff(:,t,k) = w_train_shuff(:,t-1,k) - n*r*gr;
            
        end
    end
%     mean(w_train(:,t,:),3)
%     mean(w_train_shuff(:,t,:),3)
%     mean(w_train_centr(:,t,:),3)
    LF_train(t) = norm(A*squeeze(mean(w_train(:,t,:),3)) - y)^2;
    LF_train_shuff(t)  = norm(A*mean(w_train_shuff(:,t,:),3) - y)^2;
    LF_train_centr(t)  = norm(A*squeeze(w_train_centr(:,t)) - y)^2;
end



h=figure(2);
subplot(1,2,1)
hold on

plot(w_train_centr(1,:),w_train_centr(2,:),'k.','MarkerSize',8)
plot(mean(w_train(1,:,:),3),mean(w_train(2,:,:),3),'r.','MarkerSize',8)
plot(mean(w_train_shuff(1,:,:),3),mean(w_train_shuff(2,:,:),3),'m.','MarkerSize',8)
plot(w_star(1),w_star(2),'kO','MarkerSize',10)
plot(mean(w_st_loc(1,:)),mean(w_st_loc(2,:)),'rO','MarkerSize',10)
for i = 1:n
    plot(w_st_loc(1,i),w_st_loc(2,i),'bX','MarkerSize',10)
    %     plot(w_train(1,:,i),w_train(2,:,i),'b.','MarkerSize',3)
end
xlabel('Weight 1 (slope)')
ylabel('Weight 2 (bias)')
legend({['Centralized'],['Distributed'],['Distributed w/shuff.'],['Global min (all data)'],['Mean of global mins dist. data sets']...
    ,['Global min of dist. data sets']},'Location','southeast')



subplot(1,2,2)

hold on
plot(LF_train_centr(2:end-1),'k','LineWidth',2)
plot(LF_train(2:end-1),'r','LineWidth',2)
plot(LF_train_shuff(2:end-1),'m','LineWidth',2)
ylim([min(LF_train_centr(2:end))-0.02, min(LF_train_centr(2:end))+1.0])
ylabel('Loss Func.')
xlabel('Num. Epochs')
legend({['Centralized'],['Distributed w/shuff.'],['Distributed']})

set(h,'Position',1.0e+03 *[  0.1098    0.3234    1.2528    0.4384])


