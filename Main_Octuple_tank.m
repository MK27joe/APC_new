%%## Simulation module of octuple Tank system
clear all;clc;close all;
global g_v1;
global g_v2;
global g_v3;
global g_v4;
global g_fd1;
global g_fd2;

T=5;              % Sampling time/ Integration interval
run_count=4000;   % simulation run count
v1=3;             % Nominal value of input v1
v2=3;             % Nominal value of input v2
v3=3;             % Nominal value of input v3
v4=3;             % Nominal value of input v4
fd1=2.0;          % Nominal value of disturbance 1 Fd1
fd2=2.0;          % Nominal value of disturbance 2 Fd2

g_v1=v1;  
g_v2=v2;
g_v3=v3;
g_v4=v4;
g_fd1=fd1;
g_fd2=fd2;


%%% xs=[15.59; 16.09; 13.64; 15.52; 2.97; 2.63;  2.65; 3.06]; %% Initial guess of outputs of the plant for fsolve..
xs=[15.59; 16.09; 13.64; 15.45; 2.97; 2.63;  2.65; 3.06]; %% Initial guess of outputs of the plant as per ref..

[xdot,fval]=fsolve('octuple_tank_ss',xs);                 %% to get steady state values using 'fsolve'
xs=xdot;
xp=xs;
IP= 1*idinput([run_count 4],'prbs',[0 0.04]);            %% Input generation using 'system identification toolbox'
meas_sigma= [0.5 0.5 0.5 0.5 0.25 0.25 0.25 0.25]';       %%  Measurement noise variances &%[0 0 0 0 0 0 0 0]' ;%

% Creating disturbance signals Fd1 and Fd2 
dist_fd1 = 0.5 ;dist_fd2 = 1 ;dist_sigma_Tcin = 1 ; 
randn('seed', 100); %rng(1000); % instead of 
dist_seq_fd1 =  dist_fd1 * randn(run_count, 1) ;
randn('seed', 200); %rng(2000); % instead of 
dist_seq_fd2  =  dist_fd2 * randn(run_count, 1) ;

dist_filt_fd1 = [0]; dist_filt_fd2 = [0]; 
alpha1  = [0.9]';alpha2  = [0.9]'; 
for j= 2:run_count
dist_filt_fd1(j,:) = alpha1 * dist_filt_fd1(j-1,:) + (1 - alpha1)* dist_seq_fd1 (j,:); 
dist_filt_fd2(j,:)  = alpha2 * dist_filt_fd2(j-1,:) + (1 - alpha2)* dist_seq_fd2(j,:); 
end
dk = [dist_filt_fd1 dist_filt_fd2];  %% filtred disturbance signals.

res=[];

%%  Simualtion Begins..
for k=1:run_count
    k;
% Step input variation

% if k > 100
%     g_v1= v1  + 0.5*v1;  % Input v1
%     g_v2= v2  + 0.2*v2;  % Input v2
%     g_v3= v3  + 0.2*v3;  % Input v3
%     g_v4= v4  + 0.2*v4;  % Input v4
%     g_fd1=fd1 + 0.2*fd1; % Disturbance 1 i.e Fd1  
%     g_fd2=fd2 + 0.2*fd2; % Disturbance 2 i.e Fd2
% end


% PRBS input in manuipulated variable...
    g_v1=v1+IP(k,1);   % Input v1
    g_v2=v2+IP(k,2);   % Input v2
    g_v3=v3+IP(k,3);   % Input v3
    g_v4=v4+IP(k,4);   % Input v4
    g_fd1=fd1+dk(k,1); % Disturbance 1 i.e Fd1
    g_fd2=fd2+dk(k,2); % Disturbance 2 i.e Fd2

 % Storing the data in res

   res=[res; k xp' g_v1 g_v2 g_v3 g_v4 g_fd1 g_fd2]; % Storing the data
    
   x0=xs;
   [t,x]=ode45('octuple_tank_dynamics',[0 T],x0);
   xs = x(length(t),:)';
   vk = meas_sigma.*randn(8,1); % measurement noise
   xp=xs+vk;
end
%% Simulations end...

figure(1), 
subplot(221),plot( res(:,1) , res(:,2), '-' ), grid    
ylabel('h1 (cm)');
subplot(222), plot( res(:,1) , res(:,3), '-' ), grid ;
ylabel('h2 (cm)');
subplot(223), plot( res(:,1) , res(:,4), '-' ), grid ;
xlabel('Sampling instants'), ylabel('h3 (cm)');
subplot(224), plot( res(:,1) , res(:,5), '-' ), grid ;
xlabel('Sampling instants'), ylabel('h4 (cm)') ;

% print -depsc2 '8TsimMMwMNh1t4.eps'

figure(2), 
subplot(221),plot( res(:,1) , res(:,6), '-' ), grid    
ylabel('h5 (cm)');
subplot(222), plot( res(:,1) , res(:,7), '-' ), grid ;
ylabel('h6 (cm)');
subplot(223), plot( res(:,1) , res(:,8), '-' ), grid ;
xlabel('Sampling instants'),ylabel('h7 (cm)');
subplot(224), plot( res(:,1) , res(:,9), '-' ), grid ;
xlabel('Sampling instants'), ylabel('h8 (cm)') ;

% print -depsc2 '8TsimMMwMNh5t8.eps'


figure(3),
subplot(221), plot( res(:,1) , res(:,10), '-' ), grid ;   
ylabel('pump -1 voltage (v1)'), 
subplot(222), plot( res(:,1) , res(:,11), '-' ), grid ;
ylabel('pump -2 voltage (v2)'), 
subplot(223), plot( res(:,1) , res(:,12), '-' ), grid ;
xlabel('Sampling instants'), ylabel('pump -3 voltage (v3)'), 
subplot(224), plot( res(:,1) , res(:,13), '-' ), grid ;
xlabel('Sampling instants'), ylabel('pump -4 voltage (v4)'), 

% print -depsc2 '8TsimMMwMNV1t4.eps'

figure(4),
subplot(211), plot( res(:,1) , res(:,14), '-' ), grid ;
ylabel('Disturbance-01 (Fd1)'),
subplot(212), plot( res(:,1) , res(:,15), '-' ), grid ;
xlabel('Sampling instants'), ylabel('Disturbance-02 (Fd2)'), 
% 
% print -depsc2 '8TsimMMwMNFd1n2.eps'
% 
save octuple_tank_data_26_02_25 res
   