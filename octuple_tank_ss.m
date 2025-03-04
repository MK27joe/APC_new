function [xdot]=octuple_tank_ss(x);
global g_v1;
global g_v2;
global g_v3;
global g_v4;
global g_fd1;
global g_fd2;

h1=x(1);h2=x(2);h3=x(3);h4=x(4);h5=x(5);h6=x(6);h7=x(7);h8=x(8);

A1=28;A3=28;A5=28;A7=28;              % Tank 1,3,5, 7 cross sectional area in :cm^2
A2=32;A4=32;A6=32;A8=32;              % Tank 2,4,6,8 cross sectional area in : cm^2
a1=0.071;a3=0.071;a5=0.071;a7=0.071;  % Orifice 1,3,5, 7 cross sectional area in : cm^2
a2=0.057;a4=0.057;a6=0.057;a8=0.057;  % Orifice 2,4,6, 8 cross sectional area in : cm^2
kc=0.5;                               % Output conversion factor: V/cm
g=981;                                % Acc. due to gravity cm/sec^2
k1=3.33;k2=3.35;k3=3.33;k4=3.35;      % Conversion factor : cm^2?V/Sec 
r1=0.7;r2=0.6;r3=0.65;r4=0.55;        % Fraction of split:  
beta1=0.45; beta2=0.55;               % Fraction of split for disturbance input


dh1bydt = (-a1/A1)*sqrt(2*g*h1)+(a5/A1)*sqrt(2*g*h5)+(r1*k1/A1)*g_v1 ;
dh2bydt = (-a2/A2)*sqrt(2*g*h2)+(a6/A2)*sqrt(2*g*h6)+(r2*k2/A2)*g_v2 ;
dh3bydt = (-a3/A3)*sqrt(2*g*h3)+(a7/A3)*sqrt(2*g*h7)+(r3*k3/A3)*g_v3 ;
dh4bydt = (-a4/A4)*sqrt(2*g*h4)+(a8/A4)*sqrt(2*g*h8)+(r4*k4/A4)*g_v4 ;

dh5bydt = (-a5/A5)*sqrt(2*g*h5)+((1-r4)*k4/A5)*g_v4+(beta1/A5)*g_fd1 ;
dh6bydt = (-a6/A6)*sqrt(2*g*h6)+((1-r1)*k1/A6)*g_v1+(beta2/A6)*g_fd2 ;
dh7bydt = (-a7/A7)*sqrt(2*g*h7)+((1-r2)*k2/A7)*g_v2+((1-beta1)/A7)*g_fd1 ;
dh8bydt = (-a8/A8)*sqrt(2*g*h8)+((1-r3)*k2/A8)*g_v3+((1-beta2)/A8)*g_fd2 ;

xdot=[dh1bydt; dh2bydt; dh3bydt; dh4bydt; dh5bydt; dh6bydt; dh7bydt; dh8bydt];
