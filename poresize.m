%Dmatrix = 3e-12 ;%diffusion coefficient measured m/s^2
%Dwater = 4.0621175e-11; %diffusion conefficient in water m/s^2
Dmatrix = 2e-13;
Dwater = 4.06e-11;
rp = 6; %hydrodynamic radius of dye in nm

ps = fzero(@(x) fun(x,Dmatrix,rp,Dwater),60) %ps is pore size in nm
fplot(@(x) fun(x,Dmatrix,rp,Dwater))
axis([10 18 -0.1 0.1])


function y =fun(x,Dmatrix,rp,Dwater)
y = Dmatrix/Dwater-(1-2*rp/x)^2*(1-2.104*(2*rp/x)+2.09*(2*rp/x)^2-0.95*(2*rp/x)^3);
end
