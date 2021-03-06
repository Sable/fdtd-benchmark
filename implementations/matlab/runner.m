function runner(scale)
%%
%% Driver for 3D FDTD of a hexahedral cavity with conducting walls.
%%

% Parameter initialization.
Lx=.05; Ly=.04; Lz=.03; % Cavity dimensions in meters.
Nx=25; Ny=20; Nz=15; % Number of cells in each direction.

% Because norm isn't currently supported,
% nrm=norm([Nx/Lx Ny/Ly Nz/Lz]) is plugged in.
nrm=866.0254;

Nt=scale*200; % Number of time steps.

tic();
[Ex, Ey, Ez, Hx, Hy, Hz, Ets]=fdtd(Lx, Ly, Lz, Nx, Ny, Nz, nrm, Nt);
elapsed = toc();

ADJUST = 1000;
checksum = fletcherSum([fletcherSum(floor(ADJUST*Ex)),
    fletcherSum(floor(ADJUST*Ey)),
    fletcherSum(floor(ADJUST*Ez)),
    fletcherSum(floor(ADJUST*Hx)),
    fletcherSum(floor(ADJUST*Hy)),
    fletcherSum(floor(ADJUST*Hz))]);

disp('{');
disp('"time":');
disp(elapsed);
disp(', "output":');
disp(checksum);
disp('}');

end
