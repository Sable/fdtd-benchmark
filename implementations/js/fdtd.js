var ndarray = require('ndarray');
var ops = require('ndarray-ops');
var cwise = require('cwise');
var gemm = require('ndarray-gemm');
var zeros = require('zeros');

var updateH = cwise({
    args: [
        /*H*/'array', 
        /*vE10*/'array', 
        /*vE11*/'array', 
        /*vE20*/'array', 
        /*vE21*/'array', 
        /*Dt/mu0*/'scalar', 
        /*Cy*/'scalar', 
        /*Cz*/'scalar'
    ],
    body: function (h,e11, e12, e21, e22, Dt_over_mu0, C1, C2) {
        h += Dt_over_mu0 * ((e11 - e12) * C2 - (e21 - e22) * C1) 
    }
});

var updateE = cwise({
    args: [
        'array',
        'array',
        'array',
        'array',
        'array',
        'scalar',
        'scalar',
        'scalar'
    ],
    body: function (e, h11, h12, h21, h22, Dt_over_eps0, C1, C2) {
        e += Dt_over_eps0 * ((h11 - h12) * C2 + (h21 - h22) * C1)
    }
}) 

function fdtd(Lx, Ly, Lz, Nx, Ny, Nz, nrm, Nt) {
    // Physical constants.
    var eps0=8.8541878e-12; // Permittivity of vacuum.
    var mu0=4e-7*Math.PI;   // Permeability of vacuum.
    var c0=299792458;       // Speed of light in vacuum.

    var Cx=Nx/Lx, Cy=Ny/Ly, Cz=Nz/Lz; // Inverse cell dimensions.

    var Dt=1/(c0*nrm); // Time step.

    // Allocate field arrays.
    var Ex=zeros([Nx, Ny+1, Nz+1]);
    var Ey=zeros([Nx+1, Ny, Nz+1]);
    var Ez=zeros([Nx+1, Ny+1, Nz]);
    var Hx=zeros([Nx+1, Ny, Nz]);
    var Hy=zeros([Nx, Ny+1, Nz]);
    var Hz=zeros([Nx, Ny, Nz+1]);

    // Allocate time signals.
    var Ets=zeros([Nt, 3]);

    // Initialize fields (near but not on the boundary).
    Ex.set(0, 1, 1, 1);
    Ey.set(1, 0, 1, 2);
    Ez.set(1, 1, 0, 3);

    // Time stepping.
    for (var n = 1; n <= Nt; ++n) {
        // Update H everywhere.
        // Hx=Hx+(Dt/mu0)*((Ey(:, :, 2:Nz+1)-Ey(:, :, 1:Nz))*Cz-(Ez(:, 2:Ny+1, :)-Ez(:, 1:Ny, :))*Cy);
        updateH(
            Hx,
            Ey.hi(null,null,Nz+1).lo(null,null,1),
            Ey.hi(null,null,Nz).lo(null,null,0),
            Ez.hi(null,Ny+1,null).lo(null, 1, null),
            Ez.hi(null,Ny,null).lo(null, 0, null),
            (Dt/mu0),
            Cy,
            Cz
        );

        // Hy=Hy+(Dt/mu0)*((Ez(2:Nx+1, :, :)-Ez(1:Nx, :, :))*Cx-(Ex(:, :, 2:Nz+1)-Ex(:, :, 1:Nz))*Cz);
        updateH(
            Hy,
            Ez.hi(Nz+1,null,null).lo(1,null,null),
            Ez.hi(Nx,null,null).lo(0,null,null),
            Ex.hi(null,null,Nz+1).lo(null, null, 1),
            Ex.hi(null,null, Nz).lo(null, null, 0),
            (Dt/mu0),
            Cz,
            Cx
        );

        //Hz=Hz+(Dt/mu0)*((Ex(:, 2:Ny+1, :)-Ex(:, 1:Ny, :))*Cy-(Ey(2:Nx+1, :, :)-Ey(1:Nx, :, :))*Cx);
        updateH(
            Hz,
            Ex.hi(null,Ny+1,null).lo(null,1,null),
            Ex.hi(null,Ny,null).lo(null,0,null),
            Ey.hi(Nx+1,null,null).lo(1,null,null),
            Ey.hi(Nx,null,null).lo(0,null,null),
            (Dt/mu0),
            Cx,
            Cy
        );

        // Update E everywhere except on boundary.
        //Ex(:, 2:Ny, 2:Nz)=Ex(:, 2:Ny, 2:Nz)+(Dt/eps0)*((Hz(:, 2:Ny, 2:Nz)-Hz(:, 1:Ny-1, 2:Nz))*Cy-(Hy(:, 2:Ny, 2:Nz)-Hy(:, 2:Ny, 1:Nz-1))*Cz);
        updateE(
            Ex.hi(null,Ny,Nz).lo(null,1,1),
            Hz.hi(null,Ny,Nz).lo(null,1,1),
            Hz.hi(null,Ny-1,Nz).lo(null,0,1),
            Hy.hi(null,Ny,Nz).lo(null,1,1),
            Hy.hi(null,Ny,Nz-1).lo(null,1,0),
            Cz,
            Cy
        )

        //Ey(2:Nx, :, 2:Nz)=Ey(2:Nx, :, 2:Nz)+(Dt/eps0)*((Hx(2:Nx, :, 2:Nz)-Hx(2:Nx, :, 1:Nz-1))*Cz-(Hz(2:Nx, :, 2:Nz)-Hz(1:Nx-1, :, 2:Nz))*Cx);
        updateE(
            Ey.hi(Nx,null,Nz).lo(1,null,1),
            Hx.hi(Nx,null,Nz).lo(1,null,1),
            Hx.hi(Nx,null,Nz-1).lo(1,null,0),
            Hz.hi(Nx,null,Nz).lo(1,null,1),
            Hz.hi(Nx-1,null,Nz).lo(0,null,1),
            Cx,
            Cz
        ) 

        //Ez(2:Nx, 2:Ny, :)=Ez(2:Nx, 2:Ny, :)+(Dt/eps0)*((Hy(2:Nx, 2:Ny, :)-Hy(1:Nx-1, 2:Ny, :))*Cx-(Hx(2:Nx, 2:Ny, :)-Hx(2:Nx, 1:Ny-1, :))*Cy);
        updateE(
            Ez.hi(Nx,Ny,null).lo(1,1,null),
            Hy.hi(Nx,Ny,null).lo(1,1,null),
            Hy.hi(Nx-1,Ny,null).lo(0,1,null),
            Hx.hi(Nx,Ny,null).lo(1,1,null),
            Hx.hi(Nx,Ny-1,null).lo(1,0,null),
            Cy,
            Cx
        )

        // Sample the electric field at chosen points.
        Ets.set(n-1,0,Ex.get(3,3,3))
        Ets.set(n-1,1,Ey.get(3,3,3))
        Ets.set(n-1,2,Ez.get(3,3,3)) 
    }

    return { Ex:Ex, Ey:Ey, Ez:Ez, Hx:Hx, Hy:Hy, Hz:Hz, Ets:Ets };
}
