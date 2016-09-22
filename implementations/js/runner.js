var ndarray = require('ndarray');
var show = require('ndarray-show');

function runner(scale) {
    if (typeof performance === "undefined") {
       performance = Date;
    }

     var Lx=.05, Ly=.04, Lz=.03; // Cavity dimensions in meters.
     var Nx=25, Ny=20, Nz=15;    // Number of cells in each direction.

     // Because norm isn't currently supported,
     // nrm=norm([Nx/Lx Ny/Ly Nz/Lz]) is plugged in.
     var nrm=866.0254;

     var Nt=scale*200; // Number of time steps.

     // Run kernel and measure time for core computation
     var startTime = performance.now();
     for (var i = 0; i < scale; ++i) {
         var output = fdtd(Lx,Ly,Lz,Nx,Ny,Nz,nrm,Nt);
     }
     var elapsedTime = (performance.now() - startTime) / 1000;

     var ADJUST = 1000;
     var checksum = fletcher_sum_ndarray(ndarray([
         output.Ex,
         output.Ey,
         output.Ez,
         output.Hx,
         output.Hy,
         output.Hz
     ].map(function (x) {
         ops.mulseq(x, ADJUST);
         ops.flooreq(x);
         return fletcher_sum_ndarray(x);
     })));

     console.log('{' +
     '    "time": ' + elapsedTime +
     ',   "output": ' + checksum +
     '}');
}
