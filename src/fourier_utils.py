import torch
import numpy as np

from untiled import Dataloader as DL



def which_half_space(coords, eps=1e-6):
    x = coords[..., 0]
    y = coords[..., 1]
    z = coords[..., 2]

    slab_xyz = (x < -eps)
    slab_yz = torch.logical_and(torch.logical_and(x > -eps, x < eps), y < -eps)
    slab_z = torch.logical_and(torch.logical_and(torch.logical_and(x > -eps, x < eps),
                                                 torch.logical_and(y > -eps, y < eps)), z < -eps)

    return torch.logical_or(slab_xyz, torch.logical_or(slab_yz, slab_z))

def where_DC(coords, eps=1e-6):
    x = coords[..., 0]
    y = coords[..., 1]
    z = coords[..., 2]

    slab_x = torch.logical_and(x > -eps, x < eps)
    slab_y = torch.logical_and(y > -eps, y < eps)
    slab_z = torch.logical_and(z > -eps, z < eps)

    return torch.logical_and(slab_x, torch.logical_and(slab_y, slab_z))    



def compute_ctf(nx, ny,  dfu, dfv, defocus_angle, pixel_size=2.2,volt=300, cs=2.7, w=0.07,phase_shift=0, b_factor=None):
    """
    Compute the 2D CTF

    Input:
        freqs (np.ndarray) Nx2 or BxNx2 tensor of 2D spatial frequencies
        dfu (float or Bx1 tensor): DefocusU (Angstrom)
        dfv (float or Bx1 tensor): DefocusV (Angstrom)
        defocus_angle (float or Bx1 tensor): DefocusAngle (degrees)
        volt (float or Bx1 tensor): accelerating voltage (kV)
        cs (float or Bx1 tensor): spherical aberration (mm)
        w (float or Bx1 tensor): amplitude contrast ratio
        phase_shift (float or Bx1 tensor): degrees
        b_factor (float or Bx1 tensor): envelope fcn B-factor (Angstrom^2)
    """
    fx = torch.fft.fftfreq(nx, d=pixel_size)
    fy = torch.fft.fftfreq(ny, d=pixel_size)
    x, y = torch.meshgrid(fx, fy)

    # convert units
    volt = volt * 1000
    cs = cs * 10 ** 7
    #defocus_angle = defocus_angle * np.pi / 180
    phase_shift = phase_shift * np.pi / 180
    b=dfv.size(0)
    # lam = sqrt(h^2/(2*m*e*Vr)); Vr = V + (e/(2*m*c^2))*V^2
    lam = 12.2639 / (volt + 0.97845e-6 * volt ** 2) ** .5
    # x = freqs[..., 0]
    # y = freqs[..., 1]
    ang = torch.atan2(y, x).expand(b,nx,ny)
    s2 = x ** 2 + y ** 2
    df = .5 * (dfu + dfv + (dfu - dfv) * torch.cos(2 * (ang - defocus_angle)))
    gamma = 2 * np.pi * (-.5 * df * lam * s2 + .25 * cs * lam ** 3 * s2 ** 2) - phase_shift
    ctf = (1 - w ** 2) ** .5 * torch.sin(gamma) - w * torch.cos(gamma)
    if b_factor is not None:
        ctf *= torch.exp(-b_factor / 4 * s2)
    return ctf

def torch_load(filepath,dtype=np.int32):
  return torch.from_numpy(np.array(DL.read_mrc(filepath),dtype=dtype))


def list2tensor(l):
	return torch.from_numpy(np.array(l,dtype=np.float32))

def get_ctf(ctf_params_dict,nx,ny):
  dfu=list2tensor(ctf_params_dict['dfu'])
  dfv=list2tensor(ctf_params_dict['dfv'])
  dfang=np.pi*list2tensor(ctf_params_dict['dfang'])/180.

  #print (dfu.size(),dfv.size(),dfang.size())
  volt=ctf_params_dict['volt']
  cs=ctf_params_dict['cs']
  ac=ctf_params_dict['ampcont']
  pixel_size=ctf_params_dict['apix']
  full_ctf=create_ctf(nx,ny,dfu, dfv, dfang, volt, cs, ac, pixel_size)
  return dt(full_ctf)



def create_ctf(nx, ny, dfu, dfv, dfang, volt=300, cs=2.7, ac=0.07, pixel_size=2.2,b_factor=None):
    # Calculate the Fourier space coordinates
    fx = torch.fft.fftfreq(nx, d=pixel_size)
    fy = torch.fft.fftfreq(ny, d=pixel_size)
    x, y = torch.meshgrid(fx, fy)
    try:
      b=dfv.size(0)
    except:
      #print(dfv)
      b=1

    volt = volt * 1000    # Convert kilovots to volts
    cs = cs * 10**7       # Convert spherical aberation from mm to A

    # lam = sqrt(h^2/(2*m*e*Vr)); Vr = V + (e/(2*m*c^2))*V^2

    lam = 12.2639 / np.sqrt(volt + 0.97845e-6 * volt**2)
    #lam = 12.2643247 / torch.sqrt(volt + volt**2 * 0.978466e-6)  # from cryoSPARC

    # x = freqs[:,0]
    # y = freqs[:,1]
    #ang=torch.atan2(y,x)

    ang = torch.atan2(y,x).expand(b,nx,ny).to(dfu.device)
    #print(ang.device)
    dfu=dfu.unsqueeze(-1).unsqueeze(-1)
    dfv=dfv.unsqueeze(-1).unsqueeze(-1)
    dfang=dfang.unsqueeze(-1).unsqueeze(-1)
    s2 = (x**2 + y**2).to(dfu.device)
    df = 0.5*(dfu + dfv + (dfu-dfv)*torch.cos(2*(ang-dfang)))

    gamma = 2*torch.pi*(-0.5*df*lam*s2 + 0.25*cs*lam**3*s2**2)
    ctf = np.sqrt(1-ac**2)*torch.sin(gamma) - ac*torch.cos(gamma)
    if b_factor is not None:
        ctf *= torch.exp(-b_factor / 4 * s2)
    return -torch.fft.fftshift(ctf,dim=(-2,-1))


# def create_ctf(nx, ny, dfu, dfv, dfang, volt=300, cs=2.7, ac=0.07, pixel_size=2.2,b_factor=None):
#     # Calculate the Fourier space coordinates
#     fx = torch.fft.fftfreq(nx, d=pixel_size)
#     fy = torch.fft.fftfreq(ny, d=pixel_size)
#     x, y = torch.meshgrid(fx, fy)
#     b=dfv.size(0)

#     volt = volt * 1000    # Convert kilovots to volts
#     cs = cs * 10**7       # Convert spherical aberation from mm to A

#     # lam = sqrt(h^2/(2*m*e*Vr)); Vr = V + (e/(2*m*c^2))*V^2

#     lam = 12.2639 / np.sqrt(volt + 0.97845e-6 * volt**2)
#     #lam = 12.2643247 / torch.sqrt(volt + volt**2 * 0.978466e-6)  # from cryoSPARC
    
#     # x = freqs[:,0]
#     # y = freqs[:,1]
#     #ang=torch.atan2(y,x)

#     ang = torch.atan2(y,x).expand(b,nx,ny)
#     dfu=dfu.unsqueeze(-1).unsqueeze(-1)
#     dfv=dfv.unsqueeze(-1).unsqueeze(-1)
#     dfang=dfang.unsqueeze(-1).unsqueeze(-1)
#     s2 = x**2 + y**2
#     df = 0.5*(dfu + dfv + (dfu-dfv)*torch.cos(2*(ang-dfang)))
#     gamma = 2*torch.pi*(-0.5*df*lam*s2 + 0.25*cs*lam**3*s2**2)
#     ctf = np.sqrt(1-ac**2)*torch.sin(gamma) - ac*torch.cos(gamma)
#     if b_factor is not None:
#         ctf *= torch.exp(-b_factor / 4 * s2)    
#     return -torch.fft.fftshift(ctf,dim=(-2,-1))
    #return ctf
    #return -torch.fft.fftshift(ctf,dim=(-2,-1))

def b2a(old):
  phase_shift= np.exp(-2j * np.pi * (np.arange(old.shape[0])[:,None] +  np.arange(old.shape[1])[None,:])/2 )
  return old*phase_shift

def dt(t):
    return t.detach().cpu().numpy()




def phase_shift_ft(input_fourier_image, shift,x,y):

    #print(shift.device)
    assert input_fourier_image.size(0)==shift.size(0)
    b,img_size,img_size = input_fourier_image.size()
    dx,dy=shift[:,0],shift[:,1]
    dx=dx.unsqueeze(-1).unsqueeze(-1)
    dy=dy.unsqueeze(-1).unsqueeze(-1)    
    shift_x = dx / img_size
    shift_y = dy / img_size

    x=x.expand(b,img_size,img_size)
    y=y.expand(b,img_size,img_size)    
    #print(x.device)
    phasor_shift = torch.exp(-2j * torch.pi * (shift_x *x  + shift_y * y))

    output_fourier_image = input_fourier_image * phasor_shift 
    return output_fourier_image



def ift2(t):
    return torch.fft.fftshift(torch.fft.ifftn(torch.fft.ifftshift(t,dim=(-2,-1)),dim=(-2,-1)),dim=(-2,-1)).real

def ift3(t):
    return torch.fft.fftshift(torch.fft.ifftn(torch.fft.ifftshift(t,dim=(-3,-2,-1)),dim=(-3,-2,-1)),dim=(-3,-2,-1)).real


def ft2(t):
    return torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(t,dim=(-2,-1)),dim=(-2,-1)),dim=(-2,-1))


def ft3(t):
    return torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(t,dim=(-3,-2,-1)),dim=(-3,-2,-1)),dim=(-3,-2,-1))

def shift_coords(coords, x_range, y_range, z_range, Nx, Ny, Nz, flip=False):
    """
    Shifts the coordinates and puts the DC component at (0, 0, 0).

    Parameters
    ----------
    coords: torch.tensor (..., 3)
    x_range: float
        (max_x - min_x) / 2
    y_range: float
        (max_y - min_y) / 2
    z_range: float
        (max_z - min_z) / 2
    Nx: int
    Ny: int
    Nz: int
    flip: bool
    
    Returns
    -------
    coords: torch.tensor (..., 3)
    """
    alpha = -1.
    if flip:  # "unshift" the coordinates.
        alpha = 1.

    if Nx % 2 == 0:
        x_shift = coords[..., 0] + alpha * x_range / (Nx - 1)
    else:
        x_shift = coords[..., 0]
    if Ny % 2 == 0:
        y_shift = coords[..., 1] + alpha * y_range / (Ny - 1)
    else:
        y_shift = coords[..., 1]
    if Nz % 2 == 0:
        z_shift = coords[..., 2] + alpha * z_range / (Nz - 1)
    else:
        z_shift = coords[..., 2]
    coords = torch.cat((x_shift.unsqueeze(-1),
                        y_shift.unsqueeze(-1),
                        z_shift.unsqueeze(-1)), dim=-1)
    return coords




def add_gaussian_noise_fourier_space(T, SNR):
    # Calculate the power of the signal
    signal_power = torch.sum(torch.abs(T)**2) / T.numel()
    
    # Calculate the power of the noise according to the desired SNR
    noise_power = signal_power / SNR

    # Generate Gaussian noise with the desired noise power
    noise=torch.randn_like(T) * torch.sqrt (noise_power)

    # Add the noise to the original Fourier space image
    T_noisy = T + noise

    return T_noisy