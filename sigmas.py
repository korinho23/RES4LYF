import torch
import numpy as np
from math import *
import builtins
from scipy.interpolate import CubicSpline
from scipy import special, stats
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import math


from comfy.k_diffusion.sampling import get_sigmas_polyexponential, get_sigmas_karras
import comfy.samplers

from torch import Tensor, nn
from typing import Optional, Callable, Tuple, Dict, Any, Union, TYPE_CHECKING, TypeVar

from .res4lyf import RESplain
from .helper  import get_res4lyf_scheduler_list


def rescale_linear(input, input_min, input_max, output_min, output_max):
    output = ((input - input_min) / (input_max - input_min)) * (output_max - output_min) + output_min;
    return output

class set_precision_sigmas:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                    "sigmas": ("SIGMAS", ),   
                    "precision": (["16", "32", "64"], ),
                    "set_default": ("BOOLEAN", {"default": False})
                     },
                }

    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("passthrough",)
    CATEGORY = "RES4LYF/precision"

    FUNCTION = "main"

    def main(self, precision="32", sigmas=None, set_default=False):
        match precision:
            case "16":
                if set_default is True:
                    torch.set_default_dtype(torch.float16)
                sigmas = sigmas.to(torch.float16)
            case "32":
                if set_default is True:
                    torch.set_default_dtype(torch.float32)
                sigmas = sigmas.to(torch.float32)
            case "64":
                if set_default is True:
                    torch.set_default_dtype(torch.float64)
                sigmas = sigmas.to(torch.float64)
        return (sigmas, )


class SimpleInterpolator(nn.Module):
    def __init__(self):
        super(SimpleInterpolator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

def train_interpolator(model, sigma_schedule, steps, epochs=5000, lr=0.01):
    with torch.inference_mode(False):
        model = SimpleInterpolator()
        sigma_schedule = sigma_schedule.clone()

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        x_train = torch.linspace(0, 1, steps=steps).unsqueeze(1)
        y_train = sigma_schedule.unsqueeze(1)

        # disable inference mode for training
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()

            # fwd pass
            outputs = model(x_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()

    return model

def interpolate_sigma_schedule_model(sigma_schedule, target_steps):
    model = SimpleInterpolator()
    sigma_schedule = sigma_schedule.float().detach()

    # train on original sigma schedule
    trained_model = train_interpolator(model, sigma_schedule, len(sigma_schedule))

    # generate target steps for interpolation
    x_interpolated = torch.linspace(0, 1, target_steps).unsqueeze(1)

    # inference w/o gradients
    trained_model.eval()
    with torch.no_grad():
        interpolated_sigma = trained_model(x_interpolated).squeeze()

    return interpolated_sigma




class sigmas_interpolate:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas_0": ("SIGMAS", {"forceInput": True}),
                "sigmas_1": ("SIGMAS", {"forceInput": True}),
                "mode": (["linear", "nearest", "polynomial", "exponential", "power", "model"],),
                "order": ("INT", {"default": 8, "min": 1,"max": 64,"step": 1}),
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS","SIGMAS",)
    RETURN_NAMES = ("sigmas_0", "sigmas_1")
    CATEGORY = "RES4LYF/sigmas"
    



    def interpolate_sigma_schedule_poly(self, sigma_schedule, target_steps):
        order = self.order
        sigma_schedule_np = sigma_schedule.cpu().numpy()

        # orig steps (assuming even spacing)
        original_steps = np.linspace(0, 1, len(sigma_schedule_np))

        # fit polynomial of the given order
        coefficients = np.polyfit(original_steps, sigma_schedule_np, deg=order)

        # generate new steps where we want to interpolate the data
        target_steps_np = np.linspace(0, 1, target_steps)

        # eval polynomial at new steps
        interpolated_sigma_np = np.polyval(coefficients, target_steps_np)

        interpolated_sigma = torch.tensor(interpolated_sigma_np, device=sigma_schedule.device, dtype=sigma_schedule.dtype)
        return interpolated_sigma

    def interpolate_sigma_schedule_constrained(self, sigma_schedule, target_steps):
        sigma_schedule_np = sigma_schedule.cpu().numpy()

        # orig steps
        original_steps = np.linspace(0, 1, len(sigma_schedule_np))

        # target steps for interpolation
        target_steps_np = np.linspace(0, 1, target_steps)

        # fit cubic spline with fixed start and end values
        cs = CubicSpline(original_steps, sigma_schedule_np, bc_type=((1, 0.0), (1, 0.0)))

        # eval spline at the target steps
        interpolated_sigma_np = cs(target_steps_np)

        interpolated_sigma = torch.tensor(interpolated_sigma_np, device=sigma_schedule.device, dtype=sigma_schedule.dtype)

        return interpolated_sigma
    
    def interpolate_sigma_schedule_exp(self, sigma_schedule, target_steps):
        # transform to log space
        log_sigma_schedule = torch.log(sigma_schedule)

        # define the original and target step ranges
        original_steps = torch.linspace(0, 1, steps=len(sigma_schedule))
        target_steps = torch.linspace(0, 1, steps=target_steps)

        # interpolate in log space
        interpolated_log_sigma = F.interpolate(
            log_sigma_schedule.unsqueeze(0).unsqueeze(0),  # Add fake batch and channel dimensions
            size=target_steps.shape[0],
            mode='linear',
            align_corners=True
        ).squeeze()

        # transform back to exponential space
        interpolated_sigma_schedule = torch.exp(interpolated_log_sigma)

        return interpolated_sigma_schedule
    
    def interpolate_sigma_schedule_power(self, sigma_schedule, target_steps):
        sigma_schedule_np = sigma_schedule.cpu().numpy()
        original_steps = np.linspace(1, len(sigma_schedule_np), len(sigma_schedule_np))

        # power regression using a log-log transformation
        log_x = np.log(original_steps)
        log_y = np.log(sigma_schedule_np)

        # linear regression on log-log data
        coefficients = np.polyfit(log_x, log_y, deg=1)  # degree 1 for linear fit in log-log space
        a = np.exp(coefficients[1])  # a = "b" = intercept (exp because of the log transform)
        b = coefficients[0]  # b = "m" = slope

        target_steps_np = np.linspace(1, len(sigma_schedule_np), target_steps)

        # power law prediction: y = a * x^b
        interpolated_sigma_np = a * (target_steps_np ** b)

        interpolated_sigma = torch.tensor(interpolated_sigma_np, device=sigma_schedule.device, dtype=sigma_schedule.dtype)

        return interpolated_sigma
            
    def interpolate_sigma_schedule_linear(self, sigma_schedule, target_steps):
        return F.interpolate(sigma_schedule.unsqueeze(0).unsqueeze(0), target_steps, mode='linear').squeeze(0).squeeze(0)

    def interpolate_sigma_schedule_nearest(self, sigma_schedule, target_steps):
        return F.interpolate(sigma_schedule.unsqueeze(0).unsqueeze(0), target_steps, mode='nearest').squeeze(0).squeeze(0)    
    
    def interpolate_nearest_neighbor(self, sigma_schedule, target_steps):
        original_steps = torch.linspace(0, 1, steps=len(sigma_schedule))
        target_steps = torch.linspace(0, 1, steps=target_steps)

        # interpolate original -> target steps using nearest neighbor
        indices = torch.searchsorted(original_steps, target_steps)
        indices = torch.clamp(indices, 0, len(sigma_schedule) - 1)  # clamp indices to valid range

        # set nearest neighbor via indices
        interpolated_sigma = sigma_schedule[indices]

        return interpolated_sigma


    def main(self, sigmas_0, sigmas_1, mode, order):

        self.order = order

        if   mode == "linear": 
            interpolate = self.interpolate_sigma_schedule_linear
        if   mode == "nearest": 
            interpolate = self.interpolate_nearest_neighbor
        elif mode == "polynomial":
            interpolate = self.interpolate_sigma_schedule_poly
        elif mode == "exponential":
            interpolate = self.interpolate_sigma_schedule_exp
        elif mode == "power":
            interpolate = self.interpolate_sigma_schedule_power
        elif mode == "model":
            with torch.inference_mode(False):
                interpolate = interpolate_sigma_schedule_model
        
        sigmas_0 = interpolate(sigmas_0, len(sigmas_1))
        return (sigmas_0, sigmas_1,)
    
class sigmas_noise_inversion:
    # flip sigmas for unsampling, and pad both fwd/rev directions with null bytes to disable noise scaling, etc from the model.
    # will cause model to return epsilon prediction instead of calculated denoised latent image.
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS","SIGMAS",)
    RETURN_NAMES = ("sigmas_fwd","sigmas_rev",)
    CATEGORY = "RES4LYF/sigmas"
    DESCRIPTION = "For use with unsampling. Connect sigmas_fwd to the unsampling (first) node, and sigmas_rev to the sampling (second) node."
    
    def main(self, sigmas):
        sigmas = sigmas.clone().to(torch.float64)
        
        null = torch.tensor([0.0], device=sigmas.device, dtype=sigmas.dtype)
        sigmas_fwd = torch.flip(sigmas, dims=[0])
        sigmas_fwd = torch.cat([sigmas_fwd, null])
        
        sigmas_rev = torch.cat([null, sigmas])
        sigmas_rev = torch.cat([sigmas_rev, null])
        
        return (sigmas_fwd, sigmas_rev,)


def compute_sigma_next_variance_floor(sigma):
    return (-1 + torch.sqrt(1 + 4 * sigma)) / 2

class sigmas_variance_floor:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    DESCRIPTION = ("Process a sigma schedule so that any steps that are too large for variance-locked SDE sampling are replaced with the maximum permissible value."
        "Will be very difficult to approach sigma = 0 due to the nature of the math, as steps become very small much below approximately sigma = 0.15 to 0.2.")
    
    def main(self, sigmas):
        dtype = sigmas.dtype
        sigmas = sigmas.clone().to(torch.float64)
        for i in range(len(sigmas) - 1):
            sigma_next = (-1 + torch.sqrt(1 + 4 * sigmas[i])) / 2
            
            if sigmas[i+1] < sigma_next and sigmas[i+1] > 0.0:
                print("swapped i+1 with sigma_next+0.001: ", sigmas[i+1], sigma_next + 0.001)
                sigmas[i+1] = sigma_next + 0.001
        return (sigmas.to(dtype),)


class sigmas_from_text:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("sigmas",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, text):
        text_list = [float(val) for val in text.replace(",", " ").split()]
        #text_list = [float(val.strip()) for val in text.split(",")]

        sigmas = torch.tensor(text_list).to('cuda').to(torch.float64)
        
        return (sigmas,)



class sigmas_concatenate:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas_1": ("SIGMAS", {"forceInput": True}),
                "sigmas_2": ("SIGMAS", {"forceInput": True}),
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, sigmas_1, sigmas_2):
        return (torch.cat((sigmas_1, sigmas_2)),)

class sigmas_truncate:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "sigmas_until": ("INT", {"default": 10, "min": 0,"max": 1000,"step": 1}),
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, sigmas, sigmas_until):
        return (sigmas[:sigmas_until],)

class sigmas_start:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "sigmas_until": ("INT", {"default": 10, "min": 0,"max": 1000,"step": 1}),
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, sigmas, sigmas_until):
        return (sigmas[sigmas_until:],)
        
class sigmas_split:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "sigmas_start": ("INT", {"default": 0, "min": 0,"max": 1000,"step": 1}),
                "sigmas_end": ("INT", {"default": 1000, "min": 0,"max": 1000,"step": 1}),
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, sigmas, sigmas_start, sigmas_end):
        return (sigmas[sigmas_start:sigmas_end],)

        sigmas_stop_step = sigmas_end - sigmas_start
        return (sigmas[sigmas_start:][:sigmas_stop_step],)
    
class sigmas_pad:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "value": ("FLOAT", {"default": 0.0, "min": -10000,"max": 10000,"step": 0.01})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, sigmas, value):
        return (torch.cat((sigmas, torch.tensor([value], dtype=sigmas.dtype))),)
    
class sigmas_unpad:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, sigmas):
        return (sigmas[:-1],)

class sigmas_set_floor:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "floor": ("FLOAT", {"default": 0.0291675, "min": -10000,"max": 10000,"step": 0.01}),
                "new_floor": ("FLOAT", {"default": 0.0291675, "min": -10000,"max": 10000,"step": 0.01})
            }
        }

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "set_floor"

    CATEGORY = "RES4LYF/sigmas"

    def set_floor(self, sigmas, floor, new_floor):
        sigmas[sigmas <= floor] = new_floor
        return (sigmas,)    
    
class sigmas_delete_below_floor:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "floor": ("FLOAT", {"default": 0.0291675, "min": -10000,"max": 10000,"step": 0.01})
            }
        }

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "delete_below_floor"

    CATEGORY = "RES4LYF/sigmas"

    def delete_below_floor(self, sigmas, floor):
        return (sigmas[sigmas >= floor],)    

class sigmas_delete_value:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "value": ("FLOAT", {"default": 0.0, "min": -1000,"max": 1000,"step": 0.01})
            }
        }

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "delete_value"

    CATEGORY = "RES4LYF/sigmas"

    def delete_value(self, sigmas, value):
        return (sigmas[sigmas != value],) 

class sigmas_delete_consecutive_duplicates:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas_1": ("SIGMAS", {"forceInput": True})
            }
        }

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "delete_consecutive_duplicates"

    CATEGORY = "RES4LYF/sigmas"

    def delete_consecutive_duplicates(self, sigmas_1):
        mask = sigmas_1[:-1] != sigmas_1[1:]
        mask = torch.cat((mask, torch.tensor([True])))
        return (sigmas_1[mask],) 

class sigmas_cleanup:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "sigmin": ("FLOAT", {"default": 0.0291675, "min": 0,"max": 1000,"step": 0.01})
            }
        }

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "cleanup"

    CATEGORY = "RES4LYF/sigmas"

    def cleanup(self, sigmas, sigmin):
        sigmas_culled = sigmas[sigmas >= sigmin]
    
        mask = sigmas_culled[:-1] != sigmas_culled[1:]
        mask = torch.cat((mask, torch.tensor([True])))
        filtered_sigmas = sigmas_culled[mask]
        return (torch.cat((filtered_sigmas,torch.tensor([0]))),)

class sigmas_mult:
    def __init__(self):
        pass   

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "multiplier": ("FLOAT", {"default": 1, "min": -10000,"max": 10000,"step": 0.01})
            },
            "optional": {
                "sigmas2": ("SIGMAS", {"forceInput": False})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, sigmas, multiplier, sigmas2=None):
        if sigmas2 is not None:
            return (sigmas * sigmas2 * multiplier,)
        else:
            return (sigmas * multiplier,)    

class sigmas_modulus:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "divisor": ("FLOAT", {"default": 1, "min": -1000,"max": 1000,"step": 0.01})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, sigmas, divisor):
        return (sigmas % divisor,)
        
class sigmas_quotient:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "divisor": ("FLOAT", {"default": 1, "min": -1000,"max": 1000,"step": 0.01})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, sigmas, divisor):
        return (sigmas // divisor,)

class sigmas_add:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "addend": ("FLOAT", {"default": 1, "min": -1000,"max": 1000,"step": 0.01})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, sigmas, addend):
        return (sigmas + addend,)

class sigmas_power:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "power": ("FLOAT", {"default": 1, "min": -100,"max": 100,"step": 0.01})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, sigmas, power):
        return (sigmas ** power,)

class sigmas_abs:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, sigmas):
        return (abs(sigmas),)

class sigmas2_mult:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas_1": ("SIGMAS", {"forceInput": True}),
                "sigmas_2": ("SIGMAS", {"forceInput": True}),
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, sigmas_1, sigmas_2):
        return (sigmas_1 * sigmas_2,)

class sigmas2_add:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas_1": ("SIGMAS", {"forceInput": True}),
                "sigmas_2": ("SIGMAS", {"forceInput": True}),
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, sigmas_1, sigmas_2):
        return (sigmas_1 + sigmas_2,)

class sigmas_rescale:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "start": ("FLOAT", {"default": 1.0, "min": -10000,"max": 10000,"step": 0.01}),
                "end": ("FLOAT", {"default": 0.0, "min": -10000,"max": 10000,"step": 0.01}),
                "sigmas": ("SIGMAS", ),
            },
            "optional": {
            }
        }
    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("sigmas_rescaled",)
    CATEGORY = "RES4LYF/sigmas"
    DESCRIPTION = ("Can be used to set denoise. Results are generally better than with the approach used by KSampler and most nodes with denoise values "
                   "(which slice the sigmas schedule according to step count, not the noise level). Will also flip the sigma schedule if the start and end values are reversed." 
                   )
      
    def main(self, start=0, end=-1, sigmas=None):

        s_out_1 = ((sigmas - sigmas.min()) * (start - end)) / (sigmas.max() - sigmas.min()) + end     
        
        return (s_out_1,)


class sigmas_math1:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "start": ("INT", {"default": 0, "min": 0,"max": 10000,"step": 1}),
                "stop": ("INT", {"default": 0, "min": 0,"max": 10000,"step": 1}),
                "trim": ("INT", {"default": 0, "min": -10000,"max": 0,"step": 1}),
                "x": ("FLOAT", {"default": 1, "min": -10000,"max": 10000,"step": 0.01}),
                "y": ("FLOAT", {"default": 1, "min": -10000,"max": 10000,"step": 0.01}),
                "z": ("FLOAT", {"default": 1, "min": -10000,"max": 10000,"step": 0.01}),
                "f1": ("STRING", {"default": "s", "multiline": True}),
                "rescale" : ("BOOLEAN", {"default": False}),
                "max1": ("FLOAT", {"default": 14.614642, "min": -10000,"max": 10000,"step": 0.01}),
                "min1": ("FLOAT", {"default": 0.0291675, "min": -10000,"max": 10000,"step": 0.01}),
            },
            "optional": {
                "a": ("SIGMAS", {"forceInput": False}),
                "b": ("SIGMAS", {"forceInput": False}),               
                "c": ("SIGMAS", {"forceInput": False}),
            }
        }
    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    def main(self, start=0, stop=0, trim=0, a=None, b=None, c=None, x=1.0, y=1.0, z=1.0, f1="s", rescale=False, min1=1.0, max1=1.0):
        if stop == 0:
            t_lens = [len(tensor) for tensor in [a, b, c] if tensor is not None]
            t_len = stop = min(t_lens) if t_lens else 0
        else:
            stop = stop + 1
            t_len = stop - start 
            
        stop = stop + trim
        t_len = t_len + trim
        
        t_a = t_b = t_c = None
        if a is not None:
            t_a = a[start:stop]
        if b is not None:
            t_b = b[start:stop]
        if c is not None:
            t_c = c[start:stop]               
            
        t_s = torch.arange(0.0, t_len)
    
        t_x = torch.full((t_len,), x)
        t_y = torch.full((t_len,), y)
        t_z = torch.full((t_len,), z)
        eval_namespace = {"__builtins__": None, "round": builtins.round, "np": np, "a": t_a, "b": t_b, "c": t_c, "x": t_x, "y": t_y, "z": t_z, "s": t_s, "torch": torch}
        eval_namespace.update(np.__dict__)
        
        s_out_1 = eval(f1, eval_namespace)
        
        if rescale == True:
            s_out_1 = ((s_out_1 - min(s_out_1)) * (max1 - min1)) / (max(s_out_1) - min(s_out_1)) + min1     
        
        return (s_out_1,)

class sigmas_math3:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "start": ("INT", {"default": 0, "min": 0,"max": 10000,"step": 1}),
                "stop": ("INT", {"default": 0, "min": 0,"max": 10000,"step": 1}),
                "trim": ("INT", {"default": 0, "min": -10000,"max": 0,"step": 1}),
            },
            "optional": {
                "a": ("SIGMAS", {"forceInput": False}),
                "b": ("SIGMAS", {"forceInput": False}),               
                "c": ("SIGMAS", {"forceInput": False}),
                "x": ("FLOAT", {"default": 1, "min": -10000,"max": 10000,"step": 0.01}),
                "y": ("FLOAT", {"default": 1, "min": -10000,"max": 10000,"step": 0.01}),
                "z": ("FLOAT", {"default": 1, "min": -10000,"max": 10000,"step": 0.01}),
                "f1": ("STRING", {"default": "s", "multiline": True}),
                "rescale1" : ("BOOLEAN", {"default": False}),
                "max1": ("FLOAT", {"default": 14.614642, "min": -10000,"max": 10000,"step": 0.01}),
                "min1": ("FLOAT", {"default": 0.0291675, "min": -10000,"max": 10000,"step": 0.01}),
                "f2": ("STRING", {"default": "s", "multiline": True}),
                "rescale2" : ("BOOLEAN", {"default": False}),
                "max2": ("FLOAT", {"default": 14.614642, "min": -10000,"max": 10000,"step": 0.01}),
                "min2": ("FLOAT", {"default": 0.0291675, "min": -10000,"max": 10000,"step": 0.01}),
                "f3": ("STRING", {"default": "s", "multiline": True}),
                "rescale3" : ("BOOLEAN", {"default": False}),
                "max3": ("FLOAT", {"default": 14.614642, "min": -10000,"max": 10000,"step": 0.01}),
                "min3": ("FLOAT", {"default": 0.0291675, "min": -10000,"max": 10000,"step": 0.01}),
            }
        }
    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS","SIGMAS","SIGMAS")
    CATEGORY = "RES4LYF/sigmas"
    def main(self, start=0, stop=0, trim=0, a=None, b=None, c=None, x=1.0, y=1.0, z=1.0, f1="s", f2="s", f3="s", rescale1=False, rescale2=False, rescale3=False, min1=1.0, max1=1.0, min2=1.0, max2=1.0, min3=1.0, max3=1.0):
        if stop == 0:
            t_lens = [len(tensor) for tensor in [a, b, c] if tensor is not None]
            t_len = stop = min(t_lens) if t_lens else 0
        else:
            stop = stop + 1
            t_len = stop - start 
            
        stop = stop + trim
        t_len = t_len + trim
        
        t_a = t_b = t_c = None
        if a is not None:
            t_a = a[start:stop]
        if b is not None:
            t_b = b[start:stop]
        if c is not None:
            t_c = c[start:stop]               
            
        t_s = torch.arange(0.0, t_len)
    
        t_x = torch.full((t_len,), x)
        t_y = torch.full((t_len,), y)
        t_z = torch.full((t_len,), z)
        eval_namespace = {"__builtins__": None, "np": np, "a": t_a, "b": t_b, "c": t_c, "x": t_x, "y": t_y, "z": t_z, "s": t_s, "torch": torch}
        eval_namespace.update(np.__dict__)
        
        s_out_1 = eval(f1, eval_namespace)
        s_out_2 = eval(f2, eval_namespace)
        s_out_3 = eval(f3, eval_namespace)
        
        if rescale1 == True:
            s_out_1 = ((s_out_1 - min(s_out_1)) * (max1 - min1)) / (max(s_out_1) - min(s_out_1)) + min1
        if rescale2 == True:
            s_out_2 = ((s_out_2 - min(s_out_2)) * (max2 - min2)) / (max(s_out_2) - min(s_out_2)) + min2
        if rescale3 == True:
            s_out_3 = ((s_out_3 - min(s_out_3)) * (max3 - min3)) / (max(s_out_3) - min(s_out_3)) + min3        
        
        return s_out_1, s_out_2, s_out_3

class sigmas_iteration_karras:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "steps_up": ("INT", {"default": 30, "min": 0,"max": 10000,"step": 1}),
                "steps_down": ("INT", {"default": 30, "min": 0,"max": 10000,"step": 1}),
                "rho_up": ("FLOAT", {"default": 3, "min": -10000,"max": 10000,"step": 0.01}),
                "rho_down": ("FLOAT", {"default": 4, "min": -10000,"max": 10000,"step": 0.01}),
                "s_min_start": ("FLOAT", {"default":0.0291675, "min": -10000,"max": 10000,"step": 0.01}),
                "s_max": ("FLOAT", {"default": 2, "min": -10000,"max": 10000,"step": 0.01}),
                "s_min_end": ("FLOAT", {"default": 0.0291675, "min": -10000,"max": 10000,"step": 0.01}),
            },
            "optional": {
                "momentums": ("SIGMAS", {"forceInput": False}),
                "sigmas": ("SIGMAS", {"forceInput": False}),             
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS","SIGMAS")
    RETURN_NAMES = ("momentums","sigmas")
    CATEGORY = "RES4LYF/schedulers"
    
    def main(self, steps_up, steps_down, rho_up, rho_down, s_min_start, s_max, s_min_end, sigmas=None, momentums=None):
        s_up = get_sigmas_karras(steps_up, s_min_start, s_max, rho_up)
        s_down = get_sigmas_karras(steps_down, s_min_end, s_max, rho_down) 
        s_up = s_up[:-1]
        s_down = s_down[:-1]  
        s_up = torch.flip(s_up, dims=[0])
        sigmas_new = torch.cat((s_up, s_down), dim=0)
        momentums_new = torch.cat((s_up, -1*s_down), dim=0)
        
        if sigmas is not None:
            sigmas = torch.cat([sigmas, sigmas_new])
        else:
            sigmas = sigmas_new
            
        if momentums is not None:
            momentums = torch.cat([momentums, momentums_new])
        else:
            momentums = momentums_new
        
        return (momentums,sigmas) 
 
class sigmas_iteration_polyexp:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "steps_up": ("INT", {"default": 30, "min": 0,"max": 10000,"step": 1}),
                "steps_down": ("INT", {"default": 30, "min": 0,"max": 10000,"step": 1}),
                "rho_up": ("FLOAT", {"default": 0.6, "min": -10000,"max": 10000,"step": 0.01}),
                "rho_down": ("FLOAT", {"default": 0.8, "min": -10000,"max": 10000,"step": 0.01}),
                "s_min_start": ("FLOAT", {"default":0.0291675, "min": -10000,"max": 10000,"step": 0.01}),
                "s_max": ("FLOAT", {"default": 2, "min": -10000,"max": 10000,"step": 0.01}),
                "s_min_end": ("FLOAT", {"default": 0.0291675, "min": -10000,"max": 10000,"step": 0.01}),
            },
            "optional": {
                "momentums": ("SIGMAS", {"forceInput": False}),
                "sigmas": ("SIGMAS", {"forceInput": False}),             
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS","SIGMAS")
    RETURN_NAMES = ("momentums","sigmas")
    CATEGORY = "RES4LYF/schedulers"
    
    def main(self, steps_up, steps_down, rho_up, rho_down, s_min_start, s_max, s_min_end, sigmas=None, momentums=None):
        s_up = get_sigmas_polyexponential(steps_up, s_min_start, s_max, rho_up)
        s_down = get_sigmas_polyexponential(steps_down, s_min_end, s_max, rho_down) 
        s_up = s_up[:-1]
        s_down = s_down[:-1]
        s_up = torch.flip(s_up, dims=[0])
        sigmas_new = torch.cat((s_up, s_down), dim=0)
        momentums_new = torch.cat((s_up, -1*s_down), dim=0)

        if sigmas is not None:
            sigmas = torch.cat([sigmas, sigmas_new])
        else:
            sigmas = sigmas_new

        if momentums is not None:
            momentums = torch.cat([momentums, momentums_new])
        else:
            momentums = momentums_new

        return (momentums,sigmas) 

class tan_scheduler:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "steps": ("INT", {"default": 20, "min": 0,"max": 100000,"step": 1}),
                "offset": ("FLOAT", {"default": 20, "min": 0,"max": 100000,"step": 0.1}),
                "slope": ("FLOAT", {"default": 20, "min": -100000,"max": 100000,"step": 0.1}),
                "start": ("FLOAT", {"default": 20, "min": -100000,"max": 100000,"step": 0.1}),
                "end": ("FLOAT", {"default": 20, "min": -100000,"max": 100000,"step": 0.1}),
                "sgm" : ("BOOLEAN", {"default": False}),
                "pad" : ("BOOLEAN", {"default": False}),
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/schedulers"
    
    def main(self, steps, slope, offset, start, end, sgm, pad):
        smax = ((2/pi)*atan(-slope*(0-offset))+1)/2
        smin = ((2/pi)*atan(-slope*((steps-1)-offset))+1)/2

        srange = smax-smin
        sscale = start - end
        
        if sgm:
            steps+=1

        sigmas = [  ( (((2/pi)*atan(-slope*(x-offset))+1)/2) - smin) * (1/srange) * sscale + end    for x in range(steps)]
        
        if sgm:
            sigmas = sigmas[:-1]
        if pad:
            sigmas = torch.tensor(sigmas+[0])
        else:
            sigmas = torch.tensor(sigmas)
        return (sigmas,)

class tan_scheduler_2stage:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "steps": ("INT", {"default": 40, "min": 0,"max": 100000,"step": 1}),
                "midpoint": ("INT", {"default": 20, "min": 0,"max": 100000,"step": 1}),
                "pivot_1": ("INT", {"default": 10, "min": 0,"max": 100000,"step": 1}),
                "pivot_2": ("INT", {"default": 30, "min": 0,"max": 100000,"step": 1}),
                "slope_1": ("FLOAT", {"default": 1, "min": -100000,"max": 100000,"step": 0.1}),
                "slope_2": ("FLOAT", {"default": 1, "min": -100000,"max": 100000,"step": 0.1}),
                "start": ("FLOAT", {"default": 1.0, "min": -100000,"max": 100000,"step": 0.1}),
                "middle": ("FLOAT", {"default": 0.5, "min": -100000,"max": 100000,"step": 0.1}),
                "end": ("FLOAT", {"default": 0.0, "min": -100000,"max": 100000,"step": 0.1}),
                "pad" : ("BOOLEAN", {"default": False}),
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("sigmas",)
    CATEGORY = "RES4LYF/schedulers"

    def get_tan_sigmas(self, steps, slope, pivot, start, end):
        smax = ((2/pi)*atan(-slope*(0-pivot))+1)/2
        smin = ((2/pi)*atan(-slope*((steps-1)-pivot))+1)/2

        srange = smax-smin
        sscale = start - end

        sigmas = [  ( (((2/pi)*atan(-slope*(x-pivot))+1)/2) - smin) * (1/srange) * sscale + end    for x in range(steps)]
        
        return sigmas

    def main(self, steps, midpoint, start, middle, end, pivot_1, pivot_2, slope_1, slope_2, pad):
        steps += 2
        stage_2_len = steps - midpoint
        stage_1_len = steps - stage_2_len

        tan_sigmas_1 = self.get_tan_sigmas(stage_1_len, slope_1, pivot_1, start, middle)
        tan_sigmas_2 = self.get_tan_sigmas(stage_2_len, slope_2, pivot_2 - stage_1_len, middle, end)
        
        tan_sigmas_1 = tan_sigmas_1[:-1]
        if pad:
            tan_sigmas_2 = tan_sigmas_2+[0]

        tan_sigmas = torch.tensor(tan_sigmas_1 + tan_sigmas_2)

        return (tan_sigmas,)

class tan_scheduler_2stage_simple:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "steps": ("INT", {"default": 40, "min": 0,"max": 100000,"step": 1}),
                "pivot_1": ("FLOAT", {"default": 1, "min": -100000,"max": 100000,"step": 0.01}),
                "pivot_2": ("FLOAT", {"default": 1, "min": -100000,"max": 100000,"step": 0.01}),
                "slope_1": ("FLOAT", {"default": 1, "min": -100000,"max": 100000,"step": 0.01}),
                "slope_2": ("FLOAT", {"default": 1, "min": -100000,"max": 100000,"step": 0.01}),
                "start": ("FLOAT", {"default": 1.0, "min": -100000,"max": 100000,"step": 0.01}),
                "middle": ("FLOAT", {"default": 0.5, "min": -100000,"max": 100000,"step": 0.01}),
                "end": ("FLOAT", {"default": 0.0, "min": -100000,"max": 100000,"step": 0.01}),
                "pad" : ("BOOLEAN", {"default": False}),
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("sigmas",)
    CATEGORY = "RES4LYF/schedulers"

    def get_tan_sigmas(self, steps, slope, pivot, start, end):
        smax = ((2/pi)*atan(-slope*(0-pivot))+1)/2
        smin = ((2/pi)*atan(-slope*((steps-1)-pivot))+1)/2

        srange = smax-smin
        sscale = start - end

        sigmas = [  ( (((2/pi)*atan(-slope*(x-pivot))+1)/2) - smin) * (1/srange) * sscale + end    for x in range(steps)]
        
        return sigmas

    def main(self, steps, start, middle, end, pivot_1, pivot_2, slope_1, slope_2, pad):
        steps += 2

        midpoint = int( (steps*pivot_1 + steps*pivot_2) / 2 )
        pivot_1 = int(steps * pivot_1)
        pivot_2 = int(steps * pivot_2)

        slope_1 = slope_1 / (steps/40)
        slope_2 = slope_2 / (steps/40)

        stage_2_len = steps - midpoint
        stage_1_len = steps - stage_2_len

        tan_sigmas_1 = self.get_tan_sigmas(stage_1_len, slope_1, pivot_1, start, middle)
        tan_sigmas_2 = self.get_tan_sigmas(stage_2_len, slope_2, pivot_2 - stage_1_len, middle, end)
        
        tan_sigmas_1 = tan_sigmas_1[:-1]
        if pad:
            tan_sigmas_2 = tan_sigmas_2+[0]

        tan_sigmas = torch.tensor(tan_sigmas_1 + tan_sigmas_2)

        return (tan_sigmas,)
    
class linear_quadratic_advanced:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "steps": ("INT", {"default": 40, "min": 0,"max": 100000,"step": 1}),
                "denoise": ("FLOAT", {"default": 1.0, "min": -100000,"max": 100000,"step": 0.01}),
                "inflection_percent": ("FLOAT", {"default": 0.5, "min": 0,"max": 1,"step": 0.01}),
            },
            # "optional": {
            # }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("sigmas",)
    CATEGORY = "RES4LYF/schedulers"

    def main(self, steps, denoise, inflection_percent, model=None):
        sigmas = get_sigmas(model, "linear_quadratic", steps, denoise, inflection_percent)

        return (sigmas, )


class constant_scheduler:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "steps": ("INT", {"default": 40, "min": 0,"max": 100000,"step": 1}),
                "value_start": ("FLOAT", {"default": 1.0, "min": -100000,"max": 100000,"step": 0.01}),
                "value_end": ("FLOAT", {"default": 0.0, "min": -100000,"max": 100000,"step": 0.01}),
                "cutoff_percent": ("FLOAT", {"default": 1.0, "min": 0,"max": 1,"step": 0.01}),
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("sigmas",)
    CATEGORY = "RES4LYF/schedulers"

    def main(self, steps, value_start, value_end, cutoff_percent):
        sigmas = torch.ones(steps + 1) * value_start
        cutoff_step = int(round(steps * cutoff_percent)) + 1
        sigmas = torch.concat((sigmas[:cutoff_step], torch.ones(steps + 1 - cutoff_step) * value_end), dim=0)

        return (sigmas,)
    
    
    



class ClownScheduler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": { 
                "pad_start_value":      ("FLOAT",                                     {"default": 0.0, "min":  -10000.0, "max": 10000.0, "step": 0.01}),
                "start_value":          ("FLOAT",                                     {"default": 1.0, "min":  -10000.0, "max": 10000.0, "step": 0.01}),
                "end_value":            ("FLOAT",                                     {"default": 1.0, "min":  -10000.0, "max": 10000.0, "step": 0.01}),
                "pad_end_value":        ("FLOAT",                                     {"default": 0.0, "min":  -10000.0, "max": 10000.0, "step": 0.01}),
                "scheduler":            (["constant"] + get_res4lyf_scheduler_list(), {"default": "beta57"},),
                "scheduler_start_step": ("INT",                                       {"default": 0,   "min":  0,        "max": 10000}),
                "scheduler_end_step":   ("INT",                                       {"default": 30,  "min": -1,        "max": 10000}),
                "total_steps":          ("INT",                                       {"default": 100, "min": -1,        "max": 10000}),
                "flip_schedule":        ("BOOLEAN",                                   {"default": False}),
            }, 
            "optional": {
                "model":                ("MODEL", ),
            }
        }

    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("sigmas",)
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/schedulers"

    def create_callback(self, **kwargs):
        def callback(model):
            kwargs["model"] = model  
            schedule, = self.prepare_schedule(**kwargs)
            return schedule
        return callback

    def main(self,
            model                        = None,
            pad_start_value      : float = 1.0,
            start_value          : float = 0.0,
            end_value            : float = 1.0,
            pad_end_value                = None,
            denoise              : int   = 1.0,
            scheduler                    = None,
            scheduler_start_step : int   = 0,
            scheduler_end_step   : int   = 30,
            total_steps          : int   = 60,
            flip_schedule                = False,
            ) -> Tuple[Tensor]:
        
        if model is None:
            callback = self.create_callback(pad_start_value = pad_start_value,
                                            start_value     = start_value,
                                            end_value       = end_value,
                                            pad_end_value   = pad_end_value,
                                            
                                            scheduler       = scheduler,
                                            start_step      = scheduler_start_step,
                                            end_step        = scheduler_end_step,
                                            flip_schedule   = flip_schedule,
                                            )
        else:
            default_dtype  = torch.float64
            default_device = torch.device("cuda") 
            
            if scheduler_end_step == -1:
                scheduler_total_steps = total_steps - scheduler_start_step
            else:
                scheduler_total_steps = scheduler_end_step - scheduler_start_step
            
            if total_steps == -1:
                total_steps = scheduler_start_step + scheduler_end_step
            
            end_pad_steps = total_steps - scheduler_end_step
            
            if scheduler != "constant":
                values     = get_sigmas(model, scheduler, scheduler_total_steps, denoise).to(dtype=default_dtype, device=default_device) 
                values     = ((values - values.min()) * (start_value - end_value))   /   (values.max() - values.min())   +   end_value
            else:
                values = torch.linspace(start_value, end_value, scheduler_total_steps, dtype=default_dtype, device=default_device)
            
            if flip_schedule:
                values = torch.flip(values, dims=[0])
            
            prepend    = torch.full((scheduler_start_step,),  pad_start_value, dtype=default_dtype, device=default_device)
            postpend   = torch.full((end_pad_steps,),         pad_end_value,   dtype=default_dtype, device=default_device)
            
            values     = torch.cat((prepend, values, postpend), dim=0)

        #ositive[0][1]['callback_regional'] = callback
        
        return (values,)



    def prepare_schedule(self,
                                model                    = None,
                                pad_start_value  : float = 1.0,
                                start_value      : float = 0.0,
                                end_value        : float = 1.0,
                                pad_end_value            = None,
                                weight_scheduler         = None,
                                start_step       : int   = 0,
                                end_step         : int   = 30,
                                flip_schedule            = False,
                                ) -> Tuple[Tensor]:

        default_dtype  = torch.float64
        default_device = torch.device("cuda") 
        
        return (None,)




def get_sigmas_simple_exponential(model, steps):
    s = model.model_sampling
    sigs = []
    ss = len(s.sigmas) / steps
    for x in range(steps):
        sigs += [float(s.sigmas[-(1 + int(x * ss))])]
    sigs += [0.0]
    sigs = torch.FloatTensor(sigs)
    exp = torch.exp(torch.log(torch.linspace(1, 0, steps + 1)))
    return sigs * exp

extra_schedulers = {
    "simple_exponential": get_sigmas_simple_exponential
}



def get_sigmas(model, scheduler, steps, denoise, shift=0.0, lq_inflection_percent=0.5): #adapted from comfyui
    total_steps = steps
    if denoise < 1.0:
        if denoise <= 0.0:
            return (torch.FloatTensor([]),)
        total_steps = int(steps/denoise)

    try:
        model_sampling = model.get_model_object("model_sampling")
    except:
        if hasattr(model, "model"):
            model_sampling = model.model.model_sampling
        elif hasattr(model, "inner_model"):
            model_sampling = model.inner_model.inner_model.model_sampling
        else:
            raise Exception("get_sigmas: Could not get model_sampling")

    if shift > 1e-6:
        import copy
        model_sampling = copy.deepcopy(model_sampling)
        model_sampling.set_parameters(shift=shift)
        RESplain("model_sampling shift manually set to " + str(shift), debug=True)
    
    if scheduler == "beta57":
        sigmas = comfy.samplers.beta_scheduler(model_sampling, total_steps, alpha=0.5, beta=0.7).cpu()
    elif scheduler == "linear_quadratic":
        linear_steps = int(total_steps * lq_inflection_percent)
        sigmas = comfy.samplers.linear_quadratic_schedule(model_sampling, total_steps, threshold_noise=0.025, linear_steps=linear_steps).cpu()
    else:
        sigmas = comfy.samplers.calculate_sigmas(model_sampling, scheduler, total_steps).cpu()
    
    sigmas = sigmas[-(steps + 1):]
    return sigmas

#/// Adam Kormendi /// Inspired from Unreal Engine Maths ///


# Sigmoid Function
class sigmas_sigmoid:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "variant": (["logistic", "tanh", "softsign", "hardswish", "mish", "swish"], {"default": "logistic"}),
                "gain": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 10.0, "step": 0.01}),
                "offset": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "normalize_output": ("BOOLEAN", {"default": True})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, sigmas, variant, gain, offset, normalize_output):
        # Apply gain and offset
        x = gain * (sigmas + offset)
        
        if variant == "logistic":
            result = 1.0 / (1.0 + torch.exp(-x))
        elif variant == "tanh":
            result = torch.tanh(x)
        elif variant == "softsign":
            result = x / (1.0 + torch.abs(x))
        elif variant == "hardswish":
            result = x * torch.minimum(torch.maximum(x + 3, torch.zeros_like(x)), torch.tensor(6.0)) / 6.0
        elif variant == "mish":
            result = x * torch.tanh(torch.log(1.0 + torch.exp(x)))
        elif variant == "swish":
            result = x * torch.sigmoid(x)
        
        if normalize_output:
            # Normalize to [min(sigmas), max(sigmas)]
            result = ((result - result.min()) / (result.max() - result.min())) * (sigmas.max() - sigmas.min()) + sigmas.min()
            
        return (result,)

# ----- Easing Function -----
class sigmas_easing:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "easing_type": (["sine", "quad", "cubic", "quart", "quint", "expo", "circ", 
                                 "back", "elastic", "bounce"], {"default": "cubic"}),
                "easing_mode": (["in", "out", "in_out"], {"default": "in_out"}),
                "normalize_input": ("BOOLEAN", {"default": True}),
                "normalize_output": ("BOOLEAN", {"default": True}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, sigmas, easing_type, easing_mode, normalize_input, normalize_output, strength):
        # Normalize input to [0, 1] if requested
        if normalize_input:
            t = (sigmas - sigmas.min()) / (sigmas.max() - sigmas.min())
        else:
            t = torch.clamp(sigmas, 0.0, 1.0)
        
        # Apply strength
        t_orig = t.clone()
        t = t ** strength
            
        # Apply easing function based on type and mode
        if easing_mode == "in":
            result = self._ease_in(t, easing_type)
        elif easing_mode == "out":
            result = self._ease_out(t, easing_type)
        else:  # in_out
            result = self._ease_in_out(t, easing_type)
            
        # Normalize output if requested
        if normalize_output:
            if normalize_input:
                result = ((result - result.min()) / (result.max() - result.min())) * (sigmas.max() - sigmas.min()) + sigmas.min()
            else:
                result = ((result - result.min()) / (result.max() - result.min()))
                
        return (result,)
    
    def _ease_in(self, t, easing_type):
        if easing_type == "sine":
            return 1 - torch.cos((t * math.pi) / 2)
        elif easing_type == "quad":
            return t * t
        elif easing_type == "cubic":
            return t * t * t
        elif easing_type == "quart":
            return t * t * t * t
        elif easing_type == "quint":
            return t * t * t * t * t
        elif easing_type == "expo":
            return torch.where(t == 0, torch.zeros_like(t), torch.pow(2, 10 * t - 10))
        elif easing_type == "circ":
            return 1 - torch.sqrt(1 - torch.pow(t, 2))
        elif easing_type == "back":
            c1 = 1.70158
            c3 = c1 + 1
            return c3 * t * t * t - c1 * t * t
        elif easing_type == "elastic":
            c4 = (2 * math.pi) / 3
            return torch.where(
                t == 0, 
                torch.zeros_like(t),
                torch.where(
                    t == 1,
                    torch.ones_like(t),
                    -torch.pow(2, 10 * t - 10) * torch.sin((t * 10 - 10.75) * c4)
                )
            )
        elif easing_type == "bounce":
            return 1 - self._ease_out_bounce(1 - t)
    
    def _ease_out(self, t, easing_type):
        if easing_type == "sine":
            return torch.sin((t * math.pi) / 2)
        elif easing_type == "quad":
            return 1 - (1 - t) * (1 - t)
        elif easing_type == "cubic":
            return 1 - torch.pow(1 - t, 3)
        elif easing_type == "quart":
            return 1 - torch.pow(1 - t, 4)
        elif easing_type == "quint":
            return 1 - torch.pow(1 - t, 5)
        elif easing_type == "expo":
            return torch.where(t == 1, torch.ones_like(t), 1 - torch.pow(2, -10 * t))
        elif easing_type == "circ":
            return torch.sqrt(1 - torch.pow(t - 1, 2))
        elif easing_type == "back":
            c1 = 1.70158
            c3 = c1 + 1
            return 1 + c3 * torch.pow(t - 1, 3) + c1 * torch.pow(t - 1, 2)
        elif easing_type == "elastic":
            c4 = (2 * math.pi) / 3
            return torch.where(
                t == 0, 
                torch.zeros_like(t),
                torch.where(
                    t == 1,
                    torch.ones_like(t),
                    torch.pow(2, -10 * t) * torch.sin((t * 10 - 0.75) * c4) + 1
                )
            )
        elif easing_type == "bounce":
            return self._ease_out_bounce(t)
    
    def _ease_in_out(self, t, easing_type):
        if easing_type == "sine":
            return -(torch.cos(math.pi * t) - 1) / 2
        elif easing_type == "quad":
            return torch.where(t < 0.5, 2 * t * t, 1 - torch.pow(-2 * t + 2, 2) / 2)
        elif easing_type == "cubic":
            return torch.where(t < 0.5, 4 * t * t * t, 1 - torch.pow(-2 * t + 2, 3) / 2)
        elif easing_type == "quart":
            return torch.where(t < 0.5, 8 * t * t * t * t, 1 - torch.pow(-2 * t + 2, 4) / 2)
        elif easing_type == "quint":
            return torch.where(t < 0.5, 16 * t * t * t * t * t, 1 - torch.pow(-2 * t + 2, 5) / 2)
        elif easing_type == "expo":
            return torch.where(
                t < 0.5, 
                torch.pow(2, 20 * t - 10) / 2,
                (2 - torch.pow(2, -20 * t + 10)) / 2
            )
        elif easing_type == "circ":
            return torch.where(
                t < 0.5,
                (1 - torch.sqrt(1 - torch.pow(2 * t, 2))) / 2,
                (torch.sqrt(1 - torch.pow(-2 * t + 2, 2)) + 1) / 2
            )
        elif easing_type == "back":
            c1 = 1.70158
            c2 = c1 * 1.525
            return torch.where(
                t < 0.5,
                (torch.pow(2 * t, 2) * ((c2 + 1) * 2 * t - c2)) / 2,
                (torch.pow(2 * t - 2, 2) * ((c2 + 1) * (t * 2 - 2) + c2) + 2) / 2
            )
        elif easing_type == "elastic":
            c5 = (2 * math.pi) / 4.5
            return torch.where(
                t < 0.5,
                -(torch.pow(2, 20 * t - 10) * torch.sin((20 * t - 11.125) * c5)) / 2,
                (torch.pow(2, -20 * t + 10) * torch.sin((20 * t - 11.125) * c5)) / 2 + 1
            )
        elif easing_type == "bounce":
            return torch.where(
                t < 0.5,
                (1 - self._ease_out_bounce(1 - 2 * t)) / 2,
                (1 + self._ease_out_bounce(2 * t - 1)) / 2
            )
    
    def _ease_out_bounce(self, t):
        n1 = 7.5625
        d1 = 2.75
        
        mask1 = t < 1 / d1
        mask2 = t < 2 / d1
        mask3 = t < 2.5 / d1
        
        result = torch.zeros_like(t)
        result = torch.where(mask1, n1 * t * t, result)
        result = torch.where(mask2 & ~mask1, n1 * (t - 1.5 / d1) * (t - 1.5 / d1) + 0.75, result)
        result = torch.where(mask3 & ~mask2, n1 * (t - 2.25 / d1) * (t - 2.25 / d1) + 0.9375, result)
        result = torch.where(~mask3, n1 * (t - 2.625 / d1) * (t - 2.625 / d1) + 0.984375, result)
        
        return result

# -----  Hyperbolic Function -----
class sigmas_hyperbolic:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "function": (["sinh", "cosh", "tanh", "asinh", "acosh", "atanh"], {"default": "tanh"}),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 10.0, "step": 0.01}),
                "normalize_output": ("BOOLEAN", {"default": True})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, sigmas, function, scale, normalize_output):
        # Apply scaling
        x = sigmas * scale
        
        if function == "sinh":
            result = torch.sinh(x)
        elif function == "cosh":
            result = torch.cosh(x)
        elif function == "tanh":
            result = torch.tanh(x)
        elif function == "asinh":
            result = torch.asinh(x)
        elif function == "acosh":
            # Domain of acosh is [1, inf)
            result = torch.acosh(torch.clamp(x, min=1.0))
        elif function == "atanh":
            # Domain of atanh is (-1, 1)
            result = torch.atanh(torch.clamp(x, min=-0.99, max=0.99))
        
        if normalize_output:
            # Normalize to [min(sigmas), max(sigmas)]
            result = ((result - result.min()) / (result.max() - result.min())) * (sigmas.max() - sigmas.min()) + sigmas.min()
            
        return (result,)

# ----- Gaussian Distribution Function -----
class sigmas_gaussian:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "mean": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "std": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 10.0, "step": 0.01}),
                "operation": (["pdf", "cdf", "inverse_cdf", "transform", "modulate"], {"default": "transform"}),
                "normalize_output": ("BOOLEAN", {"default": True})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, sigmas, mean, std, operation, normalize_output):
        # Standardize values (z-score)
        z = (sigmas - sigmas.mean()) / sigmas.std()
        
        if operation == "pdf":
            # Probability density function
            result = (1 / (std * math.sqrt(2 * math.pi))) * torch.exp(-0.5 * ((sigmas - mean) / std) ** 2)
        elif operation == "cdf":
            # Cumulative distribution function
            result = 0.5 * (1 + torch.erf((sigmas - mean) / (std * math.sqrt(2))))
        elif operation == "inverse_cdf":
            # Inverse CDF (quantile function)
            # First normalize to [0.01, 0.99] to avoid numerical issues
            normalized = ((sigmas - sigmas.min()) / (sigmas.max() - sigmas.min())) * 0.98 + 0.01
            result = mean + std * torch.sqrt(2) * torch.erfinv(2 * normalized - 1)
        elif operation == "transform":
            # Transform to Gaussian distribution with specified mean and std
            result = z * std + mean
        elif operation == "modulate":
            # Modulate with a Gaussian curve centered at mean
            result = sigmas * torch.exp(-0.5 * ((sigmas - mean) / std) ** 2)
        
        if normalize_output:
            # Normalize to [min(sigmas), max(sigmas)]
            result = ((result - result.min()) / (result.max() - result.min())) * (sigmas.max() - sigmas.min()) + sigmas.min()
            
        return (result,)

# ----- Percentile Function -----
class sigmas_percentile:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "percentile_min": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 49.0, "step": 0.1}),
                "percentile_max": ("FLOAT", {"default": 95.0, "min": 51.0, "max": 100.0, "step": 0.1}),
                "target_min": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.01}),
                "target_max": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step": 0.01}),
                "clip_outliers": ("BOOLEAN", {"default": True})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, sigmas, percentile_min, percentile_max, target_min, target_max, clip_outliers):
        # Convert to numpy for percentile computation
        sigmas_np = sigmas.cpu().numpy()
        
        # Compute percentiles
        p_min = np.percentile(sigmas_np, percentile_min)
        p_max = np.percentile(sigmas_np, percentile_max)
        
        # Convert back to tensor
        p_min = torch.tensor(p_min, device=sigmas.device, dtype=sigmas.dtype)
        p_max = torch.tensor(p_max, device=sigmas.device, dtype=sigmas.dtype)
        
        # Map values from [p_min, p_max] to [target_min, target_max]
        if clip_outliers:
            sigmas_clipped = torch.clamp(sigmas, p_min, p_max)
            result = ((sigmas_clipped - p_min) / (p_max - p_min)) * (target_max - target_min) + target_min
        else:
            result = ((sigmas - p_min) / (p_max - p_min)) * (target_max - target_min) + target_min
            
        return (result,)

# ----- Kernel Smooth Function -----
class sigmas_kernel_smooth:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "kernel": (["gaussian", "box", "triangle", "epanechnikov", "cosine"], {"default": "gaussian"}),
                "kernel_size": ("INT", {"default": 5, "min": 3, "max": 51, "step": 2}),  # Must be odd
                "sigma": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, sigmas, kernel, kernel_size, sigma):
        # Ensure kernel_size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1
            
        # Define kernel weights
        if kernel == "gaussian":
            # Gaussian kernel
            kernel_1d = self._gaussian_kernel(kernel_size, sigma)
        elif kernel == "box":
            # Box (uniform) kernel
            kernel_1d = torch.ones(kernel_size, device=sigmas.device, dtype=sigmas.dtype) / kernel_size
        elif kernel == "triangle":
            # Triangle kernel
            x = torch.linspace(-(kernel_size//2), kernel_size//2, kernel_size, device=sigmas.device, dtype=sigmas.dtype)
            kernel_1d = (1.0 - torch.abs(x) / (kernel_size//2))
            kernel_1d = kernel_1d / kernel_1d.sum()
        elif kernel == "epanechnikov":
            # Epanechnikov kernel
            x = torch.linspace(-(kernel_size//2), kernel_size//2, kernel_size, device=sigmas.device, dtype=sigmas.dtype)
            x = x / (kernel_size//2)  # Scale to [-1, 1]
            kernel_1d = 0.75 * (1 - x**2)
            kernel_1d = kernel_1d / kernel_1d.sum()
        elif kernel == "cosine":
            # Cosine kernel
            x = torch.linspace(-(kernel_size//2), kernel_size//2, kernel_size, device=sigmas.device, dtype=sigmas.dtype)
            x = x / (kernel_size//2) * (math.pi/2)  # Scale to [-π/2, π/2]
            kernel_1d = torch.cos(x)
            kernel_1d = kernel_1d / kernel_1d.sum()
            
        # Pad input to handle boundary conditions
        pad_size = kernel_size // 2
        padded = F.pad(sigmas.unsqueeze(0).unsqueeze(0), (pad_size, pad_size), mode='reflect')
        
        # Apply convolution
        smoothed = F.conv1d(padded, kernel_1d.unsqueeze(0).unsqueeze(0))
        
        return (smoothed.squeeze(),)
    
    def _gaussian_kernel(self, kernel_size, sigma):
        # Generate 1D Gaussian kernel
        x = torch.linspace(-(kernel_size//2), kernel_size//2, kernel_size)
        kernel = torch.exp(-x**2 / (2*sigma**2))
        return kernel / kernel.sum()

# ----- Quantile Normalization -----
class sigmas_quantile_norm:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "target_distribution": (["uniform", "normal", "exponential", "logistic", "custom"], {"default": "uniform"}),
                "num_quantiles": ("INT", {"default": 100, "min": 10, "max": 1000, "step": 10}),
            },
            "optional": {
                "reference_sigmas": ("SIGMAS", {"forceInput": False}),
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, sigmas, target_distribution, num_quantiles, reference_sigmas=None):
        # Convert to numpy for processing
        sigmas_np = sigmas.cpu().numpy()
        
        # Sort values
        sorted_values = np.sort(sigmas_np)
        
        # Create rank for each value (fractional rank)
        ranks = np.zeros_like(sigmas_np)
        for i, val in enumerate(sigmas_np):
            ranks[i] = np.searchsorted(sorted_values, val, side='right') / len(sorted_values)
        
        # Generate target distribution
        if target_distribution == "uniform":
            # Uniform distribution between min and max of sigmas
            target_values = np.linspace(sigmas_np.min(), sigmas_np.max(), num_quantiles)
        elif target_distribution == "normal":
            # Normal distribution with same mean and std as sigmas
            target_values = np.random.normal(sigmas_np.mean(), sigmas_np.std(), num_quantiles)
            target_values.sort()
        elif target_distribution == "exponential":
            # Exponential distribution with lambda=1/mean
            target_values = np.random.exponential(1/max(1e-6, sigmas_np.mean()), num_quantiles)
            target_values.sort()
        elif target_distribution == "logistic":
            # Logistic distribution
            target_values = np.random.logistic(0, 1, num_quantiles)
            target_values.sort()
            # Rescale to match sigmas range
            target_values = (target_values - target_values.min()) / (target_values.max() - target_values.min())
            target_values = target_values * (sigmas_np.max() - sigmas_np.min()) + sigmas_np.min()
        elif target_distribution == "custom" and reference_sigmas is not None:
            # Use provided reference distribution
            reference_np = reference_sigmas.cpu().numpy()
            target_values = np.sort(reference_np)
            if len(target_values) < num_quantiles:
                # Interpolate if reference is smaller
                old_indices = np.linspace(0, len(target_values)-1, len(target_values))
                new_indices = np.linspace(0, len(target_values)-1, num_quantiles)
                target_values = np.interp(new_indices, old_indices, target_values)
            else:
                # Subsample if reference is larger
                indices = np.linspace(0, len(target_values)-1, num_quantiles, dtype=int)
                target_values = target_values[indices]
        else:
            # Default to uniform
            target_values = np.linspace(sigmas_np.min(), sigmas_np.max(), num_quantiles)
        
        # Map each value to its corresponding quantile in the target distribution
        result_np = np.interp(ranks, np.linspace(0, 1, len(target_values)), target_values)
        
        # Convert back to tensor
        result = torch.tensor(result_np, device=sigmas.device, dtype=sigmas.dtype)
        
        return (result,)

# ----- Adaptive Step Function -----
class sigmas_adaptive_step:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "adaptation_type": (["gradient", "curvature", "importance", "density"], {"default": "gradient"}),
                "sensitivity": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "min_step": ("FLOAT", {"default": 0.01, "min": 0.0001, "max": 1.0, "step": 0.01}),
                "max_step": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 10.0, "step": 0.01}),
                "target_steps": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, sigmas, adaptation_type, sensitivity, min_step, max_step, target_steps):
        if len(sigmas) <= 1:
            return (sigmas,)
            
        # Compute step sizes based on chosen adaptation type
        if adaptation_type == "gradient":
            # Compute gradient (first difference)
            grads = torch.abs(sigmas[1:] - sigmas[:-1])
            # Normalize gradients
            if grads.max() > grads.min():
                norm_grads = (grads - grads.min()) / (grads.max() - grads.min())
            else:
                norm_grads = torch.ones_like(grads)
            
            # Convert to step sizes: smaller steps where gradient is large
            step_sizes = 1.0 / (1.0 + norm_grads * sensitivity)
            
        elif adaptation_type == "curvature":
            # Compute second derivative approximation
            if len(sigmas) >= 3:
                # Second difference
                second_diff = sigmas[2:] - 2*sigmas[1:-1] + sigmas[:-2]
                # Pad to match length
                second_diff = F.pad(second_diff, (0, 1), mode='replicate')
            else:
                second_diff = torch.zeros_like(sigmas[:-1])
                
            # Normalize curvature
            abs_curve = torch.abs(second_diff)
            if abs_curve.max() > abs_curve.min():
                norm_curve = (abs_curve - abs_curve.min()) / (abs_curve.max() - abs_curve.min())
            else:
                norm_curve = torch.ones_like(abs_curve)
                
            # Convert to step sizes: smaller steps where curvature is high
            step_sizes = 1.0 / (1.0 + norm_curve * sensitivity)
            
        elif adaptation_type == "importance":
            # Importance based on values: focus more on extremes
            centered = torch.abs(sigmas - sigmas.mean())
            if centered.max() > centered.min():
                importance = (centered - centered.min()) / (centered.max() - centered.min())
            else:
                importance = torch.ones_like(centered)
                
            # Steps are smaller for important regions
            step_sizes = 1.0 / (1.0 + importance[:-1] * sensitivity)
            
        elif adaptation_type == "density":
            # Density-based adaptation using kernel density estimation
            # Use a simple histogram approximation
            sigma_min, sigma_max = sigmas.min(), sigmas.max()
            bins = 20
            hist = torch.histc(sigmas, bins=bins, min=sigma_min, max=sigma_max)
            hist = hist / hist.sum()  # Normalize
            
            # Map each sigma to its bin density
            bin_indices = torch.floor((sigmas - sigma_min) / (sigma_max - sigma_min) * (bins-1)).long()
            bin_indices = torch.clamp(bin_indices, 0, bins-1)
            densities = hist[bin_indices]
            
            # Compute step sizes: smaller steps in high density regions
            step_sizes = 1.0 / (1.0 + densities[:-1] * sensitivity)
        
        # Scale step sizes to [min_step, max_step]
        if step_sizes.max() > step_sizes.min():
            step_sizes = (step_sizes - step_sizes.min()) / (step_sizes.max() - step_sizes.min())
            step_sizes = step_sizes * (max_step - min_step) + min_step
        else:
            step_sizes = torch.ones_like(step_sizes) * min_step
            
        # Cumulative sum to get positions
        positions = torch.cat([torch.tensor([0.0], device=step_sizes.device), torch.cumsum(step_sizes, dim=0)])
        
        # Normalize positions to match original range
        positions = positions / positions[-1] * (sigmas[-1] - sigmas[0]) + sigmas[0]
        
        # Resample if target_steps is specified
        if target_steps > 0:
            new_positions = torch.linspace(sigmas[0], sigmas[-1], target_steps, device=sigmas.device)
            # Interpolate to get new sigma values
            new_sigmas = torch.zeros_like(new_positions)
            
            # Simple linear interpolation
            for i, pos in enumerate(new_positions):
                # Find enclosing original positions
                idx = torch.searchsorted(positions, pos)
                idx = torch.clamp(idx, 1, len(positions)-1)
                
                # Linear interpolation
                t = (pos - positions[idx-1]) / (positions[idx] - positions[idx-1])
                new_sigmas[i] = sigmas[idx-1] * (1-t) + sigmas[idx-1] * t
                
            result = new_sigmas
        else:
            result = positions
            
        return (result,)

# ----- Chaos Function -----
class sigmas_chaos:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "system": (["logistic", "henon", "tent", "sine", "cubic"], {"default": "logistic"}),
                "parameter": ("FLOAT", {"default": 3.9, "min": 0.1, "max": 5.0, "step": 0.01}),
                "iterations": ("INT", {"default": 10, "min": 1, "max": 100, "step": 1}),
                "normalize_output": ("BOOLEAN", {"default": True}),
                "use_as_seed": ("BOOLEAN", {"default": False})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, sigmas, system, parameter, iterations, normalize_output, use_as_seed):
        # Normalize input to [0,1] for chaotic maps
        if use_as_seed:
            # Use input as initial seed
            x = (sigmas - sigmas.min()) / (sigmas.max() - sigmas.min())
        else:
            # Use single initial value and apply iterations
            x = torch.zeros_like(sigmas)
            for i in range(len(sigmas)):
                # Use i/len as initial value for variety
                x[i] = i / len(sigmas)
        
        # Apply chaos map iterations
        for _ in range(iterations):
            if system == "logistic":
                # Logistic map: x_{n+1} = r * x_n * (1 - x_n)
                x = parameter * x * (1 - x)
                
            elif system == "henon":
                # Simplified 1D version of Henon map
                x = 1 - parameter * x**2
                
            elif system == "tent":
                # Tent map
                x = torch.where(x < 0.5, parameter * x, parameter * (1 - x))
                
            elif system == "sine":
                # Sine map: x_{n+1} = r * sin(pi * x_n)
                x = parameter * torch.sin(math.pi * x)
                
            elif system == "cubic":
                # Cubic map: x_{n+1} = r * x_n * (1 - x_n^2)
                x = parameter * x * (1 - x**2)
                
        # Normalize output if requested
        if normalize_output:
            result = ((x - x.min()) / (x.max() - x.min())) * (sigmas.max() - sigmas.min()) + sigmas.min()
        else:
            result = x
            
        return (result,)

# ----- Reaction Diffusion Function -----
class sigmas_reaction_diffusion:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "system": (["gray_scott", "fitzhugh_nagumo", "brusselator"], {"default": "gray_scott"}),
                "iterations": ("INT", {"default": 10, "min": 1, "max": 100, "step": 1}),
                "dt": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 1.0, "step": 0.01}),
                "param_a": ("FLOAT", {"default": 0.04, "min": 0.01, "max": 0.1, "step": 0.001}),
                "param_b": ("FLOAT", {"default": 0.06, "min": 0.01, "max": 0.1, "step": 0.001}),
                "diffusion_a": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 1.0, "step": 0.01}),
                "diffusion_b": ("FLOAT", {"default": 0.05, "min": 0.01, "max": 1.0, "step": 0.01}),
                "normalize_output": ("BOOLEAN", {"default": True})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, sigmas, system, iterations, dt, param_a, param_b, diffusion_a, diffusion_b, normalize_output):
        # Initialize a and b based on sigmas
        a = (sigmas - sigmas.min()) / (sigmas.max() - sigmas.min())
        b = 1.0 - a
        
        # Pad for diffusion calculation (periodic boundary)
        a_pad = F.pad(a.unsqueeze(0).unsqueeze(0), (1, 1), mode='circular').squeeze()
        b_pad = F.pad(b.unsqueeze(0).unsqueeze(0), (1, 1), mode='circular').squeeze()
        
        # Simple 1D reaction-diffusion
        for _ in range(iterations):
            # Compute Laplacian (diffusion term) as second derivative
            laplacian_a = a_pad[:-2] + a_pad[2:] - 2 * a
            laplacian_b = b_pad[:-2] + b_pad[2:] - 2 * b
            
            if system == "gray_scott":
                # Gray-Scott model for pattern formation
                # a is "U" (activator), b is "V" (inhibitor)
                feed = 0.055  # feed rate
                kill = 0.062  # kill rate
                
                # Update equations
                a_new = a + dt * (diffusion_a * laplacian_a - a * b**2 + feed * (1 - a))
                b_new = b + dt * (diffusion_b * laplacian_b + a * b**2 - (feed + kill) * b)
                
            elif system == "fitzhugh_nagumo":
                # FitzHugh-Nagumo model (simplified)
                # a is the membrane potential, b is the recovery variable
                
                # Update equations
                a_new = a + dt * (diffusion_a * laplacian_a + a - a**3 - b + param_a)
                b_new = b + dt * (diffusion_b * laplacian_b + param_b * (a - b))
                
            elif system == "brusselator":
                # Brusselator model
                # a is U, b is V
                
                # Update equations
                a_new = a + dt * (diffusion_a * laplacian_a + 1 - (param_b + 1) * a + param_a * a**2 * b)
                b_new = b + dt * (diffusion_b * laplacian_b + param_b * a - param_a * a**2 * b)
            
            # Update and repad
            a, b = a_new, b_new
            a_pad = F.pad(a.unsqueeze(0).unsqueeze(0), (1, 1), mode='circular').squeeze()
            b_pad = F.pad(b.unsqueeze(0).unsqueeze(0), (1, 1), mode='circular').squeeze()
            
        # Use the activator component as the result
        result = a
        
        # Normalize output if requested
        if normalize_output:
            result = ((result - result.min()) / (result.max() - result.min())) * (sigmas.max() - sigmas.min()) + sigmas.min()
            
        return (result,)

# ----- Attractor Function -----
class sigmas_attractor:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "attractor": (["lorenz", "rossler", "aizawa", "chen", "thomas"], {"default": "lorenz"}),
                "iterations": ("INT", {"default": 5, "min": 1, "max": 50, "step": 1}),
                "dt": ("FLOAT", {"default": 0.01, "min": 0.001, "max": 0.1, "step": 0.001}),
                "component": (["x", "y", "z", "magnitude"], {"default": "x"}),
                "normalize_output": ("BOOLEAN", {"default": True})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, sigmas, attractor, iterations, dt, component, normalize_output):
        # Initialize 3D state from sigmas
        n = len(sigmas)
        
        # Normalize sigmas to a reasonable range for the attractor
        norm_sigmas = (sigmas - sigmas.min()) / (sigmas.max() - sigmas.min()) * 2.0 - 1.0
        
        # Create initial state
        x = norm_sigmas
        y = torch.roll(norm_sigmas, 1)  # Shifted version for variety
        z = torch.roll(norm_sigmas, 2)  # Another shifted version
        
        # Parameters for the attractors
        if attractor == "lorenz":
            sigma, rho, beta = 10.0, 28.0, 8.0/3.0
        elif attractor == "rossler":
            a, b, c = 0.2, 0.2, 5.7
        elif attractor == "aizawa":
            a, b, c, d, e, f = 0.95, 0.7, 0.6, 3.5, 0.25, 0.1
        elif attractor == "chen":
            a, b, c = 5.0, -10.0, -0.38
        elif attractor == "thomas":
            b = 0.208186
            
        # Run the attractor dynamics
        for _ in range(iterations):
            if attractor == "lorenz":
                # Lorenz attractor
                dx = sigma * (y - x)
                dy = x * (rho - z) - y
                dz = x * y - beta * z
                
            elif attractor == "rossler":
                # Rössler attractor
                dx = -y - z
                dy = x + a * y
                dz = b + z * (x - c)
                
            elif attractor == "aizawa":
                # Aizawa attractor
                dx = (z - b) * x - d * y
                dy = d * x + (z - b) * y
                dz = c + a * z - z**3/3 - (x**2 + y**2) * (1 + e * z) + f * z * x**3
                
            elif attractor == "chen":
                # Chen attractor
                dx = a * (y - x)
                dy = (c - a) * x - x * z + c * y
                dz = x * y - b * z
                
            elif attractor == "thomas":
                # Thomas attractor
                dx = -b * x + torch.sin(y)
                dy = -b * y + torch.sin(z)
                dz = -b * z + torch.sin(x)
                
            # Update state
            x = x + dt * dx
            y = y + dt * dy
            z = z + dt * dz
            
        # Select component
        if component == "x":
            result = x
        elif component == "y":
            result = y
        elif component == "z":
            result = z
        elif component == "magnitude":
            result = torch.sqrt(x**2 + y**2 + z**2)
            
        # Normalize output if requested
        if normalize_output:
            result = ((result - result.min()) / (result.max() - result.min())) * (sigmas.max() - sigmas.min()) + sigmas.min()
            
        return (result,)

# ----- Catmull-Rom Spline -----
class sigmas_catmull_rom:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "tension": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "points": ("INT", {"default": 100, "min": 5, "max": 1000, "step": 5}),
                "boundary_condition": (["repeat", "clamp", "mirror"], {"default": "clamp"})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, sigmas, tension, points, boundary_condition):
        n = len(sigmas)
        
        # Need at least 4 points for Catmull-Rom interpolation
        if n < 4:
            # If we have fewer, just use linear interpolation
            t = torch.linspace(0, 1, points, device=sigmas.device)
            result = torch.zeros(points, device=sigmas.device, dtype=sigmas.dtype)
            
            for i in range(points):
                idx = min(int(i * (n - 1) / (points - 1)), n - 2)
                alpha = (i * (n - 1) / (points - 1)) - idx
                result[i] = (1 - alpha) * sigmas[idx] + alpha * sigmas[idx + 1]
                
            return (result,)
        
        # Handle boundary conditions for control points
        if boundary_condition == "repeat":
            # Repeat endpoints
            p0 = sigmas[0]
            p3 = sigmas[-1]
        elif boundary_condition == "clamp":
            # Extrapolate
            p0 = 2 * sigmas[0] - sigmas[1]
            p3 = 2 * sigmas[-1] - sigmas[-2]
        elif boundary_condition == "mirror":
            # Mirror
            p0 = sigmas[1]
            p3 = sigmas[-2]
            
        # Create extended control points
        control_points = torch.cat([torch.tensor([p0], device=sigmas.device), sigmas, torch.tensor([p3], device=sigmas.device)])
        
        # Compute spline
        result = torch.zeros(points, device=sigmas.device, dtype=sigmas.dtype)
        
        # Parameter to adjust curve tension (0 = Catmull-Rom, 1 = Linear)
        alpha = 1.0 - tension
        
        for i in range(points):
            # Determine which segment we're in
            t = i / (points - 1) * (n - 1)
            idx = min(int(t), n - 2)
            
            # Normalized parameter within the segment [0, 1]
            t_local = t - idx
            
            # Get control points for this segment
            p0 = control_points[idx]
            p1 = control_points[idx + 1]
            p2 = control_points[idx + 2]
            p3 = control_points[idx + 3]
            
            # Catmull-Rom basis functions
            t2 = t_local * t_local
            t3 = t2 * t_local
            
            # Compute spline point
            result[i] = (
                (-alpha * t3 + 2 * alpha * t2 - alpha * t_local) * p0 +
                ((2 - alpha) * t3 + (alpha - 3) * t2 + 1) * p1 +
                ((alpha - 2) * t3 + (3 - 2 * alpha) * t2 + alpha * t_local) * p2 +
                (alpha * t3 - alpha * t2) * p3
            ) * 0.5
            
        return (result,)

# ----- Lambert W-Function -----
class sigmas_lambert_w:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "branch": (["principal", "secondary"], {"default": "principal"}),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 10.0, "step": 0.01}),
                "normalize_output": ("BOOLEAN", {"default": True}),
                "max_iterations": ("INT", {"default": 20, "min": 5, "max": 100, "step": 1})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, sigmas, branch, scale, normalize_output, max_iterations):
        # Apply scaling
        x = sigmas * scale
        
        # Lambert W function (numerically approximated)
        result = torch.zeros_like(x)
        
        # Process each value separately (since Lambert W is non-vectorized)
        for i in range(len(x)):
            xi = x[i].item()
            
            # Initial guess varies by branch
            if branch == "principal":
                # Valid for x >= -1/e
                if xi < -1/math.e:
                    xi = -1/math.e  # Clamp to domain
                
                # Initial guess for W₀(x)
                if xi < 0:
                    w = 0.0
                elif xi < 1:
                    w = xi * (1 - xi * (1 - 0.5 * xi))
                else:
                    w = math.log(xi)
                    
            else:  # secondary branch
                # Valid for -1/e <= x < 0
                if xi < -1/math.e:
                    xi = -1/math.e  # Clamp to lower bound
                elif xi >= 0:
                    xi = -0.01  # Clamp to upper bound
                
                # Initial guess for W₋₁(x)
                w = math.log(-xi)
                
            # Halley's method for numerical approximation
            for _ in range(max_iterations):
                ew = math.exp(w)
                wew = w * ew
                
                # If we've converged, break
                if abs(wew - xi) < 1e-10:
                    break
                
                # Halley's update
                wpe = w + 1  # w plus 1
                div = ew * wpe - (ew * w - xi) * wpe / (2 * wpe * ew)
                w_next = w - (wew - xi) / div
                
                # Check for convergence
                if abs(w_next - w) < 1e-10:
                    w = w_next
                    break
                    
                w = w_next
                
            result[i] = w
            
        # Normalize output if requested
        if normalize_output:
            result = ((result - result.min()) / (result.max() - result.min())) * (sigmas.max() - sigmas.min()) + sigmas.min()
            
        return (result,)

# ----- Zeta & Eta Functions -----
class sigmas_zeta_eta:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "function": (["riemann_zeta", "dirichlet_eta", "lerch_phi"], {"default": "riemann_zeta"}),
                "offset": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.1}),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 10.0, "step": 0.01}),
                "normalize_output": ("BOOLEAN", {"default": True}),
                "approx_terms": ("INT", {"default": 100, "min": 10, "max": 1000, "step": 10})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, sigmas, function, offset, scale, normalize_output, approx_terms):
        # Apply offset and scaling
        s = sigmas * scale + offset
        
        # Process based on function type
        if function == "riemann_zeta":
            # Riemann zeta function
            # For Re(s) > 1, ζ(s) = sum(1/n^s, n=1 to infinity)
            # For performance reasons, we'll use scipy's implementation for CPU
            # and a truncated series approximation for GPU
            
            # Move to CPU for scipy
            s_cpu = s.cpu().numpy()
            
            # Apply zeta function
            result_np = np.zeros_like(s_cpu)
            
            for i, si in enumerate(s_cpu):
                # Handle special values
                if si == 1.0:
                    # ζ(1) is the harmonic series, which diverges to infinity
                    result_np[i] = float('inf')
                elif si < 0 and si == int(si) and int(si) % 2 == 0:
                    # ζ(-2n) = 0 for n > 0
                    result_np[i] = 0.0
                else:
                    try:
                        # Use scipy for computation
                        result_np[i] = float(special.zeta(si))
                    except (ValueError, OverflowError):
                        # Fall back to approximation for problematic values
                        if si > 1:
                            # Truncated series for Re(s) > 1
                            result_np[i] = sum(1.0 / np.power(n, si) for n in range(1, approx_terms))
                        else:
                            # Use functional equation for Re(s) < 0
                            if si < 0:
                                # ζ(s) = 2^s π^(s-1) sin(πs/2) Γ(1-s) ζ(1-s)
                                # Gamma function blows up at negative integers, so use the fact that
                                # ζ(-n) = -B_{n+1}/(n+1) for n > 0, where B is a Bernoulli number
                                # However, as this gets complex, we'll use a simpler approximation
                                result_np[i] = 0.0  # Default for problematic values
            
            # Convert back to tensor
            result = torch.tensor(result_np, device=sigmas.device, dtype=sigmas.dtype)
            
        elif function == "dirichlet_eta":
            # Dirichlet eta function (alternating zeta function)
            # η(s) = sum((-1)^(n+1)/n^s, n=1 to infinity)
            
            # For GPU efficiency, compute directly using alternating series
            result = torch.zeros_like(s)
            
            # Use a fixed number of terms for approximation
            for i in range(1, approx_terms + 1):
                term = torch.pow(i, -s) * (1 if i % 2 == 1 else -1)
                result += term
                
        elif function == "lerch_phi":
            # Lerch transcendent with fixed parameters
            # Φ(z, s, a) = sum(z^n / (n+a)^s, n=0 to infinity)
            # We'll use z=0.5, a=1 for simplicity
            z, a = 0.5, 1.0
            
            result = torch.zeros_like(s)
            for i in range(approx_terms):
                term = torch.pow(z, i) / torch.pow(i + a, s)
                result += term
            
        # Replace infinities and NaNs with large or small values
        result = torch.where(torch.isfinite(result), result, torch.sign(result) * 1e10)
        
        # Normalize output if requested
        if normalize_output:
            result = ((result - result.min()) / (result.max() - result.min())) * (sigmas.max() - sigmas.min()) + sigmas.min()
            
        return (result,)

# ----- Gamma & Beta Functions -----
class sigmas_gamma_beta:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "function": (["gamma", "beta", "incomplete_gamma", "incomplete_beta", "log_gamma"], {"default": "gamma"}),
                "offset": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.1}),
                "scale": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 10.0, "step": 0.01}),
                "parameter_a": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 10.0, "step": 0.1}),
                "parameter_b": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 10.0, "step": 0.1}),
                "normalize_output": ("BOOLEAN", {"default": True})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, sigmas, function, offset, scale, parameter_a, parameter_b, normalize_output):
        # Apply offset and scaling
        x = sigmas * scale + offset
        
        # Convert to numpy for special functions
        x_np = x.cpu().numpy()
        
        # Apply function
        if function == "gamma":
            # Gamma function Γ(x)
            # For performance and stability, use scipy
            result_np = np.zeros_like(x_np)
            
            for i, xi in enumerate(x_np):
                # Handle special cases
                if xi <= 0 and xi == int(xi):
                    # Gamma has poles at non-positive integers
                    result_np[i] = float('inf')
                else:
                    try:
                        result_np[i] = float(special.gamma(xi))
                    except (ValueError, OverflowError):
                        # Use approximation for large values
                        result_np[i] = float('inf')
                        
        elif function == "log_gamma":
            # Log Gamma function log(Γ(x))
            # More numerically stable for large values
            result_np = np.zeros_like(x_np)
            
            for i, xi in enumerate(x_np):
                # Handle special cases
                if xi <= 0 and xi == int(xi):
                    # log(Γ(x)) is undefined for non-positive integers
                    result_np[i] = float('inf')
                else:
                    try:
                        result_np[i] = float(special.gammaln(xi))
                    except (ValueError, OverflowError):
                        # Use approximation for large values
                        result_np[i] = float('inf')
                    
        elif function == "beta":
            # Beta function B(a, x)
            result_np = np.zeros_like(x_np)
            
            for i, xi in enumerate(x_np):
                try:
                    result_np[i] = float(special.beta(parameter_a, xi))
                except (ValueError, OverflowError):
                    # Handle cases where beta is undefined
                    result_np[i] = float('inf')
                    
        elif function == "incomplete_gamma":
            # Regularized incomplete gamma function P(a, x)
            result_np = np.zeros_like(x_np)
            
            for i, xi in enumerate(x_np):
                if xi < 0:
                    # Undefined for negative x
                    result_np[i] = 0.0
                else:
                    try:
                        result_np[i] = float(special.gammainc(parameter_a, xi))
                    except (ValueError, OverflowError):
                        result_np[i] = 1.0  # Approach 1 for large x
                    
        elif function == "incomplete_beta":
            # Regularized incomplete beta function I(x; a, b)
            result_np = np.zeros_like(x_np)
            
            for i, xi in enumerate(x_np):
                # Clamp to [0,1] for domain of incomplete beta
                xi_clamped = min(max(xi, 0), 1)
                
                try:
                    result_np[i] = float(special.betainc(parameter_a, parameter_b, xi_clamped))
                except (ValueError, OverflowError):
                    result_np[i] = 0.5  # Default for errors
                    
        # Convert back to tensor
        result = torch.tensor(result_np, device=sigmas.device, dtype=sigmas.dtype)
        
        # Replace infinities and NaNs
        result = torch.where(torch.isfinite(result), result, torch.sign(result) * 1e10)
        
        # Normalize output if requested
        if normalize_output:
            # Handle cases where result has infinities
            if torch.isinf(result).any() or torch.isnan(result).any():
                # Replace inf/nan with max/min finite values
                max_val = torch.max(result[torch.isfinite(result)]) if torch.any(torch.isfinite(result)) else 1e10
                min_val = torch.min(result[torch.isfinite(result)]) if torch.any(torch.isfinite(result)) else -1e10
                
                result = torch.where(torch.isinf(result) & (result > 0), max_val, result)
                result = torch.where(torch.isinf(result) & (result < 0), min_val, result)
                result = torch.where(torch.isnan(result), (max_val + min_val) / 2, result)
            
            # Now normalize
            result = ((result - result.min()) / (result.max() - result.min())) * (sigmas.max() - sigmas.min()) + sigmas.min()
            
        return (result,)

# ----- Sigma Lerp -----
class sigmas_lerp:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas_a": ("SIGMAS", {"forceInput": True}),
                "sigmas_b": ("SIGMAS", {"forceInput": True}),
                "t": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "ensure_length": ("BOOLEAN", {"default": True})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, sigmas_a, sigmas_b, t, ensure_length):
        if ensure_length and len(sigmas_a) != len(sigmas_b):
            # Resize the smaller one to match the larger one
            if len(sigmas_a) < len(sigmas_b):
                sigmas_a = torch.nn.functional.interpolate(
                    sigmas_a.unsqueeze(0).unsqueeze(0), 
                    size=len(sigmas_b), 
                    mode='linear'
                ).squeeze(0).squeeze(0)
            else:
                sigmas_b = torch.nn.functional.interpolate(
                    sigmas_b.unsqueeze(0).unsqueeze(0), 
                    size=len(sigmas_a), 
                    mode='linear'
                ).squeeze(0).squeeze(0)
        
        return ((1 - t) * sigmas_a + t * sigmas_b,)

# ----- Sigma InvLerp -----
class sigmas_invlerp:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "min_value": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.01}),
                "max_value": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step": 0.01})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, sigmas, min_value, max_value):
        # Clamp values to avoid division by zero
        if min_value == max_value:
            max_value = min_value + 1e-5
            
        normalized = (sigmas - min_value) / (max_value - min_value)
        # Clamp the values to be in [0, 1]
        normalized = torch.clamp(normalized, 0.0, 1.0)
        return (normalized,)

# ----- Sigma ArcSine -----
class sigmas_arcsine:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "normalize_input": ("BOOLEAN", {"default": True}),
                "scale_output": ("BOOLEAN", {"default": True}),
                "out_min": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.01}),
                "out_max": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step": 0.01})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, sigmas, normalize_input, scale_output, out_min, out_max):
        if normalize_input:
            sigmas = torch.clamp(sigmas, -1.0, 1.0)
        else:
            # Ensure values are in valid arcsin domain
            sigmas = torch.clamp(sigmas, -1.0, 1.0)
            
        result = torch.asin(sigmas)
        
        if scale_output:
            # ArcSine output is in range [-π/2, π/2]
            # Normalize to [0, 1] and then scale to [out_min, out_max]
            result = (result + math.pi/2) / math.pi
            result = result * (out_max - out_min) + out_min
            
        return (result,)

# ----- Sigma LinearSine -----
class sigmas_linearsine:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "amplitude": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0, "step": 0.01}),
                "frequency": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "phase": ("FLOAT", {"default": 0.0, "min": -6.28, "max": 6.28, "step": 0.01}), # -2π to 2π
                "linear_weight": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, sigmas, amplitude, frequency, phase, linear_weight):
        # Create indices for the sine function
        indices = torch.linspace(0, 1, len(sigmas), device=sigmas.device)
        
        # Calculate sine component
        sine_component = amplitude * torch.sin(2 * math.pi * frequency * indices + phase)
        
        # Blend linear and sine components
        step_indices = torch.linspace(0, 1, len(sigmas), device=sigmas.device)
        result = linear_weight * sigmas + (1 - linear_weight) * (step_indices.unsqueeze(0) * sine_component)
        
        return (result.squeeze(0),)

# ----- Sigmas Append -----
class sigmas_append:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "value": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.01}),
                "count": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1})
            },
            "optional": {
                "additional_sigmas": ("SIGMAS", {"forceInput": False})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, sigmas, value, count, additional_sigmas=None):
        # Create tensor of the value to append
        append_values = torch.full((count,), value, device=sigmas.device, dtype=sigmas.dtype)
        
        # Append the values
        result = torch.cat([sigmas, append_values], dim=0)
        
        # If additional sigmas provided, append those as well
        if additional_sigmas is not None:
            result = torch.cat([result, additional_sigmas], dim=0)
            
        return (result,)

# ----- Sigma Arccosine -----
class sigmas_arccosine:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "normalize_input": ("BOOLEAN", {"default": True}),
                "scale_output": ("BOOLEAN", {"default": True}),
                "out_min": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.01}),
                "out_max": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step": 0.01})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, sigmas, normalize_input, scale_output, out_min, out_max):
        if normalize_input:
            sigmas = torch.clamp(sigmas, -1.0, 1.0)
        else:
            # Ensure values are in valid arccos domain
            sigmas = torch.clamp(sigmas, -1.0, 1.0)
            
        result = torch.acos(sigmas)
        
        if scale_output:
            # ArcCosine output is in range [0, π]
            # Normalize to [0, 1] and then scale to [out_min, out_max]
            result = result / math.pi
            result = result * (out_max - out_min) + out_min
            
        return (result,)

# ----- Sigma Arctangent -----
class sigmas_arctangent:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "scale_output": ("BOOLEAN", {"default": True}),
                "out_min": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.01}),
                "out_max": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step": 0.01})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, sigmas, scale_output, out_min, out_max):
        result = torch.atan(sigmas)
        
        if scale_output:
            # ArcTangent output is in range [-π/2, π/2]
            # Normalize to [0, 1] and then scale to [out_min, out_max]
            result = (result + math.pi/2) / math.pi
            result = result * (out_max - out_min) + out_min
            
        return (result,)

# ----- Sigma CrossProduct -----
class sigmas_crossproduct:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas_a": ("SIGMAS", {"forceInput": True}),
                "sigmas_b": ("SIGMAS", {"forceInput": True}),
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, sigmas_a, sigmas_b):
        # Ensure we have at least 3 elements in each tensor
        # If not, pad with zeros or truncate
        if len(sigmas_a) < 3:
            sigmas_a = torch.nn.functional.pad(sigmas_a, (0, 3 - len(sigmas_a)))
        if len(sigmas_b) < 3:
            sigmas_b = torch.nn.functional.pad(sigmas_b, (0, 3 - len(sigmas_b)))
        
        # Take the first 3 elements of each tensor
        a = sigmas_a[:3]
        b = sigmas_b[:3]
        
        # Compute cross product
        c = torch.zeros(3, device=sigmas_a.device, dtype=sigmas_a.dtype)
        c[0] = a[1] * b[2] - a[2] * b[1]
        c[1] = a[2] * b[0] - a[0] * b[2]
        c[2] = a[0] * b[1] - a[1] * b[0]
        
        return (c,)

# ----- Sigma DotProduct -----
class sigmas_dotproduct:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas_a": ("SIGMAS", {"forceInput": True}),
                "sigmas_b": ("SIGMAS", {"forceInput": True}),
                "normalize": ("BOOLEAN", {"default": False})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, sigmas_a, sigmas_b, normalize):
        # Ensure equal lengths by taking the minimum
        min_length = min(len(sigmas_a), len(sigmas_b))
        a = sigmas_a[:min_length]
        b = sigmas_b[:min_length]
        
        if normalize:
            a_norm = torch.norm(a)
            b_norm = torch.norm(b)
            # Avoid division by zero
            if a_norm > 0 and b_norm > 0:
                a = a / a_norm
                b = b / b_norm
        
        # Compute dot product
        result = torch.sum(a * b)
        
        # Return as a single-element tensor
        return (torch.tensor([result], device=sigmas_a.device, dtype=sigmas_a.dtype),)

# ----- Sigma Fmod -----
class sigmas_fmod:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "divisor": ("FLOAT", {"default": 1.0, "min": 0.0001, "max": 10000.0, "step": 0.01})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, sigmas, divisor):
        # Ensure divisor is not zero
        if divisor == 0:
            divisor = 0.0001
            
        result = torch.fmod(sigmas, divisor)
        return (result,)

# ----- Sigma Frac -----
class sigmas_frac:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, sigmas):
        # Get the fractional part (x - floor(x))
        result = sigmas - torch.floor(sigmas)
        return (result,)

# ----- Sigma If -----
class sigmas_if:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "condition_sigmas": ("SIGMAS", {"forceInput": True}),
                "true_sigmas": ("SIGMAS", {"forceInput": True}),
                "false_sigmas": ("SIGMAS", {"forceInput": True}),
                "threshold": ("FLOAT", {"default": 0.5, "min": -10000.0, "max": 10000.0, "step": 0.01}),
                "comp_type": (["greater", "less", "equal", "not_equal"], {"default": "greater"})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, condition_sigmas, true_sigmas, false_sigmas, threshold, comp_type):
        # Make sure we have values to compare
        max_length = max(len(condition_sigmas), len(true_sigmas), len(false_sigmas))
        
        # Extend all tensors to the maximum length using interpolation
        if len(condition_sigmas) != max_length:
            condition_sigmas = torch.nn.functional.interpolate(
                condition_sigmas.unsqueeze(0).unsqueeze(0), 
                size=max_length, 
                mode='linear'
            ).squeeze(0).squeeze(0)
            
        if len(true_sigmas) != max_length:
            true_sigmas = torch.nn.functional.interpolate(
                true_sigmas.unsqueeze(0).unsqueeze(0), 
                size=max_length, 
                mode='linear'
            ).squeeze(0).squeeze(0)
            
        if len(false_sigmas) != max_length:
            false_sigmas = torch.nn.functional.interpolate(
                false_sigmas.unsqueeze(0).unsqueeze(0), 
                size=max_length, 
                mode='linear'
            ).squeeze(0).squeeze(0)
            
        # Create mask based on comparison type
        if comp_type == "greater":
            mask = condition_sigmas > threshold
        elif comp_type == "less":
            mask = condition_sigmas < threshold
        elif comp_type == "equal":
            mask = torch.isclose(condition_sigmas, torch.tensor(threshold, device=condition_sigmas.device))
        elif comp_type == "not_equal":
            mask = ~torch.isclose(condition_sigmas, torch.tensor(threshold, device=condition_sigmas.device))
        
        # Apply the mask to select values
        result = torch.where(mask, true_sigmas, false_sigmas)
        
        return (result,)

# ----- Sigma Logarithm2 -----
class sigmas_logarithm2:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "handle_negative": ("BOOLEAN", {"default": True}),
                "epsilon": ("FLOAT", {"default": 1e-10, "min": 1e-15, "max": 0.1, "step": 1e-10})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, sigmas, handle_negative, epsilon):
        if handle_negative:
            # For negative values, compute -log2(-x) and negate the result
            mask_negative = sigmas < 0
            mask_positive = ~mask_negative
            
            # Prepare positive and negative parts
            pos_part = torch.log2(torch.clamp(sigmas[mask_positive], min=epsilon))
            neg_part = -torch.log2(torch.clamp(-sigmas[mask_negative], min=epsilon))
            
            # Create result tensor
            result = torch.zeros_like(sigmas)
            result[mask_positive] = pos_part
            result[mask_negative] = neg_part
        else:
            # Simply compute log2, clamping values to avoid log(0)
            result = torch.log2(torch.clamp(sigmas, min=epsilon))
            
        return (result,)

# ----- Sigma SmoothStep -----
class sigmas_smoothstep:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "edge0": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.01}),
                "edge1": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step": 0.01}),
                "mode": (["smoothstep", "smootherstep"], {"default": "smoothstep"})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, sigmas, edge0, edge1, mode):
        # Normalize the values to the range [0, 1]
        t = torch.clamp((sigmas - edge0) / (edge1 - edge0), 0.0, 1.0)
        
        if mode == "smoothstep":
            # Smooth step: 3t^2 - 2t^3
            result = t * t * (3.0 - 2.0 * t)
        else:  # smootherstep
            # Smoother step: 6t^5 - 15t^4 + 10t^3
            result = t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
            
        # Scale back to the original range
        result = result * (edge1 - edge0) + edge0
        
        return (result,)

# ----- Sigma SquareRoot -----
class sigmas_squareroot:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "handle_negative": ("BOOLEAN", {"default": False})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, sigmas, handle_negative):
        if handle_negative:
            # For negative values, compute sqrt(-x) and negate the result
            mask_negative = sigmas < 0
            mask_positive = ~mask_negative
            
            # Prepare positive and negative parts
            pos_part = torch.sqrt(sigmas[mask_positive])
            neg_part = -torch.sqrt(-sigmas[mask_negative])
            
            # Create result tensor
            result = torch.zeros_like(sigmas)
            result[mask_positive] = pos_part
            result[mask_negative] = neg_part
        else:
            # Only compute square root for non-negative values
            # Negative values will be set to 0
            result = torch.sqrt(torch.clamp(sigmas, min=0))
            
        return (result,)

# ----- Sigma TimeStep -----
class sigmas_timestep:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "dt": ("FLOAT", {"default": 0.1, "min": 0.0001, "max": 10.0, "step": 0.01}),
                "scaling": (["linear", "quadratic", "sqrt", "log"], {"default": "linear"}),
                "decay": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, sigmas, dt, scaling, decay):
        # Create time steps
        timesteps = torch.arange(len(sigmas), device=sigmas.device, dtype=sigmas.dtype) * dt
        
        # Apply scaling
        if scaling == "quadratic":
            timesteps = timesteps ** 2
        elif scaling == "sqrt":
            timesteps = torch.sqrt(timesteps)
        elif scaling == "log":
            # Add small epsilon to avoid log(0)
            timesteps = torch.log(timesteps + 1e-10)
            
        # Apply decay
        if decay > 0:
            decay_factor = torch.exp(-decay * timesteps)
            timesteps = timesteps * decay_factor
            
        # Normalize to match the range of sigmas
        timesteps = ((timesteps - timesteps.min()) / 
                     (timesteps.max() - timesteps.min())) * (sigmas.max() - sigmas.min()) + sigmas.min()
            
        return (timesteps,)

class sigmas_gaussian_cdf:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "mu": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "sigma": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 10.0, "step": 0.01}),
                "normalize_output": ("BOOLEAN", {"default": True})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, sigmas, mu, sigma, normalize_output):
        # Apply Gaussian CDF transformation
        result = 0.5 * (1 + torch.erf((sigmas - mu) / (sigma * math.sqrt(2))))
        
        # Normalize output if requested
        if normalize_output:
            result = ((result - result.min()) / (result.max() - result.min())) * (sigmas.max() - sigmas.min()) + sigmas.min()
            
        return (result,)

class sigmas_stepwise_multirate:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "steps": ("INT", {"default": 30, "min": 1, "max": 1000, "step": 1}),
                "rates": ("STRING", {"default": "1.0,0.5,0.25", "multiline": False}),
                "boundaries": ("STRING", {"default": "0.3,0.7", "multiline": False}),
                "start_value": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "end_value": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 100.0, "step": 0.01}),
                "pad_end": ("BOOLEAN", {"default": True})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, steps, rates, boundaries, start_value, end_value, pad_end):
        # Parse rates and boundaries
        rates_list = [float(r) for r in rates.split(',')]
        if len(rates_list) < 1:
            rates_list = [1.0]
            
        boundaries_list = [float(b) for b in boundaries.split(',')]
        if len(boundaries_list) != len(rates_list) - 1:
            # Create equal size segments if boundaries don't match rates
            boundaries_list = [i / len(rates_list) for i in range(1, len(rates_list))]
        
        # Convert boundaries to step indices
        boundary_indices = [int(b * steps) for b in boundaries_list]
        
        # Create steps array
        result = torch.zeros(steps)
        
        # Fill segments with different rates
        current_idx = 0
        for i, rate in enumerate(rates_list):
            next_idx = boundary_indices[i] if i < len(boundary_indices) else steps
            segment_length = next_idx - current_idx
            if segment_length <= 0:
                continue
                
            segment_start = start_value if i == 0 else result[current_idx-1]
            segment_end = end_value if i == len(rates_list) - 1 else start_value * (1 - boundaries_list[i])
            
            # Apply rate to the segment
            t = torch.linspace(0, 1, segment_length)
            segment = segment_start + (segment_end - segment_start) * (t ** rate)
            
            result[current_idx:next_idx] = segment
            current_idx = next_idx
        
        # Add padding zero at the end if requested
        if pad_end:
            result = torch.cat([result, torch.tensor([0.0])])
            
        return (result,)

class sigmas_harmonic_decay:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "steps": ("INT", {"default": 30, "min": 1, "max": 1000, "step": 1}),
                "start_value": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "end_value": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 100.0, "step": 0.01}),
                "harmonic_offset": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "decay_rate": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "pad_end": ("BOOLEAN", {"default": True})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, steps, start_value, end_value, harmonic_offset, decay_rate, pad_end):
        # Create harmonic series: 1/(n+offset)^rate
        n = torch.arange(1, steps + 1, dtype=torch.float32)
        harmonic_values = 1.0 / torch.pow(n + harmonic_offset, decay_rate)
        
        # Normalize to [0, 1]
        normalized = (harmonic_values - harmonic_values.min()) / (harmonic_values.max() - harmonic_values.min())
        
        # Scale to [end_value, start_value] and reverse (higher values first)
        result = start_value - (start_value - end_value) * normalized
        result = torch.flip(result, [0])
        
        # Add padding zero at the end if requested
        if pad_end:
            result = torch.cat([result, torch.tensor([0.0])])
            
        return (result,)

class sigmas_adaptive_noise_floor:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "min_noise_level": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 1.0, "step": 0.001}),
                "adaptation_factor": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "window_size": ("INT", {"default": 3, "min": 1, "max": 10, "step": 1})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, sigmas, min_noise_level, adaptation_factor, window_size):
        # Initialize result with original sigmas
        result = sigmas.clone()
        
        # Apply adaptive noise floor
        for i in range(window_size, len(sigmas)):
            # Calculate local statistics in the window
            window = sigmas[i-window_size:i]
            local_mean = torch.mean(window)
            local_var = torch.var(window)
            
            # Adapt the noise floor based on local statistics
            adaptive_floor = min_noise_level + adaptation_factor * local_var / (local_mean + 1e-6)
            
            # Apply the floor if needed
            if result[i] < adaptive_floor:
                result[i] = adaptive_floor
        
        return (result,)

class sigmas_collatz_iteration:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "iterations": ("INT", {"default": 3, "min": 1, "max": 20, "step": 1}),
                "scaling_factor": ("FLOAT", {"default": 0.1, "min": 0.0001, "max": 10.0, "step": 0.01}),
                "normalize_output": ("BOOLEAN", {"default": True})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, sigmas, iterations, scaling_factor, normalize_output):
        # Scale input to reasonable range for Collatz
        scaled_input = sigmas * scaling_factor
        
        # Apply Collatz iterations
        result = scaled_input.clone()
        
        for _ in range(iterations):
            # Create masks for even and odd values
            even_mask = (result % 2 == 0)
            odd_mask = ~even_mask
            
            # Apply Collatz function: n/2 for even, 3n+1 for odd
            result[even_mask] = result[even_mask] / 2
            result[odd_mask] = 3 * result[odd_mask] + 1
        
        # Normalize output if requested
        if normalize_output:
            result = ((result - result.min()) / (result.max() - result.min())) * (sigmas.max() - sigmas.min()) + sigmas.min()
            
        return (result,)

class sigmas_conway_sequence:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "steps": ("INT", {"default": 20, "min": 1, "max": 50, "step": 1}),
                "sequence_type": (["look_and_say", "audioactive", "paperfolding", "thue_morse"], {"default": "look_and_say"}),
                "normalize_range": ("BOOLEAN", {"default": True}),
                "min_value": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 10.0, "step": 0.01}),
                "max_value": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 50.0, "step": 0.1})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, steps, sequence_type, normalize_range, min_value, max_value):
        if sequence_type == "look_and_say":
            # Start with "1"
            s = "1"
            lengths = [1]  # Length of first term is 1
            
            # Generate look-and-say sequence
            for _ in range(min(steps - 1, 25)):  # Limit to prevent excessive computation
                next_s = ""
                i = 0
                while i < len(s):
                    count = 1
                    while i + 1 < len(s) and s[i] == s[i + 1]:
                        i += 1
                        count += 1
                    next_s += str(count) + s[i]
                    i += 1
                s = next_s
                lengths.append(len(s))
            
            # Convert to tensor
            result = torch.tensor(lengths, dtype=torch.float32)
            
        elif sequence_type == "audioactive":
            # Audioactive sequence (similar to look-and-say but counts digits)
            a = [1]
            for _ in range(min(steps - 1, 30)):
                b = []
                digit_count = {}
                for digit in a:
                    digit_count[digit] = digit_count.get(digit, 0) + 1
                
                for digit in sorted(digit_count.keys()):
                    b.append(digit_count[digit])
                    b.append(digit)
                a = b
            
            result = torch.tensor(a, dtype=torch.float32)
            if len(result) > steps:
                result = result[:steps]
            
        elif sequence_type == "paperfolding":
            # Paper folding sequence (dragon curve)
            sequence = []
            for i in range(min(steps, 30)):
                sequence.append(1 if (i & (i + 1)) % 2 == 0 else 0)
            
            result = torch.tensor(sequence, dtype=torch.float32)
            
        elif sequence_type == "thue_morse":
            # Thue-Morse sequence
            sequence = [0]
            while len(sequence) < steps:
                sequence.extend([1 - x for x in sequence])
            
            result = torch.tensor(sequence, dtype=torch.float32)[:steps]
        
        # Normalize to desired range
        if normalize_range:
            if result.max() > result.min():
                result = (result - result.min()) / (result.max() - result.min())
                result = result * (max_value - min_value) + min_value
            else:
                result = torch.ones_like(result) * min_value
        
        return (result,)

class sigmas_gilbreath_sequence:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "steps": ("INT", {"default": 30, "min": 10, "max": 100, "step": 1}),
                "levels": ("INT", {"default": 3, "min": 1, "max": 10, "step": 1}),
                "normalize_range": ("BOOLEAN", {"default": True}),
                "min_value": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 10.0, "step": 0.01}),
                "max_value": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 50.0, "step": 0.1})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, steps, levels, normalize_range, min_value, max_value):
        # Generate first few prime numbers
        def sieve_of_eratosthenes(limit):
            sieve = [True] * (limit + 1)
            sieve[0] = sieve[1] = False
            for i in range(2, int(limit**0.5) + 1):
                if sieve[i]:
                    for j in range(i*i, limit + 1, i):
                        sieve[j] = False
            return [i for i in range(limit + 1) if sieve[i]]
        
        # Get primes
        primes = sieve_of_eratosthenes(steps * 6)  # Get enough primes
        primes = primes[:steps]
        
        # Generate Gilbreath sequence levels
        sequences = [primes]
        for level in range(1, levels):
            prev_seq = sequences[level-1]
            new_seq = [abs(prev_seq[i] - prev_seq[i+1]) for i in range(len(prev_seq)-1)]
            sequences.append(new_seq)
        
        # Select the requested level
        selected_level = min(levels-1, len(sequences)-1)
        result_list = sequences[selected_level]
        
        # Ensure we have enough values
        while len(result_list) < steps:
            result_list.append(1)  # Gilbreath conjecture: eventually all 1s
        
        # Convert to tensor
        result = torch.tensor(result_list[:steps], dtype=torch.float32)
        
        # Normalize to desired range
        if normalize_range:
            if result.max() > result.min():
                result = (result - result.min()) / (result.max() - result.min())
                result = result * (max_value - min_value) + min_value
            else:
                result = torch.ones_like(result) * min_value
        
        return (result,)

class sigmas_cnf_inverse:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "time_steps": ("INT", {"default": 20, "min": 5, "max": 100, "step": 1}),
                "flow_type": (["linear", "quadratic", "sigmoid", "exponential"], {"default": "sigmoid"}),
                "reverse": ("BOOLEAN", {"default": True})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, sigmas, time_steps, flow_type, reverse):
        # Create normalized time steps
        t = torch.linspace(0, 1, time_steps)
        
        # Apply CNF flow transformation
        if flow_type == "linear":
            flow = t
        elif flow_type == "quadratic":
            flow = t**2
        elif flow_type == "sigmoid":
            flow = 1 / (1 + torch.exp(-10 * (t - 0.5)))
        elif flow_type == "exponential":
            flow = torch.exp(3 * t) - 1
            flow = flow / flow.max()  # Normalize to [0,1]
        
        # Reverse flow if requested
        if reverse:
            flow = 1 - flow
        
        # Interpolate sigmas according to flow
        # First normalize sigmas to [0,1] for interpolation
        normalized_sigmas = (sigmas - sigmas.min()) / (sigmas.max() - sigmas.min())
        
        # Create indices for interpolation
        indices = flow * (len(sigmas) - 1)
        
        # Linear interpolation
        result = torch.zeros(time_steps, device=sigmas.device, dtype=sigmas.dtype)
        for i in range(time_steps):
            idx_low = int(indices[i])
            idx_high = min(idx_low + 1, len(sigmas) - 1)
            frac = indices[i] - idx_low
            
            result[i] = (1 - frac) * normalized_sigmas[idx_low] + frac * normalized_sigmas[idx_high]
        
        # Scale back to original sigma range
        result = result * (sigmas.max() - sigmas.min()) + sigmas.min()
        
        return (result,)

class sigmas_riemannian_flow:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "steps": ("INT", {"default": 30, "min": 5, "max": 100, "step": 1}),
                "metric_type": (["euclidean", "hyperbolic", "spherical", "lorentzian"], {"default": "hyperbolic"}),
                "curvature": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "start_value": ("FLOAT", {"default": 10.0, "min": 0.1, "max": 50.0, "step": 0.1}),
                "end_value": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 10.0, "step": 0.01})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, steps, metric_type, curvature, start_value, end_value):
        # Create parameter t in [0, 1]
        t = torch.linspace(0, 1, steps)
        
        # Apply different Riemannian metrics
        if metric_type == "euclidean":
            # Simple linear interpolation in Euclidean space
            result = start_value * (1 - t) + end_value * t
            
        elif metric_type == "hyperbolic":
            # Hyperbolic space geodesic
            K = -curvature  # Negative curvature for hyperbolic space
            
            # Convert to hyperbolic coordinates (using Poincaré disk model)
            x_start = torch.tanh(start_value / 2)
            x_end = torch.tanh(end_value / 2)
            
            # Distance in hyperbolic space
            d = torch.acosh(1 + 2 * ((x_start - x_end)**2) / ((1 - x_start**2) * (1 - x_end**2)))
            
            # Geodesic interpolation
            lambda_t = torch.sinh(t * d) / torch.sinh(d)
            result = 2 * torch.atanh((1 - lambda_t) * x_start + lambda_t * x_end)
            
        elif metric_type == "spherical":
            # Spherical space geodesic (great circle)
            K = curvature  # Positive curvature for spherical space
            
            # Convert to angular coordinates
            theta_start = start_value * torch.sqrt(K)
            theta_end = end_value * torch.sqrt(K)
            
            # Geodesic interpolation along great circle
            result = torch.sin((1 - t) * theta_start + t * theta_end) / torch.sqrt(K)
            
        elif metric_type == "lorentzian":
            # Lorentzian spacetime-inspired metric (time dilation effect)
            gamma = 1 / torch.sqrt(1 - curvature * t**2)  # Lorentz factor
            result = start_value * (1 - t) + end_value * t
            result = result * gamma  # Apply time dilation
        
        # Ensure the values are in the desired range
        result = torch.clamp(result, min=min(start_value, end_value), max=max(start_value, end_value))
        
        # Ensure result is decreasing if start_value > end_value
        if start_value > end_value and result[0] < result[-1]:
            result = torch.flip(result, [0])
            
        return (result,)

class sigmas_langevin_dynamics:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "steps": ("INT", {"default": 30, "min": 5, "max": 100, "step": 1}),
                "start_value": ("FLOAT", {"default": 10.0, "min": 0.1, "max": 50.0, "step": 0.1}),
                "end_value": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 10.0, "step": 0.01}),
                "temperature": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 10.0, "step": 0.01}),
                "friction": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 99999, "step": 1})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, steps, start_value, end_value, temperature, friction, seed):
        # Set random seed for reproducibility
        torch.manual_seed(seed)
        
        # Potential function (quadratic well centered at end_value)
        def U(x):
            return 0.5 * (x - end_value)**2
        
        # Gradient of the potential
        def grad_U(x):
            return x - end_value
        
        # Initialize state
        x = torch.tensor([start_value], dtype=torch.float32)
        v = torch.zeros(1)  # Initial velocity
        
        # Discretization parameters
        dt = 1.0 / steps
        sqrt_2dt = math.sqrt(2 * dt)
        
        # Storage for trajectory
        trajectory = [start_value]
        
        # Langevin dynamics integration (velocity Verlet with Langevin thermostat)
        for _ in range(steps - 1):
            # Half step in velocity
            v = v - dt * friction * v - dt * grad_U(x) / 2
            
            # Full step in position
            x = x + dt * v
            
            # Random force (thermal noise)
            noise = torch.randn(1) * sqrt_2dt * temperature
            
            # Another half step in velocity with noise
            v = v - dt * friction * v - dt * grad_U(x) / 2 + noise
            
            # Store current position
            trajectory.append(x.item())
        
        # Convert to tensor
        result = torch.tensor(trajectory, dtype=torch.float32)
        
        # Ensure we reach the end value
        result[-1] = end_value
        
        return (result,)

class sigmas_persistent_homology:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "steps": ("INT", {"default": 30, "min": 5, "max": 100, "step": 1}),
                "start_value": ("FLOAT", {"default": 10.0, "min": 0.1, "max": 50.0, "step": 0.1}),
                "end_value": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 10.0, "step": 0.01}),
                "persistence_type": (["linear", "exponential", "logarithmic", "sigmoidal"], {"default": "exponential"}),
                "birth_density": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "death_density": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, steps, start_value, end_value, persistence_type, birth_density, death_density):
        # Basic filtration function (linear by default)
        t = torch.linspace(0, 1, steps)
        
        # Persistence diagram simulation
        # Create birth and death times
        birth_points = int(steps * birth_density)
        death_points = int(steps * death_density)
        
        # Filtration function based on selected type
        if persistence_type == "linear":
            filtration = t
        elif persistence_type == "exponential":
            filtration = 1 - torch.exp(-5 * t)
        elif persistence_type == "logarithmic":
            filtration = torch.log(1 + 9 * t) / torch.log(torch.tensor([10.0]))
        elif persistence_type == "sigmoidal":
            filtration = 1 / (1 + torch.exp(-10 * (t - 0.5)))
        
        # Generate birth-death pairs
        birth_indices = torch.linspace(0, steps // 2, birth_points).long()
        death_indices = torch.linspace(steps // 2, steps - 1, death_points).long()
        
        # Create persistence barcode
        barcode = torch.zeros(steps)
        for b_idx in birth_indices:
            for d_idx in death_indices:
                if b_idx < d_idx:
                    # Add a persistence feature from birth to death
                    barcode[b_idx:d_idx] += 1
        
        # Normalize and weight the barcode
        if barcode.max() > 0:
            barcode = barcode / barcode.max()
        
        # Modulate the filtration function with the persistence barcode
        result = filtration * (0.7 + 0.3 * barcode)
        
        # Scale to desired range
        result = start_value + (end_value - start_value) * result
        
        return (result,)

class sigmas_normalizing_flows:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "steps": ("INT", {"default": 30, "min": 5, "max": 100, "step": 1}),
                "start_value": ("FLOAT", {"default": 10.0, "min": 0.1, "max": 50.0, "step": 0.1}),
                "end_value": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 10.0, "step": 0.01}),
                "flow_type": (["affine", "planar", "radial", "realnvp"], {"default": "realnvp"}),
                "num_transforms": ("INT", {"default": 3, "min": 1, "max": 10, "step": 1}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 99999, "step": 1})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, steps, start_value, end_value, flow_type, num_transforms, seed):
        # Set random seed for reproducibility
        torch.manual_seed(seed)
        
        # Create base linear schedule from start_value to end_value
        base_schedule = torch.linspace(start_value, end_value, steps)
        
        # Apply different normalizing flow transformations
        if flow_type == "affine":
            # Affine transformation: f(x) = a*x + b
            result = base_schedule.clone()
            for _ in range(num_transforms):
                a = torch.rand(1) * 0.5 + 0.75  # Scale in [0.75, 1.25]
                b = (torch.rand(1) - 0.5) * 0.2  # Shift in [-0.1, 0.1]
                result = a * result + b
                
        elif flow_type == "planar":
            # Planar flow: f(x) = x + u * tanh(w * x + b)
            result = base_schedule.clone()
            for _ in range(num_transforms):
                u = torch.rand(1) * 0.4 - 0.2  # in [-0.2, 0.2]
                w = torch.rand(1) * 2 - 1  # in [-1, 1]
                b = torch.rand(1) * 0.2 - 0.1  # in [-0.1, 0.1]
                result = result + u * torch.tanh(w * result + b)
                
        elif flow_type == "radial":
            # Radial flow: f(x) = x + beta * (x - x0) / (alpha + |x - x0|)
            result = base_schedule.clone()
            for _ in range(num_transforms):
                # Pick a random reference point within the range
                idx = torch.randint(0, steps, (1,))
                x0 = result[idx]
                
                alpha = torch.rand(1) * 0.5 + 0.5  # in [0.5, 1.0]
                beta = torch.rand(1) * 0.4 - 0.2  # in [-0.2, 0.2]
                
                # Apply radial flow
                diff = result - x0
                r = torch.abs(diff)
                result = result + beta * diff / (alpha + r)
                
        elif flow_type == "realnvp":
            # Simplified RealNVP-inspired flow with masking
            result = base_schedule.clone()
            
            for _ in range(num_transforms):
                # Create alternating mask
                mask = torch.zeros(steps)
                mask[::2] = 1  # Mask even indices
                
                # Generate scale and shift parameters
                log_scale = torch.rand(steps) * 0.2 - 0.1  # in [-0.1, 0.1]
                shift = torch.rand(steps) * 0.2 - 0.1  # in [-0.1, 0.1]
                
                # Apply affine coupling transformation
                scale = torch.exp(log_scale * mask)
                masked_shift = shift * mask
                
                # Transform
                result = result * scale + masked_shift
        
        # Rescale to ensure we maintain start_value and end_value
        if result[0] != start_value or result[-1] != end_value:
            result = (result - result[0]) / (result[-1] - result[0]) * (end_value - start_value) + start_value
        
        return (result,)


class sigmas_cnf_flow:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "flow_steps": ("INT", {"default": 20, "min": 5, "max": 100, "step": 1}),
                "flow_type": (["hutchinson", "dopri5", "euler"], {"default": "euler"}),
                "drift_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "diffusion_scale": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 5.0, "step": 0.1}),
                "regularization": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 1.0, "step": 0.01})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, sigmas, flow_steps, flow_type, drift_scale, diffusion_scale, regularization):
        # Drift function (simple example)
        def drift_fn(t, sigma_t):
            drift = -sigma_t
            # Add regularization (smoothing)
            if regularization > 0:
                drift_smooth = torch.nn.functional.pad(drift.unsqueeze(0).unsqueeze(0), (1, 1), mode='reflect')
                drift_smooth = torch.nn.functional.avg_pool1d(drift_smooth, kernel_size=3, stride=1).squeeze()
                drift = (1 - regularization) * drift + regularization * drift_smooth
            return drift_scale * drift
        
        # Diffusion function
        def diffusion_fn(t, sigma_t):
            return diffusion_scale * torch.ones_like(sigma_t)
        
        # Initial value
        sigma_0 = sigmas.clone()
        
        # Time steps
        ts = torch.linspace(0, 1, flow_steps, device=sigma_0.device)
        dt = 1.0 / (flow_steps - 1)
        
        # CNF solver selection
        if flow_type == "euler":
            # Simple Euler method
            sigma_t = sigma_0
            trajectory = [sigma_t.clone()]
            
            for i in range(1, flow_steps):
                t = ts[i-1]
                # Drift term
                dsigma = drift_fn(t, sigma_t) * dt
                # Diffusion term (only for SDE)
                if diffusion_scale > 0:
                    noise = torch.randn_like(sigma_t) * torch.sqrt(torch.tensor(dt, device=sigma_t.device))
                    dsigma += diffusion_fn(t, sigma_t) * noise
                
                sigma_t = sigma_t + dsigma
                trajectory.append(sigma_t.clone())
                
            result = torch.stack(trajectory)
            
        elif flow_type == "dopri5":
            # Dopri5 approximation (5th order Runge-Kutta with adaptive step size)
            sigma_t = sigma_0
            trajectory = [sigma_t.clone()]
            
            for i in range(1, flow_steps):
                t = ts[i-1]
                
                # Runge-Kutta coefficients
                k1 = drift_fn(t, sigma_t)
                k2 = drift_fn(t + 0.2*dt, sigma_t + 0.2*dt*k1)
                k3 = drift_fn(t + 0.3*dt, sigma_t + 0.3*dt*k1 + 0.9*dt*k2)
                k4 = drift_fn(t + 0.8*dt, sigma_t + 0.8*dt*k1 + 0.9*dt*k2 + 2.4*dt*k3)
                k5 = drift_fn(t + 0.8*dt, sigma_t - 0.8*dt*k1 + 1.4*dt*k2 - 0.4*dt*k3 + 1.6*dt*k4)
                k6 = drift_fn(t + dt, sigma_t + 0.05*dt*k1 + 0.25*dt*k4 + 0.2*dt*k5)
                
                # 5th order update
                sigma_t = sigma_t + dt * (0.0616667*k1 + 0.0*k2 + 0.276*k3 + 0.328333*k4 + 0.2*k5 + 0.133333*k6)
                
                # Diffusion term (only for SDE)
                if diffusion_scale > 0:
                    noise = torch.randn_like(sigma_t) * torch.sqrt(torch.tensor(dt, device=sigma_t.device))
                    sigma_t += diffusion_fn(t, sigma_t) * noise
                
                trajectory.append(sigma_t.clone())
            
            result = torch.stack(trajectory)
            
        elif flow_type == "hutchinson":
            # Hutchinson's stochastic trace estimator
            sigma_t = sigma_0
            trajectory = [sigma_t.clone()]
            
            for i in range(1, flow_steps):
                t = ts[i-1]
                
                # Forward difference for divergence estimation
                eps = 1e-5
                v = torch.randn_like(sigma_t)
                f_x = drift_fn(t, sigma_t)
                f_x_eps = drift_fn(t, sigma_t + eps * v)
                div_estimate = ((f_x_eps - f_x) * v).sum() / eps
                
                # Drift with divergence correction
                dsigma = (drift_fn(t, sigma_t) - 0.5 * diffusion_scale**2 * div_estimate) * dt
                
                # Diffusion term
                if diffusion_scale > 0:
                    noise = torch.randn_like(sigma_t) * torch.sqrt(torch.tensor(dt, device=sigma_t.device))
                    dsigma += diffusion_scale * noise
                
                sigma_t = sigma_t + dsigma
                trajectory.append(sigma_t.clone())
            
            result = torch.stack(trajectory)
        
        # Concatenate and return results
        return (result,)


class sigmas_trajectory_flow:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "target_sigmas": ("SIGMAS", {"forceInput": True}),
                "optimization_steps": ("INT", {"default": 10, "min": 1, "max": 50, "step": 1}),
                "learning_rate": ("FLOAT", {"default": 0.01, "min": 0.001, "max": 0.1, "step": 0.001}),
                "regularization": ("FLOAT", {"default": 0.005, "min": 0.0, "max": 0.1, "step": 0.001}),
                "smoothness": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, sigmas, target_sigmas, optimization_steps, learning_rate, regularization, smoothness):
        with torch.inference_mode(False):
            # Ensure equal lengths
            if len(sigmas) != len(target_sigmas):
                if len(sigmas) > len(target_sigmas):
                    target_sigmas = torch.nn.functional.interpolate(
                        target_sigmas.unsqueeze(0).unsqueeze(0), 
                        size=len(sigmas), 
                        mode='linear'
                    ).squeeze(0).squeeze(0)
                else:
                    sigmas = torch.nn.functional.interpolate(
                        sigmas.unsqueeze(0).unsqueeze(0), 
                        size=len(target_sigmas), 
                        mode='linear'
                    ).squeeze(0).squeeze(0)
            
            # Create optimizable trajectory
            trajectory = sigmas.clone().detach().requires_grad_(True)
            
            # Optimizer
            optimizer = torch.optim.Adam([trajectory], lr=learning_rate)
            
            # Optimization loop
            for step in range(optimization_steps):
                optimizer.zero_grad()
                
                # Transport cost (endpoint matching)
                transport_cost = ((trajectory[0] - sigmas[0])**2 + (trajectory[-1] - target_sigmas[-1])**2)
                
                # Path regularization (smoothness)
                diff = trajectory[1:] - trajectory[:-1]
                smoothness_cost = smoothness * torch.mean(diff**2)
                
                # Kinetic energy (total variation)
                kinetic_energy = torch.sum(torch.abs(diff))
                
                # Total loss
                loss = transport_cost + smoothness_cost + regularization * kinetic_energy
                
                loss.backward()
                optimizer.step()
            
            # Return optimized trajectory
            return (trajectory.detach(),)




class sigmas_rectified_flow:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "rectification_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "probability_flow": ("BOOLEAN", {"default": True}),
                "time_steps": ("INT", {"default": 20, "min": 5, "max": 100, "step": 1}),
                "normalize_output": ("BOOLEAN", {"default": True})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, sigmas, rectification_strength, probability_flow, time_steps, normalize_output):
        # Create time steps
        t = torch.linspace(0, 1, time_steps, device=sigmas.device)
        
        # Interpolate original sigmas to match time_steps
        original_sigmas = torch.nn.functional.interpolate(
            sigmas.unsqueeze(0).unsqueeze(0), 
            size=time_steps, 
            mode='linear'
        ).squeeze(0).squeeze(0)
        
        # Create rectified path
        start_point = original_sigmas[0]
        end_point = original_sigmas[-1]
        
        # Straight line path (optimal transport)
        linear_path = start_point * (1 - t) + end_point * t
        
        # Apply rectification
        if probability_flow:
            # Probability flow ODE rectification
            # Calculate drift term
            drift = torch.zeros_like(original_sigmas)
            drift[1:-1] = (original_sigmas[2:] - original_sigmas[:-2]) / 2  # Central difference
            drift[0] = original_sigmas[1] - original_sigmas[0]
            drift[-1] = original_sigmas[-1] - original_sigmas[-2]
            
            # Score function approximation
            score = -drift / (original_sigmas + 1e-6)
            
            # Rectified path
            rectified_path = torch.zeros_like(original_sigmas)
            
            # Solve ODE with improved Euler method
            dt = 1.0 / (time_steps - 1)
            x = start_point.clone()
            rectified_path[0] = x
            
            for i in range(1, time_steps):
                # Interpolate score at current position
                idx = int(i * (len(score) - 1) / (time_steps - 1))
                frac = i * (len(score) - 1) / (time_steps - 1) - idx
                s = (1 - frac) * score[idx] + frac * score[min(idx + 1, len(score) - 1)]
                
                # Improved Euler step
                k1 = -s * x
                x_mid = x + 0.5 * dt * k1
                k2 = -s * x_mid
                x = x + dt * k2
                
                rectified_path[i] = x
        else:
            # Simple linear interpolation between straight line and original
            rectified_path = (1 - rectification_strength) * original_sigmas + rectification_strength * linear_path
        
        # Normalize output if requested
        if normalize_output:
            rectified_path = (rectified_path - rectified_path.min()) / (rectified_path.max() - rectified_path.min())
            rectified_path = rectified_path * (sigmas.max() - sigmas.min()) + sigmas.min()
        
        return (rectified_path,)


class sigmas_stochastic_flow:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "noise_schedule": (["linear", "cosine", "adaptive", "brownian"], {"default": "cosine"}),
                "stochasticity": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 999999, "step": 1})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, sigmas, noise_schedule, stochasticity, seed):
        # Set random seed
        torch.manual_seed(seed)
        
        # Create noise schedule
        t = torch.linspace(0, 1, len(sigmas), device=sigmas.device)
        
        if noise_schedule == "linear":
            noise_level = stochasticity * t * (1 - t)  # Parabolic profile
            
        elif noise_schedule == "cosine":
            noise_level = stochasticity * 0.5 * (1 - torch.cos(math.pi * t))
            
        elif noise_schedule == "adaptive":
            # More noise in regions of high gradient
            grad = torch.zeros_like(sigmas)
            grad[1:-1] = torch.abs(sigmas[2:] - sigmas[:-2]) / 2  # Central difference
            grad[0] = torch.abs(sigmas[1] - sigmas[0])
            grad[-1] = torch.abs(sigmas[-1] - sigmas[-2])
            
            # Normalize gradient
            grad = grad / grad.max()
            
            # Adaptive noise level
            noise_level = stochasticity * grad * (1 - t)  # Higher at start, lower at end
            
        elif noise_schedule == "brownian":
            # Brownian bridge setup
            noise_level = stochasticity * torch.sqrt(t * (1 - t))
        
        # Generate noise
        noise = torch.randn_like(sigmas) * noise_level.view(-1)
        
        # Apply noise to create stochastic path
        result = sigmas + noise
        
        # Ensure endpoints match the original
        result[0] = sigmas[0]
        result[-1] = sigmas[-1]
        
        return (result,)

class sigmas_wavelet_flow:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "wavelet_type": (["haar", "simple", "adaptive"], {"default": "adaptive"}),
                "decomposition_level": ("INT", {"default": 3, "min": 1, "max": 10, "step": 1}),
                "threshold": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "reconstruction_mode": (["soft", "hard", "garrote"], {"default": "soft"})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, sigmas, wavelet_type, decomposition_level, threshold, reconstruction_mode):
        # Pad input to nearest power of 2 length
        n = len(sigmas)
        next_pow2 = 2 ** math.ceil(math.log2(n))
        padded = torch.nn.functional.pad(sigmas, (0, next_pow2 - n))
        
        # Decomposition function
        def wavelet_decompose(data, level):
            coeffs = []
            approx = data.clone()
            
            for i in range(level):
                # Choose wavelet filter
                if wavelet_type == "haar":
                    # Haar wavelet
                    h0 = torch.tensor([0.7071, 0.7071], device=data.device)  # Low-pass
                    h1 = torch.tensor([0.7071, -0.7071], device=data.device) # High-pass
                elif wavelet_type == "simple":
                    # Simple Daubechies-like
                    h0 = torch.tensor([0.6, 0.4], device=data.device)
                    h1 = torch.tensor([0.4, -0.6], device=data.device)
                elif wavelet_type == "adaptive":
                    # Adaptive filter based on signal properties
                    local_var = torch.var(approx)
                    alpha = torch.clamp(0.5 + 0.3 * local_var, 0.5, 0.8)
                    h0 = torch.tensor([alpha, 1-alpha], device=data.device)
                    h1 = torch.tensor([1-alpha, -alpha], device=data.device)
                
                # Apply filters
                length = len(approx) // 2
                new_approx = torch.zeros(length, device=data.device)
                detail = torch.zeros(length, device=data.device)
                
                for j in range(length):
                    idx = j * 2
                    new_approx[j] = approx[idx] * h0[0] + approx[idx+1] * h0[1]
                    detail[j] = approx[idx] * h1[0] + approx[idx+1] * h1[1]
                
                coeffs.append(detail)
                approx = new_approx
                
                if len(approx) <= 2:
                    break
            
            coeffs.append(approx)
            return coeffs
        
        # Reconstruction function
        def wavelet_reconstruct(coeffs, level):
            approx = coeffs[-1]
            
            for i in range(level):
                detail = coeffs[level-i-1]
                length = len(approx)
                
                # Choose wavelet filter
                if wavelet_type == "haar":
                    g0 = torch.tensor([0.7071, 0.7071], device=approx.device)
                    g1 = torch.tensor([-0.7071, 0.7071], device=approx.device)
                elif wavelet_type == "simple":
                    g0 = torch.tensor([0.4, 0.6], device=approx.device)
                    g1 = torch.tensor([-0.6, 0.4], device=approx.device)
                elif wavelet_type == "adaptive":
                    local_var = torch.var(approx)
                    alpha = torch.clamp(0.5 + 0.3 * local_var, 0.5, 0.8)
                    g0 = torch.tensor([1-alpha, alpha], device=approx.device)
                    g1 = torch.tensor([alpha, -1+alpha], device=approx.device)
                
                # Reconstruct
                new_approx = torch.zeros(length*2, device=approx.device)
                for j in range(length):
                    idx = j * 2
                    new_approx[idx] = approx[j] * g0[0] + detail[j] * g1[0]
                    new_approx[idx+1] = approx[j] * g0[1] + detail[j] * g1[1]
                
                approx = new_approx
            
            return approx
        
        # Apply wavelet decomposition
        coeffs = wavelet_decompose(padded, decomposition_level)
        
        # Thresholding
        for i in range(len(coeffs)-1):  # Skip approximation coefficients
            detail = coeffs[i]
            abs_detail = torch.abs(detail)
            mask = abs_detail > threshold
            
            if reconstruction_mode == "hard":
                # Hard thresholding - keep or zero
                detail = detail * mask
            elif reconstruction_mode == "soft":
                # Soft thresholding - shrink toward zero
                detail = torch.sign(detail) * torch.max(abs_detail - threshold, torch.zeros_like(detail)) * mask
            elif reconstruction_mode == "garrote":
                # Non-negative Garrote
                detail = detail * torch.max(1 - threshold**2 / (abs_detail**2 + 1e-10), torch.zeros_like(detail)) * mask
            
            coeffs[i] = detail
        
        # Reconstruct signal
        reconstructed = wavelet_reconstruct(coeffs, min(decomposition_level, len(coeffs)-1))
        
        # Truncate back to original size
        result = reconstructed[:n]
        
        return (result,)



class sigmas_momentum_flow:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "momentum": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 0.99, "step": 0.01}),
                "flow_acceleration": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "barrier_strength": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "steps": ("INT", {"default": 30, "min": 10, "max": 100, "step": 1})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, sigmas, momentum, flow_acceleration, barrier_strength, steps):
        # Create a smooth path using physical momentum principles
        
        # Initial values
        start_value = sigmas[0]
        end_value = sigmas[-1]
        
        # Initialize position and velocity
        position = torch.zeros(steps, device=sigmas.device, dtype=sigmas.dtype)
        velocity = torch.zeros(steps, device=sigmas.device, dtype=sigmas.dtype)
        
        # Set initial position
        position[0] = start_value
        
        # Target direction
        target_direction = torch.sign(end_value - start_value)
        target_distance = torch.abs(end_value - start_value)
        
        # Time step
        dt = 1.0 / steps
        
        # Simulation parameters
        mass = 1.0
        
        # Run simulation
        for i in range(1, steps):
            # Time-varying acceleration (stronger near the start)
            t = i / steps
            acceleration_scale = flow_acceleration * (1 - t) + 0.1 * t
            
            # Attractive force to end point
            attractive_force = target_direction * (target_distance / steps) * acceleration_scale
            
            # Barrier repulsion (to avoid crossing the end point)
            distance_to_end = torch.abs(position[i-1] - end_value)
            repulsive_force = 0
            
            if distance_to_end < 0.1 * target_distance:
                repulsive_force = -target_direction * barrier_strength * (0.1 * target_distance / (distance_to_end + 1e-6))
            
            # Total force
            force = attractive_force + repulsive_force
            
            # Acceleration (F = ma)
            acceleration = force / mass
            
            # Update velocity with momentum
            velocity[i] = momentum * velocity[i-1] + (1 - momentum) * acceleration * dt
            
            # Update position
            position[i] = position[i-1] + velocity[i] * dt
        
        # Ensure endpoint matches exactly
        position[-1] = end_value
        
        # Optional: Interpolate the results to match the original sigmas profile
        if len(sigmas) != steps:
            result = torch.nn.functional.interpolate(
                position.unsqueeze(0).unsqueeze(0), 
                size=len(sigmas), 
                mode='linear'
            ).squeeze(0).squeeze(0)
        else:
            result = position
            
        return (result,)


class sigmas_flow_matching:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas_source": ("SIGMAS", {"forceInput": True}),
                "sigmas_target": ("SIGMAS", {"forceInput": True}),
                "interpolation_steps": ("INT", {"default": 20, "min": 5, "max": 100, "step": 1}),
                "flow_matching_type": (["ode", "sde", "probability_flow", "schrodinger_bridge"], 
                                     {"default": "probability_flow"}),
                "regularization": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 1.0, "step": 0.01}),
                "solver": (["euler", "heun", "rk4", "midpoint"], {"default": "rk4"})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, sigmas_source, sigmas_target, interpolation_steps, flow_matching_type, regularization, solver):
        # Ensure equal lengths
        if len(sigmas_source) != len(sigmas_target):
            if len(sigmas_source) > len(sigmas_target):
                sigmas_target = torch.nn.functional.interpolate(
                    sigmas_target.unsqueeze(0).unsqueeze(0), 
                    size=len(sigmas_source), 
                    mode='linear'
                ).squeeze(0).squeeze(0)
            else:
                sigmas_source = torch.nn.functional.interpolate(
                    sigmas_source.unsqueeze(0).unsqueeze(0), 
                    size=len(sigmas_target), 
                    mode='linear'
                ).squeeze(0).squeeze(0)
        
        # Create flow vector fields
        if flow_matching_type == "ode":
            # ODE flow
            velocity_field = (sigmas_target - sigmas_source)
        elif flow_matching_type == "sde":
            # SDE flow with added noise
            velocity_field = (sigmas_target - sigmas_source)
            noise_scale = torch.linspace(0.1, 0.001, len(sigmas_source))
        elif flow_matching_type == "probability_flow":
            # Probability flow
            drift = (sigmas_target - sigmas_source)
            score = torch.gradient(torch.log(sigmas_source + 1e-6))[0]
            velocity_field = drift + 0.5 * score
        elif flow_matching_type == "schrodinger_bridge":
            # Schrödinger bridge approximation
            forward_drift = (sigmas_target - sigmas_source)
            backward_drift = (sigmas_source - sigmas_target)
            velocity_field = 0.5 * (forward_drift - backward_drift)
        
        # Apply regularization
        if regularization > 0:
            # Smoothing kernel
            kernel_size = 5
            padding = kernel_size // 2
            kernel = torch.ones(1, 1, kernel_size) / kernel_size
            velocity_field_padded = torch.nn.functional.pad(velocity_field.unsqueeze(0).unsqueeze(0), 
                                                          (padding, padding), mode='reflect')
            velocity_field_smooth = torch.nn.functional.conv1d(velocity_field_padded, kernel).squeeze()
            velocity_field = (1 - regularization) * velocity_field + regularization * velocity_field_smooth
        
        # Select solver and apply
        t = torch.linspace(0, 1, interpolation_steps)
        result = torch.zeros(interpolation_steps, device=sigmas_source.device, dtype=sigmas_source.dtype)
        result[0] = sigmas_source[0]  # Initial value
        
        if solver == "euler":
            # Euler solver
            dt = 1.0 / (interpolation_steps - 1)
            current_state = sigmas_source.clone()
            for i in range(1, interpolation_steps):
                # Euler step
                if flow_matching_type == "sde":
                    # Add random term for SDE
                    noise = torch.randn_like(current_state) * noise_scale * torch.sqrt(torch.tensor(dt))
                    current_state = current_state + dt * velocity_field + noise
                else:
                    current_state = current_state + dt * velocity_field
                result[i] = current_state[0]  # Store first element in result
        
        elif solver == "rk4":
            # 4th order Runge-Kutta solver
            dt = 1.0 / (interpolation_steps - 1)
            current_state = sigmas_source.clone()
            
            for i in range(1, interpolation_steps):
                # RK4 steps
                k1 = velocity_field
                k2 = velocity_field + 0.5 * dt * k1
                k3 = velocity_field + 0.5 * dt * k2
                k4 = velocity_field + dt * k3
                
                # Update state
                current_state = current_state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
                result[i] = current_state[0]

        return (result,)


import torch
import math

class sigmas_variance_guided:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "steps": ("INT", {"default": 30, "min": 5, "max": 100, "step": 1}),
                "start_value": ("FLOAT", {"default": 10.0, "min": 0.1, "max": 50.0, "step": 0.1}),
                "end_value": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 10.0, "step": 0.01}),
                "variance_sensitivity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.01}),
                "adaptive_range": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "pad_end": ("BOOLEAN", {"default": True}),
                "normalize_output": ("BOOLEAN", {"default": False})
            },
            "optional": {
                "reference_sigmas": ("SIGMAS", {"forceInput": False})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, steps, start_value, end_value, variance_sensitivity, adaptive_range, pad_end, normalize_output, reference_sigmas=None):
        device = "cuda"
        # Initialize base linear schedule
        t = torch.linspace(0, 1, steps, device=device)
        base_schedule = start_value * (1 - t) + end_value * t
        
        # Add variance-guided adaptivity
        if reference_sigmas is not None:
            # Get device from reference sigmas
            device = reference_sigmas.device
            
            # Use reference sigmas for variance analysis
            ref_sigmas = torch.nn.functional.interpolate(
                reference_sigmas.unsqueeze(0).unsqueeze(0), 
                size=steps, 
                mode='linear'
            ).squeeze()
            
            # Calculate signal variance over windows
            window_size = max(3, int(steps * 0.1))
            padding = window_size // 2
            padded = torch.nn.functional.pad(ref_sigmas.unsqueeze(0).unsqueeze(0), (padding, padding), mode='reflect')
            
            # Calculate local variance using conv
            kernel = torch.ones(1, 1, window_size, device=padded.device) / window_size
            local_means = torch.nn.functional.conv1d(padded, kernel, padding=0).squeeze()
            
            padded_squared = torch.nn.functional.pad(ref_sigmas.unsqueeze(0).unsqueeze(0) ** 2, (padding, padding), mode='reflect')
            local_means_squared = torch.nn.functional.conv1d(padded_squared, kernel, padding=0).squeeze()
            
            local_var = local_means_squared - local_means ** 2
            local_var = torch.clamp(local_var, min=1e-8)  # Avoid numerical issues
            
            # Normalize variance to [0, 1] range
            if local_var.max() > local_var.min():
                norm_var = (local_var - local_var.min()) / (local_var.max() - local_var.min()) 
            else:
                norm_var = torch.zeros_like(local_var)
                
            # Apply variance-guided modification to schedule
            variance_factor = 1.0 + variance_sensitivity * (norm_var - 0.5) * adaptive_range
            result = base_schedule * variance_factor
            
            # Ensure endpoints remain fixed
            result[0] = start_value
            result[-1] = end_value
        else:
            # Without reference, create synthetic variance pattern
            # More variance at early and late stages, less in the middle
            synthetic_var = 4 * t * (1 - t)  # Parabolic pattern
            variance_factor = 1.0 + variance_sensitivity * (synthetic_var - 0.5) * adaptive_range
            result = base_schedule * variance_factor
            
            # Ensure endpoints remain fixed
            result[0] = start_value 
            result[-1] = end_value
        
        # Normalize output if requested
        if normalize_output:
            result = (result - result.min()) / (result.max() - result.min()) * (start_value - end_value) + end_value
            result[0] = start_value
            result[-1] = end_value
            
        # Add padding zero at the end if requested
        if pad_end:
            result = torch.cat([result, torch.tensor([0.0], device=result.device, dtype=result.dtype)])
            
        return (result,)


class sigmas_error_feedback:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "feedback_strength": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "error_threshold": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "adaptation_steps": ("INT", {"default": 5, "min": 1, "max": 20, "step": 1}),
                "preserve_total_steps": ("BOOLEAN", {"default": True}),
                "normalize_output": ("BOOLEAN", {"default": False})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, sigmas, feedback_strength, error_threshold, adaptation_steps, preserve_total_steps, normalize_output):
        # Copy input sigmas
        result = sigmas.clone()
        
        # Simulate error feedback mechanism
        for _ in range(adaptation_steps):
            # Calculate approximated error (normally this would come from actual diffusion steps)
            # Here we're using a simple heuristic based on sigma values
            estimated_error = torch.zeros_like(result)
            
            # Areas of high gradient in sigma are more error-prone
            gradient = torch.zeros_like(result)
            gradient[1:-1] = torch.abs(result[2:] - result[:-2]) / 2  # Central difference
            gradient[0] = torch.abs(result[1] - result[0])
            gradient[-1] = torch.abs(result[-1] - result[-2])
            
            # Normalize gradient
            if gradient.max() > gradient.min():
                norm_gradient = (gradient - gradient.min()) / (gradient.max() - gradient.min())
                estimated_error = norm_gradient
            
            # Apply feedback where error exceeds threshold
            error_mask = estimated_error > error_threshold
            
            if torch.any(error_mask):
                # For high error regions, add more steps (make steps smaller)
                high_error_indices = torch.where(error_mask)[0]
                
                for idx in high_error_indices:
                    if idx < len(result) - 1:
                        # Insert a new point between idx and idx+1
                        midpoint_value = (result[idx] + result[idx+1]) / 2
                        
                        # Adjust surrounding points
                        result[idx] = result[idx] * (1 - feedback_strength) + (result[idx] + feedback_strength * (result[idx] - midpoint_value))
                        result[idx+1] = result[idx+1] * (1 - feedback_strength) + (result[idx+1] + feedback_strength * (midpoint_value - result[idx+1]))
        
        # If preserving total steps, resample to original length
        if preserve_total_steps:
            result = torch.nn.functional.interpolate(
                result.unsqueeze(0).unsqueeze(0), 
                size=len(sigmas), 
                mode='linear'
            ).squeeze()
            
        # Normalize output if requested
        if normalize_output:
            start_value = result[0].item()
            end_value = result[-1].item()
            result = (result - result.min()) / (result.max() - result.min()) * (start_value - end_value) + end_value
            result[0] = start_value
            result[-1] = end_value
            
        return (result,)


class sigmas_perceptual_loss:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "steps": ("INT", {"default": 30, "min": 5, "max": 100, "step": 1}),
                "start_value": ("FLOAT", {"default": 10.0, "min": 0.1, "max": 50.0, "step": 0.1}),
                "end_value": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 10.0, "step": 0.01}),
                "perceptual_gamma": ("FLOAT", {"default": 2.2, "min": 1.0, "max": 5.0, "step": 0.1}),
                "perceptual_emphasis": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "pad_end": ("BOOLEAN", {"default": True}),
                "normalize_output": ("BOOLEAN", {"default": False})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, steps, start_value, end_value, perceptual_gamma, perceptual_emphasis, pad_end, normalize_output):
        device = "cuda"
        # Create base linear schedule
        t = torch.linspace(0, 1, steps, device=device)
        linear_schedule = start_value * (1 - t) + end_value * t
        
        # Apply perceptual scaling (similar to gamma correction in images)
        # This allocates more steps to perceptually important regions
        perceptual_t = t ** (1/perceptual_gamma)
        perceptual_schedule = start_value * (1 - perceptual_t) + end_value * perceptual_t
        
        # Blend between linear and perceptual schedules
        result = (1 - perceptual_emphasis) * linear_schedule + perceptual_emphasis * perceptual_schedule
        
        # Normalize output if requested
        if normalize_output:
            result = (result - result.min()) / (result.max() - result.min()) * (start_value - end_value) + end_value
            result[0] = start_value
            result[-1] = end_value
        
        # Add padding zero at the end if requested
        if pad_end:
            result = torch.cat([result, torch.tensor([0.0], device=device, dtype=result.dtype)])
            
        return (result,)


class sigmas_symplectic_integrator:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "steps": ("INT", {"default": 30, "min": 5, "max": 100, "step": 1}),
                "start_value": ("FLOAT", {"default": 10.0, "min": 0.1, "max": 50.0, "step": 0.1}),
                "end_value": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 10.0, "step": 0.01}),
                "integrator_type": (["verlet", "leapfrog", "stormer", "yoshida"], {"default": "verlet"}),
                "conserve_factor": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "pad_end": ("BOOLEAN", {"default": True}),
                "normalize_output": ("BOOLEAN", {"default": False})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, steps, start_value, end_value, integrator_type, conserve_factor, pad_end, normalize_output):
        device = "cuda"
        # Define a potential function (harmonic oscillator)
        def potential(x):
            return 0.5 * (x - end_value)**2
            
        # Define force function (negative gradient of potential)
        def force(x):
            return -(x - end_value)
        
        # Initialize position and velocity
        x = torch.ones(1, device=device) * start_value
        v = torch.zeros(1, device=device)
        
        # Time step
        dt = torch.tensor(1.0 / steps, device=device)
        
        # Storage for trajectory
        trajectory = [x.item()]
        
        # Apply symplectic integrator
        if integrator_type == "verlet":
            # Velocity Verlet integration
            for _ in range(steps - 1):
                # Half step in velocity
                v += 0.5 * dt * force(x)
                
                # Full step in position
                x += dt * v
                
                # Another half step in velocity
                v += 0.5 * dt * force(x)
                
                # Apply conservation constraint
                v *= (1.0 - conserve_factor * dt)  # Damping
                
                trajectory.append(x.item())
                
        elif integrator_type == "leapfrog":
            # Leapfrog integration
            v -= 0.5 * dt * force(x)  # Initial half-step backward
            
            for _ in range(steps - 1):
                # Full position step
                x += dt * v
                
                # Full velocity step
                v -= dt * force(x)
                
                # Apply conservation constraint
                v *= (1.0 - conserve_factor * dt)  # Damping
                
                trajectory.append(x.item())
                
        elif integrator_type == "stormer":
            # Störmer-Verlet integration (position-focused)
            x_prev = x.clone()
            x_curr = x + dt * v + 0.5 * dt**2 * force(x)
            trajectory.append(x_curr.item())
            
            for _ in range(steps - 2):
                # Update using Störmer formula
                x_next = 2 * x_curr - x_prev + dt**2 * force(x_curr)
                
                # Apply conservation
                x_next = x_next * (1 - conserve_factor) + (x_curr + (x_curr - x_prev)) * conserve_factor
                
                x_prev = x_curr
                x_curr = x_next
                
                trajectory.append(x_curr.item())
                
        elif integrator_type == "yoshida":
            # Yoshida 4th order integrator
            w0 = -(2**(1/3)) / (2 - 2**(1/3))
            w1 = 1 / (2 - 2**(1/3))
            
            c = [w1/2, (w0+w1)/2, (w0+w1)/2, w1/2]
            d = [w1, w0, w1, 0]
            
            for _ in range(steps - 1):
                for i in range(4):
                    # Position half-step
                    x += c[i] * dt * v
                    
                    # Velocity step
                    v += d[i] * dt * force(x)
                
                # Apply conservation constraint
                v *= (1.0 - conserve_factor * dt)
                
                trajectory.append(x.item())
        
        # Convert to tensor
        result = torch.tensor(trajectory, device=device)
        
        # Make sure end point is exactly end_value
        result[-1] = end_value
        
        # Normalize output if requested
        if normalize_output:
            result = (result - result.min()) / (result.max() - result.min()) * (start_value - end_value) + end_value
            result[0] = start_value
            result[-1] = end_value
        
        # Add padding zero at the end if requested
        if pad_end:
            result = torch.cat([result, torch.tensor([0.0], device=device, dtype=result.dtype)])
            
        return (result,)


class sigmas_accelerated_gradient:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "steps": ("INT", {"default": 30, "min": 5, "max": 100, "step": 1}),
                "start_value": ("FLOAT", {"default": 10.0, "min": 0.1, "max": 50.0, "step": 0.1}),
                "end_value": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 10.0, "step": 0.01}),
                "acceleration": (["nesterov", "adagrad", "rmsprop", "momentum"], {"default": "nesterov"}),
                "momentum_factor": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 0.99, "step": 0.01}),
                "pad_end": ("BOOLEAN", {"default": True}),
                "normalize_output": ("BOOLEAN", {"default": False})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, steps, start_value, end_value, acceleration, momentum_factor, pad_end, normalize_output):
        device = "cuda"
        # Define the objective function (minimize distance to end_value)
        def objective(x):
            return (x - end_value)**2
            
        def gradient(x):
            return 2 * (x - end_value)
        
        # Initialize position
        x = torch.ones(1, device=device) * start_value
        
        # Initialize algorithm-specific variables
        velocity = torch.zeros(1, device=device)
        acc_gradient = torch.zeros(1, device=device)
        
        # Learning rate
        lr = torch.tensor((start_value - end_value) / steps, device=device)
        
        # Storage for trajectory
        trajectory = [x.item()]
        
        # Apply accelerated gradient method
        if acceleration == "nesterov":
            # Nesterov Accelerated Gradient
            for i in range(steps - 1):
                # Compute β for the current iteration
                beta = torch.tensor(3.0 / (5.0 + i), device=device)  # Adaptive scheme
                
                # Lookahead step (Nesterov's trick)
                x_lookahead = x - beta * velocity
                
                # Update velocity with lookahead gradient
                velocity = momentum_factor * velocity + lr * gradient(x_lookahead)
                
                # Update position
                x = x - velocity
                
                trajectory.append(x.item())
                
        elif acceleration == "adagrad":
            # Adagrad
            for _ in range(steps - 1):
                # Compute gradient
                grad = gradient(x)
                
                # Accumulate squared gradients
                acc_gradient += grad**2
                
                # Update position with adaptive learning rate
                x = x - lr * grad / torch.sqrt(acc_gradient + 1e-8)
                
                trajectory.append(x.item())
                
        elif acceleration == "rmsprop":
            # RMSProp
            decay_rate = torch.tensor(0.9, device=device)
            
            for _ in range(steps - 1):
                # Compute gradient
                grad = gradient(x)
                
                # Update accumulated gradient
                acc_gradient = decay_rate * acc_gradient + (1 - decay_rate) * grad**2
                
                # Update position
                x = x - lr * grad / torch.sqrt(acc_gradient + 1e-8)
                
                trajectory.append(x.item())
                
        elif acceleration == "momentum":
            # Standard Momentum
            for _ in range(steps - 1):
                # Compute gradient
                grad = gradient(x)
                
                # Update velocity
                velocity = momentum_factor * velocity + lr * grad
                
                # Update position
                x = x - velocity
                
                trajectory.append(x.item())
        
        # Convert to tensor
        result = torch.tensor(trajectory, device=device)
        
        # Make sure end point is exactly end_value
        result[-1] = end_value
        
        # Normalize output if requested
        if normalize_output:
            result = (result - result.min()) / (result.max() - result.min()) * (start_value - end_value) + end_value
            result[0] = start_value
            result[-1] = end_value
        
        # Add padding zero at the end if requested
        if pad_end:
            result = torch.cat([result, torch.tensor([0.0], device=device, dtype=result.dtype)])
            
        return (result,)


class sigmas_phase_space:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "steps": ("INT", {"default": 30, "min": 5, "max": 100, "step": 1}),
                "start_value": ("FLOAT", {"default": 10.0, "min": 0.1, "max": 50.0, "step": 0.1}),
                "end_value": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 10.0, "step": 0.01}),
                "phase_strategy": (["hamiltonian", "canonical", "action_angle"], {"default": "hamiltonian"}),
                "energy_conservation": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "pad_end": ("BOOLEAN", {"default": True}),
                "normalize_output": ("BOOLEAN", {"default": False})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, steps, start_value, end_value, phase_strategy, energy_conservation, pad_end, normalize_output):
        device = "cuda"
        # Define Hamiltonian energy function
        def hamiltonian(q, p):
            # Kinetic energy + potential energy
            return 0.5 * p**2 + 0.5 * (q - end_value)**2
            
        # Position and momentum partial derivatives
        def dq_dt(q, p):
            return p  # dH/dp
            
        def dp_dt(q, p):
            return -(q - end_value)  # -dH/dq
        
        # Initialize position and momentum
        q = torch.ones(1, device=device) * start_value
        
        # Set initial momentum for different strategies
        if phase_strategy == "hamiltonian":
            # Standard Hamiltonian trajectory
            p = torch.zeros(1, device=device)
        elif phase_strategy == "canonical":
            # Canonical ensemble (thermodynamic)
            p = torch.randn(1, device=device) * torch.sqrt(torch.abs(torch.tensor(start_value - end_value, device=device)))
        elif phase_strategy == "action_angle":
            # Action-angle variables (oscillatory)
            p = torch.sin(torch.tensor([math.pi/4], device=device)) * (start_value - end_value)
            
        # Time step
        dt = torch.tensor(1.0 / steps, device=device)
        
        # Initial energy
        initial_energy = hamiltonian(q, p)
        
        # Storage for trajectory
        trajectory = [q.item()]
        
        # Symplectic integration in phase space
        for i in range(1, steps):
            # Update position and momentum using leapfrog integration
            p = p + 0.5 * dt * dp_dt(q, p)
            q = q + dt * dq_dt(q, p)
            p = p + 0.5 * dt * dp_dt(q, p)
            
            # Energy conservation constraint
            current_energy = hamiltonian(q, p)
            if current_energy > 0 and initial_energy > 0:
                # Scale momentum to preserve energy (partially)
                energy_ratio = (initial_energy / current_energy) ** 0.5
                energy_ratio = 1.0 + (energy_ratio - 1.0) * energy_conservation
                p = p * energy_ratio
                
            # Track position
            trajectory.append(q.item())
            
            # Gradually guide to the end value in later steps
            if i > steps * 0.8:
                # Add a small force directing towards end_value
                force_factor = (i - steps * 0.8) / (steps * 0.2)
                q = q * (1 - 0.1 * force_factor) + end_value * 0.1 * force_factor
        
        # Convert to tensor
        result = torch.tensor(trajectory, device=device)
        
        # Make sure end point is exactly end_value
        result[-1] = end_value
        
        # Normalize output if requested
        if normalize_output:
            result = (result - result.min()) / (result.max() - result.min()) * (start_value - end_value) + end_value
            result[0] = start_value
            result[-1] = end_value
        
        # Add padding zero at the end if requested
        if pad_end:
            result = torch.cat([result, torch.tensor([0.0], device=device, dtype=result.dtype)])
            
        return (result,)


class sigmas_reverse_kl:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "steps": ("INT", {"default": 30, "min": 5, "max": 100, "step": 1}),
                "start_value": ("FLOAT", {"default": 10.0, "min": 0.1, "max": 50.0, "step": 0.1}),
                "end_value": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 10.0, "step": 0.01}),
                "alpha": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "beta": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "pad_end": ("BOOLEAN", {"default": True}),
                "normalize_output": ("BOOLEAN", {"default": False})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, steps, start_value, end_value, alpha, beta, pad_end, normalize_output):
        device = "cuda"
        # Create time steps
        t = torch.linspace(0, 1, steps, device=device)
        
        # Standard forward KL divergence path (quadratic)
        forward_kl = start_value * (1 - t)**2 + end_value * (1 - (1 - t)**2)
        
        # Reverse KL path (focuses more on mode coverage)
        # Using a skewed logistic function to approximate reverse KL optimal path
        reverse_kl = start_value * (1 - torch.sigmoid((t - alpha) * beta)) + end_value * torch.sigmoid((t - alpha) * beta)
        
        # Hybrid approach: combine both paths
        # Early process: more focus on forward KL
        # Later process: more focus on reverse KL
        blending = torch.sigmoid(5 * (t - 0.5))  # Sigmoid blending factor
        result = (1 - blending) * forward_kl + blending * reverse_kl
        
        # Normalize output if requested
        if normalize_output:
            result = (result - result.min()) / (result.max() - result.min()) * (start_value - end_value) + end_value
            result[0] = start_value
            result[-1] = end_value
        
        # Add padding zero at the end if requested
        if pad_end:
            result = torch.cat([result, torch.tensor([0.0], device=device, dtype=result.dtype)])
            
        return (result,)


class sigmas_mutual_information:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "steps": ("INT", {"default": 30, "min": 5, "max": 100, "step": 1}),
                "start_value": ("FLOAT", {"default": 10.0, "min": 0.1, "max": 50.0, "step": 0.1}),
                "end_value": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 10.0, "step": 0.01}),
                "information_rate": (["constant", "decreasing", "oscillatory"], {"default": "decreasing"}),
                "rate_parameter": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 2.0, "step": 0.01}),
                "pad_end": ("BOOLEAN", {"default": True}),
                "normalize_output": ("BOOLEAN", {"default": False})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, steps, start_value, end_value, information_rate, rate_parameter, pad_end, normalize_output):
        device = "cuda"
        # Initialize result
        result = torch.zeros(steps, device=device)
        result[0] = start_value
        
        # Information transport rate determines step sizes
        if information_rate == "constant":
            # Constant information change per step - often exponential curve
            # mutual information ~ log(sigma_t/sigma_{t+1})
            log_ratio = torch.log(torch.tensor(start_value / end_value, device=device))
            info_per_step = log_ratio / (steps - 1)
            
            for i in range(1, steps):
                result[i] = result[i-1] * torch.exp(-info_per_step)
                
        elif information_rate == "decreasing":
            # Decreasing information rate - more steps early on
            # Rate decreases according to power law
            t = torch.linspace(0, 1, steps, device=device)
            decay_factor = t ** rate_parameter
            
            # Normalize to ensure we reach end_value
            normalized_decay = decay_factor / decay_factor.sum()
            cumulative_info = torch.cumsum(normalized_decay, dim=0) * torch.log(torch.tensor(start_value / end_value, device=device))
            
            result = start_value * torch.exp(-cumulative_info)
            
        elif information_rate == "oscillatory":
            # Oscillatory information rate - alternating between high and low info steps
            base_log_ratio = torch.log(torch.tensor(start_value / end_value, device=device)) / (steps - 1)
            
            for i in range(1, steps):
                # Add oscillation to information rate
                oscillation = 0.5 * rate_parameter * torch.sin(torch.tensor(2 * math.pi * i / (steps / 5), device=device))
                adjusted_ratio = base_log_ratio * (1 + oscillation)
                
                # Update with adjusted ratio
                result[i] = result[i-1] * torch.exp(-adjusted_ratio)
                
            # Re-normalize to ensure endpoints match
            result = start_value * (result / result[0])
            result[-1] = end_value  # Ensure exact match
        
        # Normalize output if requested
        if normalize_output:
            result = (result - result.min()) / (result.max() - result.min()) * (start_value - end_value) + end_value
            result[0] = start_value
            result[-1] = end_value
            
        # Add padding zero at the end if requested
        if pad_end:
            result = torch.cat([result, torch.tensor([0.0], device=device, dtype=result.dtype)])
            
        return (result,)


class sigmas_cross_entropy:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "steps": ("INT", {"default": 30, "min": 5, "max": 100, "step": 1}),
                "start_value": ("FLOAT", {"default": 10.0, "min": 0.1, "max": 50.0, "step": 0.1}),
                "end_value": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 10.0, "step": 0.01}),
                "cross_entropy_focus": (["precision", "recall", "balanced"], {"default": "balanced"}),
                "robustness": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "pad_end": ("BOOLEAN", {"default": True}),
                "normalize_output": ("BOOLEAN", {"default": False})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, steps, start_value, end_value, cross_entropy_focus, robustness, pad_end, normalize_output):
        device = "cuda"
        # Create time steps
        t = torch.linspace(0, 1, steps, device=device)
        
        if cross_entropy_focus == "precision":
            # Focus on precision (harder denoising steps at the beginning)
            # Cross-entropy optimal for precision uses higher noise levels longer
            # a*(1-t)^b form with b < 1
            b = 0.7 - 0.4 * robustness  # Lower b = more focus on precision
            sigma_values = start_value * (1 - t)**b + end_value * (1 - (1 - t)**b)
            
        elif cross_entropy_focus == "recall":
            # Focus on recall (harder denoising steps at the end)
            # Cross-entropy optimal for recall uses faster noise reduction early
            # Uses a sigmoid-based transition
            midpoint = 0.7 - 0.4 * robustness  # Higher midpoint = more focus on recall
            slope = 8 + 4 * robustness
            sigma_values = start_value * (1 - torch.sigmoid(slope * (t - midpoint))) + end_value * torch.sigmoid(slope * (t - midpoint))
            
        else:  # balanced
            # Balanced approach (balanced precision and recall)
            # Uses a smoother transition, approximating optimal cross-entropy path
            # Approximated by beta distribution CDF
            alpha = 2 - robustness
            beta_param = 2 - robustness
            
            # Approximation of beta CDF using incomplete beta function
            # Using a simpler approximation with torch
            normalized_t = torch.clamp(t, 0.001, 0.999)  # Avoid extremes for numerical stability
            beta_cdf = torch.pow(normalized_t, alpha) / (torch.pow(normalized_t, alpha) + torch.pow(1 - normalized_t, beta_param))
            
            sigma_values = start_value * (1 - beta_cdf) + end_value * beta_cdf
        
        # Normalize output if requested
        if normalize_output:
            sigma_values = (sigma_values - sigma_values.min()) / (sigma_values.max() - sigma_values.min()) * (start_value - end_value) + end_value
            sigma_values[0] = start_value
            sigma_values[-1] = end_value
        
        # Add padding zero at the end if requested
        if pad_end:
            sigma_values = torch.cat([sigma_values, torch.tensor([0.0], device=device, dtype=sigma_values.dtype)])
            
        return (sigma_values,)


class sigmas_manifold_preserving:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "steps": ("INT", {"default": 30, "min": 5, "max": 100, "step": 1}),
                "start_value": ("FLOAT", {"default": 10.0, "min": 0.1, "max": 50.0, "step": 0.1}),
                "end_value": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 10.0, "step": 0.01}),
                "manifold_dim": ("INT", {"default": 2, "min": 1, "max": 10, "step": 1}),
                "curvature": ("FLOAT", {"default": 0.2, "min": -1.0, "max": 1.0, "step": 0.01}),
                "pad_end": ("BOOLEAN", {"default": True}),
                "normalize_output": ("BOOLEAN", {"default": False})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, steps, start_value, end_value, manifold_dim, curvature, pad_end, normalize_output):
        device = "cuda"
        # Create normalized parameter space
        t = torch.linspace(0, 1, steps, device=device)
        
        # Calculate effective dimensionality factor
        # Higher dimensions require special handling to preserve manifold structure
        dim_factor = torch.sqrt(torch.tensor(manifold_dim, dtype=torch.float32, device=device))
        
        if curvature > 0:
            # Positive curvature (spherical-like manifold)
            # Use spherical coordinates transformation
            # Adapted path length accounting for positive curvature
            theta = t * torch.pi/2  # Angle from 0 to π/2
            radius = torch.sin(theta * (1 + curvature * 0.5))
            sigma_values = start_value * (1 - radius) + end_value * radius
            
        elif curvature < 0:
            # Negative curvature (hyperbolic-like manifold)
            # Use hyperbolic transformation
            # Path is longer in negatively curved space
            # Convert curvature to tensor
            curv_tensor = torch.tensor(abs(curvature), device=device)
            sinh_factor = torch.sinh(t * dim_factor * curv_tensor)
            normalized = sinh_factor / sinh_factor[-1]  # Normalize to [0,1]
            sigma_values = start_value * (1 - normalized) + end_value * normalized
            
        else:
            # Zero curvature (flat manifold)
            # Standard interpolation with dimension scaling
            # Higher dimensions need more even spacing due to curse of dimensionality
            power = 1.0 / dim_factor
            normalized = torch.pow(t, power)
            sigma_values = start_value * (1 - normalized) + end_value * normalized
        
        # Normalize output if requested
        if normalize_output:
            sigma_values = (sigma_values - sigma_values.min()) / (sigma_values.max() - sigma_values.min()) * (start_value - end_value) + end_value
            sigma_values[0] = start_value
            sigma_values[-1] = end_value
        
        # Add padding zero at the end if requested
        if pad_end:
            sigma_values = torch.cat([sigma_values, torch.tensor([0.0], device=device, dtype=sigma_values.dtype)])
            
        return (sigma_values,)


class sigmas_curvature_geodesic:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "steps": ("INT", {"default": 30, "min": 5, "max": 100, "step": 1}),
                "start_value": ("FLOAT", {"default": 10.0, "min": 0.1, "max": 50.0, "step": 0.1}),
                "end_value": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 10.0, "step": 0.01}),
                "metric_type": (["euclidean", "riemannian", "lognormal", "wasserstein"], {"default": "riemannian"}),
                "curvature_weight": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "pad_end": ("BOOLEAN", {"default": True}),
                "normalize_output": ("BOOLEAN", {"default": False})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, steps, start_value, end_value, metric_type, curvature_weight, pad_end, normalize_output):
        device = "cuda"
        # Create parameter space
        t = torch.linspace(0, 1, steps, device=device)
        
        if metric_type == "euclidean":
            # Standard Euclidean geodesic (straight line)
            sigma_values = start_value * (1 - t) + end_value * t
            
        elif metric_type == "riemannian":
            # Riemannian metric for diffusion models
            # This uses a custom metric where distance is measured by log ratio
            # Geodesic follows exponential decay
            log_ratio = torch.log(torch.tensor(start_value / end_value, device=device))
            sigma_values = start_value * torch.exp(-t * log_ratio)
            
        elif metric_type == "lognormal":
            # Log-normal metric (natural for variance parameters)
            # Geodesic in this space follows a geometric sequence
            log_start = torch.log(torch.tensor(start_value, device=device))
            log_end = torch.log(torch.tensor(end_value, device=device))
            sigma_values = torch.exp(log_start * (1 - t) + log_end * t)
            
        elif metric_type == "wasserstein":
            # 2-Wasserstein metric (optimal transport)
            # Square root interpolation of variance parameters
            sqrt_start = torch.sqrt(torch.tensor(start_value, device=device))
            sqrt_end = torch.sqrt(torch.tensor(end_value, device=device))
            sigma_values = (sqrt_start * (1 - t) + sqrt_end * t) ** 2
        
        # Apply curvature-aware adjustment (focusing more on high-curvature regions)
        if curvature_weight > 0:
            # Compute approximate scalar curvature in the path
            # Second derivative approximation
            curvature = torch.zeros_like(sigma_values)
            curvature[1:-1] = torch.abs(sigma_values[2:] - 2 * sigma_values[1:-1] + sigma_values[:-2])
            curvature[0] = curvature[1]
            curvature[-1] = curvature[-2]
            
            # Normalize curvature
            norm_curvature = curvature / (curvature.max() + 1e-8)
            
            # Adaptive step size based on curvature
            # More steps in high curvature regions
            step_size_weights = 1.0 / (1.0 + curvature_weight * norm_curvature)
            step_size_weights = step_size_weights / step_size_weights.sum()
            
            # Compute adjusted t values
            t_adjusted = torch.cumsum(step_size_weights, dim=0)
            t_adjusted = t_adjusted / t_adjusted[-1]  # Normalize to [0,1]
            
            # Recompute sigma values with adjusted t
            if metric_type == "euclidean":
                sigma_values = start_value * (1 - t_adjusted) + end_value * t_adjusted
            elif metric_type == "riemannian":
                sigma_values = start_value * torch.exp(-t_adjusted * log_ratio)
            elif metric_type == "lognormal":
                sigma_values = torch.exp(log_start * (1 - t_adjusted) + log_end * t_adjusted)
            elif metric_type == "wasserstein":
                sigma_values = (sqrt_start * (1 - t_adjusted) + sqrt_end * t_adjusted) ** 2
        
        # Normalize output if requested
        if normalize_output:
            sigma_values = (sigma_values - sigma_values.min()) / (sigma_values.max() - sigma_values.min()) * (start_value - end_value) + end_value
            sigma_values[0] = start_value
            sigma_values[-1] = end_value
        
        # Add padding zero at the end if requested
        if pad_end:
            sigma_values = torch.cat([sigma_values, torch.tensor([0.0], device=device, dtype=sigma_values.dtype)])
            
        return (sigma_values,)
    
    
class sigmas_persistent_homology:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "steps": ("INT", {"default": 30, "min": 5, "max": 100, "step": 1}),
                "start_value": ("FLOAT", {"default": 10.0, "min": 0.1, "max": 50.0, "step": 0.1}),
                "end_value": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 10.0, "step": 0.01}),
                "persistence_type": (["linear", "exponential", "logarithmic", "critical_points"], {"default": "critical_points"}),
                "persistence_param": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "pad_end": ("BOOLEAN", {"default": True}),
                "normalize_output": ("BOOLEAN", {"default": False})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, steps, start_value, end_value, persistence_type, persistence_param, pad_end, normalize_output):
        device = "cuda"
        # Initialize result
        result = torch.zeros(steps, device=device)
        
        if persistence_type == "linear":
            # Linear filtration with persistence-based weighting
            t = torch.linspace(0, 1, steps, device=device)
            result = start_value * (1 - t) + end_value * t
            
        elif persistence_type == "exponential":
            # Exponential decay filtration (standard in TDA)
            log_ratio = torch.log(torch.tensor(start_value / end_value, device=device))
            t = torch.linspace(0, 1, steps, device=device)
            result = start_value * torch.exp(-t * log_ratio)
            
        elif persistence_type == "logarithmic":
            # Logarithmic filtration (slower initial decay)
            t = torch.linspace(0, 1, steps, device=device)
            # log(1+t)/log(2) ranges from 0 to 1 as t goes from 0 to 1
            log_t = torch.log(1 + t) / torch.log(torch.tensor(2.0, device=device))
            result = start_value * (1 - log_t) + end_value * log_t
            
        elif persistence_type == "critical_points":
            # Critical points filtration based on persistence diagram
            # Simulates persistence diagram with more critical points at specific scales
            
            # Create persistence diagram (birth, death) pairs
            # These are key scales where topological features appear/disappear
            n_critical = int(5 + 10 * persistence_param)  # Number of critical points
            
            # Simple simulation of persistence diagram with controlled distribution
            persistence_points = []
            
            # Add some fixed critical points
            persistence_points.append((0.8, 0.6))  # High persistence feature at 0.8 scale
            persistence_points.append((0.5, 0.4))  # Medium persistence feature at 0.5 scale
            persistence_points.append((0.2, 0.15)) # Low persistence feature at 0.2 scale
            
            # Add random critical points with persistence-dependent distribution
            for _ in range(n_critical - 3):
                # Birth time biased by persistence parameter
                birth = 0.1 + 0.8 * (torch.rand(1, device=device).item() ** (1.5 - persistence_param))
                # Death time proportional to birth (higher persistence at higher scales)
                persistence = 0.05 + 0.15 * torch.rand(1, device=device).item() * birth
                death = birth - persistence
                death = max(0.01, death)  # Ensure death > 0
                persistence_points.append((birth, death))
            
            # Convert to tensor format
            birth_times = torch.tensor([p[0] for p in persistence_points], device=device)
            death_times = torch.tensor([p[1] for p in persistence_points], device=device)
            
            # Create filtration guided by persistence diagram
            t = torch.linspace(0, 1, steps, device=device)
            
            # Reshape filtration based on critical points
            filtration = torch.zeros_like(t)
            
            for i, ti in enumerate(t):
                # Convert normalized t to scale parameter
                scale = 1.0 - ti.item()  # Higher scale at beginning
                
                # Count features alive at this scale
                alive_count = ((birth_times >= scale) & (death_times <= scale)).float().sum()
                
                # Normalize to [0,1] by expected maximum
                filtration[i] = torch.clamp(alive_count / n_critical, 0.0, 1.0)
            
            # Convert to sigma values
            result = start_value * (1 - filtration) + end_value * filtration
            
        # Ensure endpoints match exactly
        result[0] = start_value
        result[-1] = end_value
        
        # Normalize output if requested
        if normalize_output:
            result = (result - result.min()) / (result.max() - result.min()) * (start_value - end_value) + end_value
            result[0] = start_value
            result[-1] = end_value
        
        # Add padding zero at the end if requested
        if pad_end:
            result = torch.cat([result, torch.tensor([0.0], device=device, dtype=result.dtype)])
            
        return (result,)


# Hybrid Functions

class sigmas_multi_resolution:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "steps": ("INT", {"default": 30, "min": 5, "max": 100, "step": 1}),
                "start_value": ("FLOAT", {"default": 10.0, "min": 0.1, "max": 50.0, "step": 0.1}),
                "end_value": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 10.0, "step": 0.01}),
                "cascade_levels": ("INT", {"default": 3, "min": 1, "max": 5, "step": 1}),
                "level_ratios": ("STRING", {"default": "0.5,0.3,0.2", "multiline": False}),
                "pad_end": ("BOOLEAN", {"default": True}),
                "normalize_output": ("BOOLEAN", {"default": False})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, steps, start_value, end_value, cascade_levels, level_ratios, pad_end, normalize_output):
        device = "cuda"
        # Parse level ratios
        try:
            ratios = [float(x) for x in level_ratios.split(',')]
            # Ensure we have enough ratios for the cascade levels
            if len(ratios) < cascade_levels:
                ratios.extend([0.1] * (cascade_levels - len(ratios)))
            # Normalize ratios to sum to 1
            ratios = torch.tensor(ratios[:cascade_levels], device=device)
            ratios = ratios / ratios.sum()
        except:
            # Default fallback if parsing fails
            ratios = torch.ones(cascade_levels, device=device) / cascade_levels
        
        # Calculate steps per level
        steps_per_level = [max(2, int(steps * r.item())) for r in ratios]
        
        # Adjust to match total steps
        while sum(steps_per_level) > steps:
            # Remove from the level with the most steps
            max_idx = steps_per_level.index(max(steps_per_level))
            steps_per_level[max_idx] -= 1
        
        while sum(steps_per_level) < steps:
            # Add to the level with the least steps
            min_idx = steps_per_level.index(min(steps_per_level))
            steps_per_level[min_idx] += 1
        
        # Calculate intermediate values between levels
        level_values = torch.zeros(cascade_levels + 1, device=device)
        level_values[0] = start_value
        level_values[-1] = end_value
        
        if cascade_levels > 1:
            # Calculate intermediate pivot points
            # Using a logarithmic distribution for better sigma spacing
            log_start = torch.log(torch.tensor(start_value, device=device))
            log_end = torch.log(torch.tensor(end_value, device=device))
            log_range = log_start - log_end
            
            # Place intermediate levels at logarithmically spaced points
            for i in range(1, cascade_levels):
                position = i / cascade_levels
                # Bias toward spending more steps at higher noise levels
                adjusted_pos = position ** 1.5
                log_val = log_start - adjusted_pos * log_range
                level_values[i] = torch.exp(log_val)
        
        # Create multi-resolution cascade
        result = torch.zeros(steps, device=device)
        current_idx = 0
        
        for level in range(cascade_levels):
            level_step_count = steps_per_level[level]
            start_val = level_values[level].item()
            end_val = level_values[level + 1].item()
            
            # Choose different interpolation strategy per level
            if level == 0:
                # First level - more steps early (capturing large structures)
                # Cosine interpolation
                t = torch.linspace(0, 1, level_step_count, device=device)
                level_sigmas = start_val * torch.cos(t * torch.pi/2) + end_val * (1 - torch.cos(t * torch.pi/2))
            elif level == cascade_levels - 1:
                # Last level - more steps late (fine details)
                # Power function
                t = torch.linspace(0, 1, level_step_count, device=device)
                power = torch.tensor(0.7, device=device)
                level_sigmas = start_val * (1 - t**power) + end_val * t**power
            else:
                # Middle levels - balanced
                # Linear interpolation
                t = torch.linspace(0, 1, level_step_count, device=device)
                level_sigmas = start_val * (1 - t) + end_val * t
            
            # Add to result
            result[current_idx:current_idx + level_step_count] = level_sigmas
            current_idx += level_step_count
        
        # Normalize output if requested
        if normalize_output:
            result = (result - result.min()) / (result.max() - result.min()) * (start_value - end_value) + end_value
            result[0] = start_value
            result[-1] = end_value
        
        # Add padding zero at the end if requested
        if pad_end:
            result = torch.cat([result, torch.tensor([0.0], device=device, dtype=result.dtype)])
            
        return (result,)


class sigmas_sde_ode_hybrid:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "steps": ("INT", {"default": 30, "min": 5, "max": 100, "step": 1}),
                "start_value": ("FLOAT", {"default": 10.0, "min": 0.1, "max": 50.0, "step": 0.1}),
                "end_value": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 10.0, "step": 0.01}),
                "transition_point": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 0.9, "step": 0.01}),
                "sde_coefficient": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
                "pad_end": ("BOOLEAN", {"default": True}),
                "normalize_output": ("BOOLEAN", {"default": False})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, steps, start_value, end_value, transition_point, sde_coefficient, pad_end, normalize_output):
        device = "cuda"
        # Convert transition point to step index
        transition_step = int(steps * transition_point)
        transition_step = max(1, min(transition_step, steps - 1))  # Ensure valid range
        
        # Create time steps
        t = torch.linspace(0, 1, steps, device=device)
        
        # SDE portion (first part) - with noise term
        # Using Variance Preserving SDE formulation
        def sde_drift(t_val, sigma_val):
            return -0.5 * sigma_val / (1 - t_val)
        
        def sde_diffusion(t_val):
            return torch.sqrt(torch.tensor(sde_coefficient, device=device))
        
        # ODE portion (second part) - deterministic
        # Using probability flow ODE
        def ode_drift(t_val, sigma_val):
            return -sigma_val / (1 - t_val)
        
        # Initialize with start value
        sigma_values = torch.zeros(steps, device=device)
        sigma_values[0] = start_value
        
        # SDE integration for first portion
        for i in range(1, transition_step):
            dt = t[i] - t[i-1]
            
            # Euler-Maruyama method for SDE
            drift = sde_drift(t[i-1], sigma_values[i-1]) * dt
            diffusion = sde_diffusion(t[i-1]) * torch.sqrt(dt) * torch.randn(1, device=device)
            
            sigma_values[i] = sigma_values[i-1] + drift + diffusion
        
        # Compute intermediate value at transition point
        transition_value = sigma_values[transition_step-1]
        
        # Determine target end value for ODE portion
        # Ensure smooth transition by matching expected trajectory
        expected_drift = sde_drift(t[transition_step-1], transition_value)
        expected_next = transition_value + expected_drift * (t[transition_step] - t[transition_step-1])
        
        # ODE integration for second portion (probability flow)
        # Set up ODE with adjusted end point
        t_ode = (t[transition_step:] - t[transition_step]) / (1 - t[transition_step])
        
        # Exponential solution to the ODE
        remaining_steps = steps - transition_step
        log_ratio = torch.log(torch.tensor(expected_next / end_value, device=device))
        
        ode_values = expected_next * torch.exp(-t_ode * log_ratio)
        sigma_values[transition_step:] = ode_values
        
        # Ensure end value is exactly as specified
        sigma_values[-1] = end_value
        
        # Normalize output if requested
        if normalize_output:
            sigma_values = (sigma_values - sigma_values.min()) / (sigma_values.max() - sigma_values.min()) * (start_value - end_value) + end_value
            sigma_values[0] = start_value
            sigma_values[-1] = end_value
        
        # Add padding zero at the end if requested
        if pad_end:
            sigma_values = torch.cat([sigma_values, torch.tensor([0.0], device=device, dtype=sigma_values.dtype)])
            
        return (sigma_values,)


class sigmas_progressive_distillation:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "steps": ("INT", {"default": 30, "min": 5, "max": 100, "step": 1}),
                "start_value": ("FLOAT", {"default": 10.0, "min": 0.1, "max": 50.0, "step": 0.1}),
                "end_value": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 10.0, "step": 0.01}),
                "distillation_rounds": ("INT", {"default": 3, "min": 1, "max": 5, "step": 1}),
                "distillation_alpha": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "pad_end": ("BOOLEAN", {"default": True}),
                "normalize_output": ("BOOLEAN", {"default": False})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, steps, start_value, end_value, distillation_rounds, distillation_alpha, pad_end, normalize_output):
        device = "cuda"
        # Start with a basic cosine noise schedule
        t = torch.linspace(0, 1, steps, device=device)
        base_schedule = start_value * torch.cos(t * torch.pi/2) + end_value * (1 - torch.cos(t * torch.pi/2))
        
        # Progressive distillation simulation
        current_schedule = base_schedule
        
        for r in range(distillation_rounds):
            # Teacher model (current schedule)
            teacher_schedule = current_schedule.clone()
            
            # Student model (trying to learn from half the steps)
            # In real distillation, the student would have half the step count
            # Here we're simulating the effect on sigma values
            
            # Compute distilled points
            student_points = []
            
            # For each step in the student model
            for i in range(0, steps, 2):
                if i + 1 < steps:
                    # Progressive distillation tends to merge consecutive steps
                    # Weighted average of consecutive teacher steps
                    i_alpha = distillation_alpha + (1 - distillation_alpha) * (r / distillation_rounds)
                    merged_sigma = teacher_schedule[i] * i_alpha + teacher_schedule[i+1] * (1 - i_alpha)
                    student_points.append(merged_sigma.item())
                else:
                    # Handle odd number of steps
                    student_points.append(teacher_schedule[i].item())
            
            # Interpolate student back to full step count
            student_schedule = torch.tensor(student_points, device=device)
            student_full = torch.nn.functional.interpolate(
                student_schedule.unsqueeze(0).unsqueeze(0), 
                size=steps, 
                mode='linear'
            ).squeeze()
            
            # Update current schedule with student
            # Each round, the student has more influence
            round_alpha = 0.6 ** (r + 1)  # Exponential decay of teacher influence
            current_schedule = teacher_schedule * round_alpha + student_full * (1 - round_alpha)
            
            # Apply knowledge distillation adjustments
            # Distillation tends to make schedules more efficient
            # Lower values in early steps, higher in late steps
            t_adj = (1 - t) ** (1.1 + 0.1 * r)  # Progressively more aggressive adjustment
            adjustment = 0.1 * (r + 1) / distillation_rounds  # Increased effect with rounds
            
            current_schedule = current_schedule * (1 - adjustment * t_adj)
            
            # Ensure endpoints remain fixed
            current_schedule[0] = start_value
            current_schedule[-1] = end_value
        
        # Normalize output if requested
        if normalize_output:
            current_schedule = (current_schedule - current_schedule.min()) / (current_schedule.max() - current_schedule.min()) * (start_value - end_value) + end_value
            current_schedule[0] = start_value
            current_schedule[-1] = end_value
        
        # Add padding zero at the end if requested
        if pad_end:
            current_schedule = torch.cat([current_schedule, torch.tensor([0.0], device=device, dtype=current_schedule.dtype)])
            
        return (current_schedule,)


# Quantum-inspired approaches

class sigmas_schrodinger_bridge:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "steps": ("INT", {"default": 30, "min": 5, "max": 100, "step": 1}),
                "start_value": ("FLOAT", {"default": 10.0, "min": 0.1, "max": 50.0, "step": 0.1}),
                "end_value": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 10.0, "step": 0.01}),
                "bridge_iterations": ("INT", {"default": 3, "min": 1, "max": 10, "step": 1}),
                "convergence_rate": ("FLOAT", {"default": 0.7, "min": 0.1, "max": 0.9, "step": 0.01}),
                "pad_end": ("BOOLEAN", {"default": True}),
                "normalize_output": ("BOOLEAN", {"default": False})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, steps, start_value, end_value, bridge_iterations, convergence_rate, pad_end, normalize_output):
        device = "cuda"
        # Initialize with linear schedule
        t = torch.linspace(0, 1, steps, device=device)
        forward_schedule = start_value * (1 - t) + end_value * t
        backward_schedule = torch.flip(forward_schedule, [0])
        
        # Normalize backward schedule
        backward_schedule = start_value * (backward_schedule - backward_schedule.min()) / (backward_schedule.max() - backward_schedule.min())
        
        # Iteratively approximate Schrödinger bridge
        for _ in range(bridge_iterations):
            # Forward iteration of Schrödinger bridge
            forward_drift = -torch.log(forward_schedule[1:] / forward_schedule[:-1]) / (t[1:] - t[:-1])
            forward_drift = torch.cat([forward_drift[:1], forward_drift])  # Pad first element
            
            # Backward iteration
            backward_drift = torch.log(backward_schedule[1:] / backward_schedule[:-1]) / (t[1:] - t[:-1])
            backward_drift = torch.cat([backward_drift, backward_drift[-1:]])  # Pad last element
            
            # Average drifts (Schrödinger bridge principle)
            # The true optimal bridge is where forward and backward processes match
            avg_drift = 0.5 * (forward_drift - backward_drift)
            
            # Update schedules
            new_forward = torch.zeros_like(forward_schedule)
            new_forward[0] = start_value
            
            for i in range(1, steps):
                # Integrate the average drift
                dt = t[i] - t[i-1]
                new_forward[i] = new_forward[i-1] * torch.exp(-avg_drift[i-1] * dt)
            
            # Update with convergence rate
            forward_schedule = (1 - convergence_rate) * forward_schedule + convergence_rate * new_forward
            
            # Normalize to maintain endpoints
            forward_schedule = (forward_schedule - forward_schedule[-1]) / (forward_schedule[0] - forward_schedule[-1])
            forward_schedule = forward_schedule * (start_value - end_value) + end_value
            
            # Update backward schedule
            backward_schedule = torch.flip(forward_schedule, [0])
        
        # Final refinement: ensure monotonicity
        prev = forward_schedule[0]
        for i in range(1, steps):
            forward_schedule[i] = torch.minimum(prev, forward_schedule[i])
            prev = forward_schedule[i]
        
        # Normalize output if requested
        if normalize_output:
            forward_schedule = (forward_schedule - forward_schedule.min()) / (forward_schedule.max() - forward_schedule.min()) * (start_value - end_value) + end_value
            forward_schedule[0] = start_value
            forward_schedule[-1] = end_value
        
        # Add padding zero at the end if requested
        if pad_end:
            forward_schedule = torch.cat([forward_schedule, torch.tensor([0.0], device=device, dtype=forward_schedule.dtype)])
            
        return (forward_schedule,)


class sigmas_wave_packet:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "steps": ("INT", {"default": 30, "min": 5, "max": 100, "step": 1}),
                "start_value": ("FLOAT", {"default": 10.0, "min": 0.1, "max": 50.0, "step": 0.1}),
                "end_value": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 10.0, "step": 0.01}),
                "packet_shape": (["gaussian", "hermite", "laguerre", "coherent"], {"default": "gaussian"}),
                "quantum_parameter": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 2.0, "step": 0.01}),
                "pad_end": ("BOOLEAN", {"default": True}),
                "normalize_output": ("BOOLEAN", {"default": False})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, steps, start_value, end_value, packet_shape, quantum_parameter, pad_end, normalize_output):
        device = "cuda"
        # Time evolution parameter
        t = torch.linspace(0, 1, steps, device=device)
        
        if packet_shape == "gaussian":
            # Gaussian wave packet evolution
            # Width evolves according to quantum spreading
            width = torch.sqrt(1 + (quantum_parameter * t)**2)
            
            # Position evolves linearly
            position = 1 - t  # From 1 to 0
            
            # Quantum Gaussian wave packet probability density
            # |Ψ(x,t)|^2 = (1/sqrt(2π*width^2)) * exp(-(x-position)^2/(2*width^2))
            # We evaluate at x=0 (fixed reference point) as t evolves
            packet_density = torch.exp(-(position**2) / (2 * width**2)) / (torch.sqrt(2 * torch.tensor(math.pi, device=device)) * width)
            
            # Normalize
            packet_density = packet_density / packet_density.max()
            
            # Convert to sigma values (wave packet controls noise schedule)
            sigma_values = start_value * packet_density + end_value * (1 - packet_density)
            
        elif packet_shape == "hermite":
            # Hermite-Gaussian wave packet (quantum harmonic oscillator eigenstate)
            # The n=1 excited state has a node at the center
            
            # Position evolves according to harmonic oscillator
            position = torch.cos(quantum_parameter * torch.tensor(math.pi, device=device) * t)
            
            # First Hermite polynomial: H_1(x) = 2x
            hermite = 2 * position
            
            # Wavefunction: Ψ(x,t) = H_1(x) * exp(-x^2/2)
            # Probability: |Ψ|^2
            packet_density = (hermite**2) * torch.exp(-position**2)
            
            # Normalize
            packet_density = packet_density / packet_density.max()
            
            # Convert to sigma values
            sigma_values = start_value * packet_density + end_value * (1 - packet_density)
            
        elif packet_shape == "laguerre":
            # Laguerre-Gaussian beam profile
            # Used in quantum optics for orbital angular momentum states
            
            # Radial parameter
            radius = 1 - t  # From 1 to 0
            
            # Laguerre polynomial L^0_1(x) = 1 - x
            laguerre = 1 - (quantum_parameter * radius**2)
            
            # Probability density: |Ψ|^2 = L^2 * exp(-r^2)
            packet_density = (laguerre**2) * torch.exp(-radius**2)
            
            # Normalize
            packet_density = packet_density / packet_density.max()
            
            # Convert to sigma values
            sigma_values = start_value * packet_density + end_value * (1 - packet_density)
            
        elif packet_shape == "coherent":
            # Coherent state (quantum harmonic oscillator with classical-like behavior)
            # Position oscillates while maintaining minimum uncertainty
            
            # Time-evolving position
            phase = 2 * torch.tensor(math.pi, device=device) * t
            re_alpha = quantum_parameter * torch.cos(phase)
            im_alpha = quantum_parameter * torch.sin(phase)
            
            # Coherent state probability at origin: |⟨0|α⟩|^2 = exp(-|α|^2)
            alpha_squared = re_alpha**2 + im_alpha**2
            packet_density = torch.exp(-alpha_squared)
            
            # Normalize and invert (lower probability = higher sigma)
            packet_density = 1 - packet_density / packet_density.min()
            packet_density = packet_density / packet_density.max()
            
            # Convert to sigma values
            sigma_values = start_value * packet_density + end_value * (1 - packet_density)
        
        # Ensure endpoints are exactly as specified
        sigma_values[0] = start_value
        sigma_values[-1] = end_value
        
        # Ensure monotonicity
        for i in range(1, steps):
            sigma_values[i] = torch.minimum(sigma_values[i-1], sigma_values[i])
        
        # Normalize output if requested
        if normalize_output:
            sigma_values = (sigma_values - sigma_values.min()) / (sigma_values.max() - sigma_values.min()) * (start_value - end_value) + end_value
            sigma_values[0] = start_value
            sigma_values[-1] = end_value
        
        # Add padding zero at the end if requested
        if pad_end:
            sigma_values = torch.cat([sigma_values, torch.tensor([0.0], device=device, dtype=sigma_values.dtype)])
            
        return (sigma_values,)


class sigmas_quantum_annealing:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "steps": ("INT", {"default": 30, "min": 5, "max": 100, "step": 1}),
                "start_value": ("FLOAT", {"default": 10.0, "min": 0.1, "max": 50.0, "step": 0.1}),
                "end_value": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 10.0, "step": 0.01}),
                "annealing_profile": (["linear", "quadratic", "exponential", "adiabatic"], {"default": "adiabatic"}),
                "temperature": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "pad_end": ("BOOLEAN", {"default": True}),
                "normalize_output": ("BOOLEAN", {"default": False})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, steps, start_value, end_value, annealing_profile, temperature, pad_end, normalize_output):
        device = "cuda"
        # Create normalized time
        s = torch.linspace(0, 1, steps, device=device)
        
        # Initialize result
        sigma_values = torch.zeros(steps, device=device)
        
        if annealing_profile == "linear":
            # Linear annealing schedule
            annealing_factor = s
            
        elif annealing_profile == "quadratic":
            # Quadratic annealing (faster at the end)
            annealing_factor = s**2
            
        elif annealing_profile == "exponential":
            # Exponential annealing (rapid cooling)
            decay_rate = torch.tensor(5.0, device=device)  # Controls decay speed
            annealing_factor = (1 - torch.exp(-decay_rate * s)) / (1 - torch.exp(-decay_rate))
            
        elif annealing_profile == "adiabatic":
            # Quantum adiabatic annealing
            # Rate proportional to gap^2, which typically narrows near phase transition
            
            # Simulate phase transition in the middle
            gap = 0.1 + 0.9 * (1 - torch.exp(-(s - 0.5)**2 / (2 * 0.15**2)))
            
            # Calculate progress according to adiabatic principle
            # ds/dt ∝ gap^2 (can go faster when energy gap is large)
            velocity = gap**2
            
            # Integrate velocity to get position
            annealing_factor = torch.cumsum(velocity, dim=0)
            annealing_factor = annealing_factor / annealing_factor[-1]  # Normalize to [0,1]
            
        # Calculate transverse field strength A(s) and problem Hamiltonian strength B(s)
        # In quantum annealing: H(s) = (1-s)·A(s)·H_transverse + s·B(s)·H_problem
        A_s = 1 - annealing_factor
        B_s = annealing_factor
        
        # Quantum effect: add thermal fluctuations
        if temperature > 0:
            # Thermal fluctuations are stronger at high temperature and high transverse field
            noise_scale = temperature * A_s
            thermal_noise = noise_scale * torch.randn(steps, device=device)
            
            # Add noise but preserve monotonicity
            A_s = A_s + thermal_noise
            A_s = torch.clamp(A_s, min=0, max=1)
            
            # Ensure monotonicity of A_s (decreasing)
            for i in range(1, steps):
                A_s[i] = torch.minimum(A_s[i], A_s[i-1])
            
            # Recalculate B_s
            B_s = 1 - A_s
        
        # Convert to sigma values
        # Higher transverse field (A_s) = higher sigma (more noise)
        # Higher problem Hamiltonian (B_s) = lower sigma (less noise)
        sigma_values = start_value * A_s + end_value * B_s
        
        # Ensure endpoints match exactly
        sigma_values[0] = start_value
        sigma_values[-1] = end_value
        
        # Normalize output if requested
        if normalize_output:
            sigma_values = (sigma_values - sigma_values.min()) / (sigma_values.max() - sigma_values.min()) * (start_value - end_value) + end_value
            sigma_values[0] = start_value
            sigma_values[-1] = end_value
        
        # Add padding zero at the end if requested
        if pad_end:
            sigma_values = torch.cat([sigma_values, torch.tensor([0.0], device=device, dtype=sigma_values.dtype)])
            
        return (sigma_values,)


# Neural and adaptive methods

class sigmas_learned_path:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "steps": ("INT", {"default": 30, "min": 5, "max": 100, "step": 1}),
                "start_value": ("FLOAT", {"default": 10.0, "min": 0.1, "max": 50.0, "step": 0.1}),
                "end_value": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 10.0, "step": 0.01}),
                "network_type": (["cosine", "edm", "improved", "karras", "custom"], {"default": "improved"}),
                "custom_param": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "pad_end": ("BOOLEAN", {"default": True}),
                "normalize_output": ("BOOLEAN", {"default": False})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, steps, start_value, end_value, network_type, custom_param, pad_end, normalize_output):
        device = "cuda"
        # Time steps
        t = torch.linspace(0, 1, steps, device=device)
        
        if network_type == "cosine":
            # Cosine-based learned schedule (DDPM++, improved DDPM)
            # One of the most widely used learned schedules
            sigma_values = start_value * torch.cos(t * torch.pi/2) + end_value * (1 - torch.cos(t * torch.pi/2))
            
        elif network_type == "edm":
            # EDM (Elucidating Diffusion Models) learned schedule
            # log-normal distribution based
            sigma_max = start_value
            sigma_min = end_value
            rho = 7  # EDM paper used 7
            
            # EDM formulation (reparameterized for easier tuning)
            sigma_values = (sigma_max ** (1/rho) + t * (sigma_min ** (1/rho) - sigma_max ** (1/rho))) ** rho
            
        elif network_type == "improved":
            # Improved noise schedule - learned from DDPM training
            # S-shaped with slower decay in critical mid-range
            # Based on insights from training dynamics
            
            # Create sigmoid-like S-curve with controlled slope
            beta1 = torch.tensor(1e-4, device=device)
            beta2 = torch.tensor(0.02, device=device)
            
            # Betas schedule increasing from beta1 to beta2
            betas = torch.linspace(beta1, beta2, steps, device=device)
            
            # Apply learned corrections to focus on mid-range
            # Slower change in critical noise region
            modulation = 1 - torch.exp(-(t - 0.5)**2 / 0.1)
            betas = betas * modulation
            
            # Convert betas to sigmas
            alphas = 1 - betas
            alphas_cumprod = torch.cumprod(alphas, dim=0)
            sigmas_squared = (1 - alphas_cumprod) / alphas_cumprod
            sigma_values = torch.sqrt(sigmas_squared)
            
            # Scale to match desired range
            sigma_values = sigma_values / sigma_values[0] * start_value
            sigma_values[-1] = end_value  # Ensure exact end value
            
        elif network_type == "karras":
            # Karras et al. learned schedule
            # Optimized for specific step count ranges
            # Parameterized to match findings from extensive experimentation
            
            # Convert log-sigmas
            log_sigma_max = torch.log(torch.tensor(start_value, device=device))
            log_sigma_min = torch.log(torch.tensor(end_value, device=device))
            
            # Karras formula with learned parameters
            rho = 7  # Karras paper explores different values
            log_sigmas = log_sigma_min + (log_sigma_max - log_sigma_min) * (1 - t) ** rho
            sigma_values = torch.exp(log_sigmas)
            
        elif network_type == "custom":
            # Custom learned schedule based on research findings
            # Mixed schedule that adapts based on the custom parameter
            
            # Basic schedules
            linear = start_value * (1 - t) + end_value * t
            
            # Log-normal distribution
            log_start = torch.log(torch.tensor(start_value, device=device))
            log_end = torch.log(torch.tensor(end_value, device=device))
            lognormal = torch.exp(log_start * (1 - t) + log_end * t)
            
            # Cosine schedule
            cosine = start_value * torch.cos(t * torch.pi/2) + end_value * (1 - torch.cos(t * torch.pi/2))
            
            # Custom blend based on findings
            # Parameter controls balance between different schedules
            alpha = custom_param
            
            # When alpha is low, favor cosine early and lognormal late
            # When alpha is high, favor linear early and cosine late
            mix1 = (1 - t) * ((1 - alpha) * cosine + alpha * linear)
            mix2 = t * ((1 - alpha) * lognormal + alpha * cosine)
            
            sigma_values = mix1 + mix2
        
        # Ensure endpoints match exactly
        sigma_values[0] = start_value
        sigma_values[-1] = end_value
        
        # Normalize output if requested
        if normalize_output:
            sigma_values = (sigma_values - sigma_values.min()) / (sigma_values.max() - sigma_values.min()) * (start_value - end_value) + end_value
            sigma_values[0] = start_value
            sigma_values[-1] = end_value
        
        # Add padding zero at the end if requested
        if pad_end:
            sigma_values = torch.cat([sigma_values, torch.tensor([0.0], device=device, dtype=sigma_values.dtype)])
            
        return (sigma_values,)


class sigmas_transformer_guided:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "steps": ("INT", {"default": 30, "min": 5, "max": 100, "step": 1}),
                "start_value": ("FLOAT", {"default": 10.0, "min": 0.1, "max": 50.0, "step": 0.1}),
                "end_value": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 10.0, "step": 0.01}),
                "attention_heads": ("INT", {"default": 4, "min": 1, "max": 8, "step": 1}),
                "attention_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "pad_end": ("BOOLEAN", {"default": True}),
                "normalize_output": ("BOOLEAN", {"default": False})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, steps, start_value, end_value, attention_heads, attention_strength, pad_end, normalize_output):
        device = "cuda"
        # Start with a base schedule
        t = torch.linspace(0, 1, steps, device=device)
        base_schedule = start_value * (1 - t) + end_value * t
        
        # Simulate transformer attention mechanism
        
        # 1. Create positional encoding
        position = torch.arange(0, steps, device=device).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, attention_heads*2, 2, device=device) * (-math.log(10000.0) / (attention_heads*2)))
        pos_enc = torch.zeros(steps, attention_heads*2, device=device)
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        
        # 2. Create value embedding (base schedule normalized to [-1,1])
        values = 2 * (base_schedule - base_schedule.min()) / (base_schedule.max() - base_schedule.min()) - 1
        values = values.unsqueeze(1).repeat(1, attention_heads)  # [steps, heads]
        
        # 3. Compute attention scores (simplified self-attention)
        # Dot product of positional encodings
        pos_enc_reshaped = pos_enc.view(steps, attention_heads, 2)
        query = pos_enc_reshaped  # [steps, heads, 2]
        key = pos_enc_reshaped    # [steps, heads, 2]
        
        # Compute attention scores
        attn_scores = torch.bmm(
            query.transpose(0, 1),           # [heads, steps, 2]
            key.transpose(0, 1).transpose(1, 2)  # [heads, 2, steps]
        )  # [heads, steps, steps]
        
        # Scale scores
        attn_scores = attn_scores / math.sqrt(2)  # Scaling factor
        
        # Apply softmax for each position
        attn_probs = torch.softmax(attn_scores, dim=2)  # [heads, steps, steps]
        
        # 4. Apply attention to values
        weighted_values = torch.bmm(
            attn_probs,                       # [heads, steps, steps]
            values.transpose(0, 1)            # [heads, steps]
        )  # [heads, steps]
        
        # 5. Combine attention outputs across heads
        attn_output = weighted_values.mean(dim=0)  # [steps]
        
        # 6. Convert back to sigma range
        attn_output = (attn_output + 1) / 2  # Back to [0,1]
        attn_schedule = start_value * (1 - attn_output) + end_value * attn_output
        
        # 7. Blend with base schedule according to attention strength
        sigma_values = (1 - attention_strength) * base_schedule + attention_strength * attn_schedule
        
        # Ensure monotonicity
        for i in range(1, steps):
            sigma_values[i] = torch.minimum(sigma_values[i-1], sigma_values[i])
            
        # Ensure endpoints match exactly
        sigma_values[0] = start_value
        sigma_values[-1] = end_value
        
        # Normalize output if requested
        if normalize_output:
            sigma_values = (sigma_values - sigma_values.min()) / (sigma_values.max() - sigma_values.min()) * (start_value - end_value) + end_value
            sigma_values[0] = start_value
            sigma_values[-1] = end_value
        
        # Add padding zero at the end if requested
        if pad_end:
            sigma_values = torch.cat([sigma_values, torch.tensor([0.0], device=device, dtype=sigma_values.dtype)])
            
        return (sigma_values,)


class sigmas_meta_learned:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "steps": ("INT", {"default": 30, "min": 5, "max": 100, "step": 1}),
                "start_value": ("FLOAT", {"default": 10.0, "min": 0.1, "max": 50.0, "step": 0.1}),
                "end_value": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 10.0, "step": 0.01}),
                "style": (["general", "photography", "illustration", "faces", "landscapes"], {"default": "general"}),
                "adaptation_strength": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
                "pad_end": ("BOOLEAN", {"default": True}),
                "normalize_output": ("BOOLEAN", {"default": False})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, steps, start_value, end_value, style, adaptation_strength, pad_end, normalize_output):
        device = "cuda"
        # Initialize result
        t = torch.linspace(0, 1, steps, device=device)
        
        # Base learned schedule
        # Improved noise schedule from large-scale training
        base_powers = torch.linspace(1, 3, steps, device=device)
        base_decay = torch.pow(1 - t, base_powers)
        base_schedule = start_value * base_decay + end_value * (1 - base_decay)
        
        # Meta-learned adjustments for different domains
        if style == "general":
            # Balanced schedule good for general content
            # This is our default schedule
            adjustment = torch.zeros_like(t)
            
        elif style == "photography":
            # Photographic content needs more steps in the mid-noise region
            # where texture and detail are refined
            mid_boost = torch.exp(-(t - 0.3)**2 / 0.1) * 0.15
            adjustment = -mid_boost  # Negative means more noise (slower reduction)
            
        elif style == "illustration":
            # Illustration benefits from slower noise reduction early
            # to establish basic forms, then faster at the end
            early_slow = torch.exp(-(t - 0.15)**2 / 0.1) * 0.2
            adjustment = -early_slow
            
        elif style == "faces":
            # Faces are sensitive to noise schedule in middle-to-late steps
            # where facial features are refined
            face_features = torch.exp(-(t - 0.7)**2 / 0.1) * 0.1
            adjustment = -face_features
            
        elif style == "landscapes":
            # Landscapes benefit from distinct noise treatment at different scales
            # Early for large structures, mid for medium details, late for fine texture
            large_structures = torch.exp(-(t - 0.2)**2 / 0.05) * 0.15
            medium_details = torch.exp(-(t - 0.5)**2 / 0.05) * 0.1
            fine_texture = torch.exp(-(t - 0.8)**2 / 0.05) * 0.05
            adjustment = -(large_structures + medium_details + fine_texture)
        
        # Apply style-specific adjustment modulated by adaptation strength
        adaptive_adjustment = adjustment * adaptation_strength
        
        # Compute log-domain adjustment for better scaling
        log_base = torch.log(base_schedule)
        log_adjusted = log_base + adaptive_adjustment
        adjusted_schedule = torch.exp(log_adjusted)
        
        # Ensure monotonicity
        for i in range(1, steps):
            adjusted_schedule[i] = torch.minimum(adjusted_schedule[i-1], adjusted_schedule[i])
            
        # Ensure endpoints match exactly
        adjusted_schedule[0] = start_value
        adjusted_schedule[-1] = end_value
        
        # Normalize output if requested
        if normalize_output:
            adjusted_schedule = (adjusted_schedule - adjusted_schedule.min()) / (adjusted_schedule.max() - adjusted_schedule.min()) * (start_value - end_value) + end_value
            adjusted_schedule[0] = start_value
            adjusted_schedule[-1] = end_value
        
        # Add padding zero at the end if requested
        if pad_end:
            adjusted_schedule = torch.cat([adjusted_schedule, torch.tensor([0.0], device=device, dtype=adjusted_schedule.dtype)])
            
        return (adjusted_schedule,)


# KL-Optimal Scheduler 

class sigmas_kl_optimal:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "steps": ("INT", {"default": 30, "min": 5, "max": 1000, "step": 1}),
                "start_value": ("FLOAT", {"default": 10.0, "min": 0.1, "max": 50.0, "step": 0.1}),
                "end_value": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 10.0, "step": 0.01}),
                "schedule_type": (["standard", "mutual_information", "elbo_optimal"], {"default": "elbo_optimal"}),
                "beta": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0, "step": 0.01}),
                "pad_end": ("BOOLEAN", {"default": True}),
                "normalize_output": ("BOOLEAN", {"default": False})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, steps, start_value, end_value, schedule_type, beta, pad_end, normalize_output):
        device = "cuda"
        # Create empty array for result
        result = torch.zeros(steps, device=device, dtype=torch.float32)
        
        if schedule_type == "standard":
            # Standard KL-optimal schedule based on diffusion variance
            # This schedule concentrates more steps where the KL divergence changes quickly
            t = torch.linspace(0, 1, steps, device=device)
            exp_t = torch.exp(-torch.tensor(beta, device=device) * t)
            
            # Variance schedule that's optimized for ELBO
            variance = start_value * exp_t + end_value * (1 - exp_t)
            
            # Convert variance to sigma (standard deviation)
            result = torch.sqrt(variance)
            
        elif schedule_type == "mutual_information":
            # Schedule optimized for mutual information across diffusion steps
            # This tends to add more steps at the beginning and end of diffusion
            alphas = torch.linspace(0, 1, steps, device=device)
            
            # Create informational-theoretic optimal schedule
            # Formula derived from optimizing mutual information between adjacent steps
            result[0] = start_value
            for i in range(1, steps):
                t = i / (steps - 1)
                # Information-theoretic rate according to mutual information objective
                mi_rate = torch.sqrt(torch.tensor(beta, device=device) * (1 - torch.exp(-2 * beta * t)))
                result[i] = result[i-1] * torch.exp(-mi_rate / steps)
            
            # Rescale to match desired range
            result = (result - result[-1]) / (result[0] - result[-1]) * (start_value - end_value) + end_value
            
        elif schedule_type == "elbo_optimal":
            # Optimal schedule for ELBO (Evidence Lower Bound)
            # This is the most theoretically justified KL-optimal schedule
            
            # Create time points
            t = torch.linspace(0, 1, steps, device=device)
            
            # Discretized KL optimal schedule based on squared cosine
            # This approximates the true KL-optimal continuous schedule
            # Formula based on improved DDPM paper and follow-up work
            alpha_t = torch.cos((t + beta) / (1 + beta) * torch.tensor(math.pi/2, device=device)) ** 2
            alpha_t = alpha_t / alpha_t[0]  # Normalize
            
            # Convert to sigma
            sigma_t = torch.sqrt((1 - alpha_t) / alpha_t)
            
            # Scale to desired range
            result = sigma_t * start_value
            result[-1] = end_value  # Ensure exact end value
        
        # Normalize output if requested
        if normalize_output:
            result = (result - result.min()) / (result.max() - result.min()) * (start_value - end_value) + end_value
            result[0] = start_value
            result[-1] = end_value
        
        # Add padding zero at the end if requested
        if pad_end:
            result = torch.cat([result, torch.tensor([0.0], device=device, dtype=result.dtype)])
            
        return (result,)

        




