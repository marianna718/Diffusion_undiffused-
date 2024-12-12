from clip import CLIP
from encoder import VAE_Encoder
from decoder import VAE_Decoder
from diffusion import Diffussion
import model_convertor

def preload_models_from_standard_weights(ckpt_path, device):
    state_dict = model_convertor.load_from_standard_weights(ckpt_path, device)
    # the funcion in top loads the pretrained model

    # transfer model to the device used
    encoder = VAE_Encoder().to(device)
    # load the parameters for that model
    encoder.load_state_dict(state_dict["encoder"], strict=True)

    decoder = VAE_Decoder().to(device)
    decoder.load_state_dict(state_dict["decoder"], strict=True)

    diffusion = Diffussion().to(device)
    diffusion.load_state_dict(state_dict["diffusion"], strict=True)

    clip = CLIP().to(device)
    clip.load_state_dict(state_dict["clip"], strict=True)

    return {
        "clip" : clip,
        "encoder" : encoder,
        "decoder" : decoder,
        "diffusion"  :diffusion
    }