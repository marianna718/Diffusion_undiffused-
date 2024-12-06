import torch 
import numpy as np
from tqdm import tqdm
from ddpm import DDPMSampler

WIDTH = 512
HEIGHT = 512
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8

# the uncond_prompt can also been noticed as the negative prompt
# thats when we wont cat images, but not cat on the sofa
# the negative prompt wll be sofa
def generate(prompt: str, uncond_prompt:str, input_image=None,
             strenght = 0.8, do_cfg = True, cfg_scale=7.5, sampler_name ="ddpm", n_inference_steps=50,
             mosel={}, seed =None, device = None, idle_device = None, tokenizer=None):
    # first we disable for infeance 
    with torch.no_grad():
        if not (0 < strenght < 1):
            raise ValueError("Strenght must be between 0 and 1")
        
        # createing condition for CPU
        if idle_device:
            to_idle: lambda x: x.to(idle_device)
        else:
            to_idle: lambda x: x

        # to genarate the noise we will use random number generatot
        generator = torch.Generator(device=device)
        if seed is None:  #change this line to !=
            generate.seed()
        else:
            generator.manual_seed()

        

        # taking from pretrained models the one for CLIP 
        clip = models["clip"]
        clip.to(device)

        if do_cfg:
            # Conver the prompt into tokens using the tokenizer
            cond_tokens = tokenizer.batch_encode_plus([prompt], padding="max_length", max_length=77).input_ids
            # (Batch_size, seq_len)
            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
            # (Batch_size, seq_len) -> (Bath_size, seq_len, Dim)
            cond_context = clip(cond_tokens)

            # now we will do the same with unconditional_tokens
            uncond_tokens = tokenizer.batch_encode_plus([uncond_prompt], padding="max_length", max_length=77).input_ids
            uncond_tokens = torch.tensor(uncond_tokens, dtype= torch.long, device = device)
            # (Batch_size, seq_len) -> (batch_size, seq_len, dim)
            uncond_context = clip(uncond_tokens)


            # now we will concatinate this two prompts
            # (2, seq_len, dim) -> (2,77, 768)
            # here its 2 becouse we are runing through the model 2 prompts
            # one conditional and one unconditionsal

            context = torch.cat([cond_context, uncond_context])
        else:
            # convert it into a list of tokens
            tokens = tokenizer.batch_encode_plus([prompt], padding="max_length", max_length=77).input_ids
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)
            # (1,77,768)
            context = clip(tokens)
        to_idle(clip)

        # then we load the sampler
        if sampler_name == "ddpm":
            sampler = DDPMSampler(generator)
            # then we tell the sampler how many steps we need
            # becuouse we need to tell the sampler how many steps we need 
            sampler.set_inference_steps(n_inference_steps)

        else:
            raise ValueError(f"Unknown sampler {sampler_name}")
        
        latents_shape = (1,4, LATENTS_HEIGHT, LATENTS_WIDTH)


        # now lets see what happens if the user specifies an input image

        if input_image:
            encoder = models["encoder"]
            encoder.to(device)

            # now we need to reshape it to make shure it feets the input size
            # 512x512
            input_image_tensor = input_image.resize((WIDTH, HEIGHT))
            input_image_tensor = np.array(input_image_tensor)

            # (height, width, channel)
            input_image_tensor = torch.tensor(input_image_tensor, dtype= torch.float32)

            # now we need to rescale the numbers in the 3 rgb layers to the -1,1 range
            input_image_tensor = rescale(input_image_tensor, (0,255) (-1,1))

            # we add the batch dimmention 
            # (height, width, channel) -> (batch_size, height, width, channel)
            input_image_tensor = input_image_tensor.unsqueeze(0)
            # and then we change the order of the dimmentions
            input_image_tensor = input_image_tensor.permute(0, 3, 1,2)
            # this is becouse the encoder of VAE takes this size

            # no we will sample some noise
            # the more noise we add the more creative the model can be 
            # the strenght feature defines how much noise we want to add
            encoder_noise = torch.randn(latents_shape, generator=generator, device=device)

            # run the image through the encoder of the VAE
            latents = encoder(input_image_tensor, encoder_noise)

            # by setting the strenght we will set time step schedualer
            # and with level of noise to start
            sampler.set_strength(strenght= strenght)
            latents =  sampler.add_noise(latents, sampler.timesteps[0])

            to_idle(encoder)
        else:
            # if we are doing text-to-image strt with random noise N(0,1)
            latents = torch.randn(latents, generator=generator, device=device)

        diffusion = models["diffusion"]
        diffusion.to(device)

        # and then our sampler will define the time steps 
        # in training we have like 1000 but in inference 15  
        # each time step in som emeaning acctually defines a 
        # denoising level

        # Loop of denoiseing
        timesteps = tqdm(sampler.timesteps)
        for i, timestep in enumerate(timesteps):
            # (1,320)
            time_embedding = get_time_embedding(timestep).to(device)

            # (Batch_size, 4, latent_height, latent_width)
            model_input = latents

            if do_cfg:  #classifier free guidance
                # (batch_size,4,latents_height, latents_width) -> (2* batch_size, 4, latentt_height, latent_width)
                model_input = model_input.repeat(2,1,1,1)

            # model_output is the predicted noise by the UNEt
            model_output = diffusion(model_input, context, time_embedding)



            if do_cfg:
                # and if we are doing cfd we need to pass conditional output and unconditional output
                # becouse we are doing cfg, from previous statetment we can see
                # that the batch size is 2, so we can split it into 2 tenosrs
                output_cond, output_uncond = model_output.chunk(2)
                model_output = cfg_scale * (output_cond - output_uncond) +output_uncond

            # Remove noise predicted by the UNET
            latents = sampler.step(timestep, latents, model_output)


        to_idle(diffusion)

        decoder = models["deocoder"]
        decoder.to(device)

        images = decoder(latents)
        to_idle(decoder)

        # rescaleing the image back
        images = rescale(images, (-1,1), (0,255), clamp = True)

        # (Batch_size, channel, height , width) -> (batch_size, height, width, channels )
        images = images.permute(0,2,3,1)
        images = images.to("cpu", torch.units8).numpy()
        return images
    



def rescale(x, old_range, new_range, clamp = False):
    old_min, old_max = old_range
    new_min, new_max = new_range

    x -= old_min
    x*= (new_max-new_min)/ (old_max - old_min)
    x += new_min
    if clamp:
        x = x.clamp(new_min, new_max)
    return x


def get_time_embedding(timestep):
    # we will firs define the cosine and sine of our sequances
    # (160,)
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32)/160)
    # (1,160)
    x = torch.tensor([timestep], dtype=torch.float32)[:,None] * freqs[None]
    # (1,320)
    return torch.cat([torch.cos(x), torch.sin(x)], dim =-1)




