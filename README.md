VAE_Encder -> main purpose to decrease the number of featurres but increase the number of features 


VAE_ResidualBlock -> this is block that consists of Normalizations and convalution and is used in VAE_Encoder


Our models main architecture consists of the encoder decoder variational autoencoder for the images 

also the text encoder tho the embeding space (Clip encoder) (this goes into the U-net model)


IN our U-net we can see very big similarity with the encoder and the decoder , the bedinging of the U-net is very much similar to the 
encoder and the second half to the decoder, becouse we are reduceing the size of the image3 at the begining and then extending it
the main key difference is the fact that in the U-net we are also using the prompt embediding to thell what we want to get as a result in the end 
And the best way to combine the image and prompt is by using CROSS ATTENTION

becouse we need to give the u-net not only the image but also the time step at which it waws noisified , thats why we will pass in the time step as a time embedding 


What we are going to do is take the noise, text and run them through the unet with accordance to time schedualer 


in the inference we can see how the pipeline is built and while building it we can see how the schadualer works


Classifier Free Guidance (Combine output)

output = w * (output_conditioned - output_unconditioned) + output_unconditioned

so we will inference form the model twise one with the prompt and one without


for example in the text - to - image 
we find the nose of the image with the help of the UNET, so basically we run our model throu the unet , it will detect the noise, and what schedualer does is it denoises the image, and every time we pass it to the schedualer we kinda ask it how much noise is in there and then we pass the image again to the begening of the unet, and again it detects the noise and pase it back to the schedualer and so on ....
untill we finish this time steps , then once we finish we give it to the latent and then to the decoder
see pipline.py