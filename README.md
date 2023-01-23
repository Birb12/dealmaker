# Welcome to Dealmaker!

### Dealmaker is an collection of NLP models that can generate descriptions of deals given an input of a product, its discount, and its features.
### A list of the models is below:

* GPT-Neo-1.3B - A custom-trained, open source model with 1.3B parameters. A demo can be found [here](https://huggingface.co/IanA/GPTNeo-Dealmaker). The trained model's weights, along with its tokenizer, can be downloaded in the "Files" tab. To play around with the model, type in your input in the Hosted Inference API. If you wish to make any changes to this model, please clone the repository and use your own dataset. The model can then be trained with train.py. Keep in mind that the corresponding dataset class was created with a specific dataset in mind, and will have to be modified to suit your specific needs.
* GPT-3-DaVinci - The cream of the crop. Utilizing OpenAI's API, this is the most powerful model and, consequently, yields the best results. A demo can be found [here](https://52d423a3-f63b-4395.gradio.live/). For commercial use, please take a look at the dataset manager in the GPT-3-DaVinci folder and change it to fit your needs. You may also look into the create_finetuned_model method to further increase the model's accuracy.

## All changes that need to be made, if you wish to use your own personalized version of the models instead of the demos, are highlighted with the comment FIXME
