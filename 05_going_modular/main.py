from going_modular import data_setup, model_builder
import torch

torch.manual_seed(42)

train_data, test_data, classes = data_setup.create_dataloaders(

                                  )

device = "cuda" if torch.cuda.is_available() else "cpu"

# intantiate model
model = model_builder.TinyVGG(input_shape=3,
                              hidden_units=10,
                              output_shape=len(classes)).to(device)