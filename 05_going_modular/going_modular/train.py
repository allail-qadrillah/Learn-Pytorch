"""
Trains a Pytorch image classificaction odel using device-agnostic code.
"""
import data_setup, engine, model_builder, utils
import torch
import os

from torchvision import transforms

# setup hyperparameters
NUM_EPOCHS = 3
BATCH_SIZE = 32
HIDDEN_UNITS = 10
LEARNING_RATE = 0.001

# setup directories
train_dir = "../data/pizza_steak_sushi/train"
test_dir = "../data/pizza_steak_sushi/test"

# setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"

# create transforms
data_transform = transforms.Compose([
  transforms.Resize((64,64)),
  transforms.ToTensor()
])

# create DataLoaders 
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
  train_dir=train_dir,
  test_dir=test_dir,
  transform=data_transform,
  batch_size=BATCH_SIZE
)

# create model
model = model_builder.TinyVGG(input_shape=3,
                              hidden_units=10,
                              output_shape=len(class_names)).to(device)

# set loss and optimizer
loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(),
                             lr=LEARNING_RATE)

# start training 
if __name__ == '__main__':
  # from multiprocessing import freeze_support
  # freeze_support()  # Run code for process object if this in not the main process

  engine.train(model=model,
              train_dataloader=train_dataloader,
              test_dataloader=test_dataloader,
              optimizer=optimizer,
              loss_fn=loss,
              epochs=NUM_EPOCHS,
              device=device)

  # save model
  utils.save_model(model=model,
                  target_dir="../models",
                  model_name="05_going_modular_script_mode_tinyvgg_model.pth")
